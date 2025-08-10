# graph_builder.py
import os
import faiss
import asyncio
from uuid import uuid4
from typing import List, TypedDict, Annotated, Literal

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# Local module imports
from data_processing import (
    get_docs_from_url, get_chunks, url_to_cache_path, embed_batches_concurrently
)
from llm_services import stream_rag_chain
from react_agent import reasoning_agent
from utils import contains_api_or_url
from config import (
    embeddings, QA_CONCURRENCY, EMBED_BATCH_SIZE, EMBED_BATCH_API_AVAILABLE
)

# --- Agent State Definition ---
class AgentState(TypedDict):
    doc_url: str
    questions: List[str]
    cache_path: str
    retriever: FAISS
    answers: List[str]
    current_question_index: int
    error_message: str
    messages: Annotated[list[AnyMessage], add_messages]
    web_content: str
    initial_context: str | None

# --- Graph Node Implementations ---
def initialize_processing(state: AgentState) -> AgentState:
    """Initializes state variables for a new run."""
    return {
        "cache_path": url_to_cache_path(state['doc_url']),
        "answers": [],
        "current_question_index": 0,
        "error_message": None,
        "messages": [],
        "web_content": ""
    }

async def perform_reasoning(state: AgentState) -> AgentState:
    """Node that executes the ReAct reasoning agent."""
    return {"answers": await reasoning_agent(state["doc_url"], state["questions"])}

def check_for_api_context(state: AgentState) -> AgentState:
    """Node that checks the initial document for API-related keywords."""
    docs = state["retriever"].invoke("Api endpoints and urls. search for http")
    return {"initial_context": "\n\n".join(d.page_content for d in docs)}

def route_after_context_check(state: AgentState) -> Literal["perform_reasoning", "generate_all_answers"]:
    """Conditional edge to decide between the ReAct agent and standard RAG."""
    return "perform_reasoning" if contains_api_or_url(state.get("initial_context", "")) else "generate_all_answers"

def check_cache(state: AgentState) -> str:
    """Conditional edge to check if a cached FAISS index exists."""
    if "retriever" in state and state["retriever"] is not None:
        print("⚡ Using in-memory retriever (no FAISS reload).")
        return "process_document_flow"
    return "load_from_cache" if os.path.exists(state['cache_path']) else "process_document_flow"

def load_from_cache(state: AgentState) -> AgentState:
    """Node to load a FAISS index from the local cache."""
    vs = FAISS.load_local(state['cache_path'], embeddings, allow_dangerous_deserialization=True)
    return {"retriever": vs.as_retriever(search_type="mmr", search_kwargs={'k': 15})}

def process_document(state: AgentState) -> AgentState:
    """Node to process, chunk, and embed a document into a FAISS index."""
    docs = get_docs_from_url(state['doc_url'])
    if not docs:
        return {"error_message": "Failed to download/parse document."}
    
    chunks = asyncio.run(get_chunks(docs, state['doc_url']))
    if not chunks:
        return {"error_message": "Document chunks are empty."}

    sample_vec = embeddings.embed_query("hello world")
    index = faiss.IndexFlatL2(len(sample_vec))
    vs = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})

    if EMBED_BATCH_API_AVAILABLE:
        try:
            asyncio.run(embed_batches_concurrently(vs, chunks, EMBED_BATCH_SIZE))
        except Exception as e:
            print(f"⚠️ Concurrent embedding failed: {e}. Falling back to sequential embedding.")
            vs.add_documents(documents=chunks, ids=[str(uuid4()) for _ in chunks])
    else:
        vs.add_documents(documents=chunks, ids=[str(uuid4()) for _ in chunks])

    vs.save_local(state['cache_path'])
    return {"retriever": vs.as_retriever(search_type="mmr", search_kwargs={'k': 15})}

async def generate_all_answers(state: AgentState) -> AgentState:
    """Node to generate answers for all questions in parallel."""
    if state.get('error_message'):
        return {"answers": ["Could not generate answers due to an earlier error."]}

    sem = asyncio.Semaphore(QA_CONCURRENCY)

    async def _fetch_context(idx: int, q: str):
        if idx == 0 and state.get('initial_context'):
            return idx, state['initial_context']
        docs = await asyncio.to_thread(state['retriever'].invoke, q)
        return idx, "\n\n".join(d.page_content for d in docs)

    contexts = sorted(await asyncio.gather(*[_fetch_context(i, q) for i, q in enumerate(state['questions'])]), key=lambda x: x[0])
    
    async def _answer_one(idx: int, q: str, ctx: str):
        async with sem:
            inputs = {"context": ctx, "question": q}
            parts = [chunk async for chunk in stream_rag_chain(inputs)]
            final = "".join(parts).strip() or "⚠️ Empty answer from LLM."
            print(f"🧠 Answered question {idx+1}")
            return idx, final
    
    results = sorted(await asyncio.gather(*[_answer_one(i, q, c[1]) for i, (q, c) in enumerate(zip(state['questions'], contexts))]), key=lambda x: x[0])
    return {"answers": [r[1] for r in results], "initial_context": None}


# --- Graph Construction ---
workflow = StateGraph(AgentState)
workflow.add_node("initialize", initialize_processing)
workflow.add_node("check_cache_node", lambda state: {})
workflow.add_node("load_from_cache", load_from_cache)
workflow.add_node("check_for_api_context", check_for_api_context)
workflow.add_node("perform_reasoning", perform_reasoning)
workflow.add_node("process_document_flow", process_document)
workflow.add_node("generate_all_answers", generate_all_answers)

workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "check_cache_node")
workflow.add_edge("load_from_cache", "check_for_api_context")
workflow.add_edge("process_document_flow", "check_for_api_context")
workflow.add_conditional_edges("check_for_api_context", route_after_context_check, {
    "perform_reasoning": "perform_reasoning",
    "generate_all_answers": "generate_all_answers"
})
workflow.add_conditional_edges("check_cache_node", check_cache, {
    "load_from_cache": "load_from_cache",
    "process_document_flow": "process_document_flow"
})
workflow.add_edge("perform_reasoning", END)
workflow.add_edge("generate_all_answers", END)

# Compiled graph, ready for use
jarvis = workflow.compile()
# Generate the PNG bytes
png_data = jarvis.get_graph().draw_mermaid_png()

# Save PNG to file
with open("workflow.png", "wb") as f:
    f.write(png_data)

print("Workflow image saved as langgraph_workflow.png")