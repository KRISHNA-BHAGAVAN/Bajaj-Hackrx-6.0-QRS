# llm_services.py
import random
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from openai import RateLimitError, APIError, OpenAIError

from config import RAG_LLM, GEMINI_API_KEYS, get_gemini_llm, LLM_STREAMING_ENABLED
from prompt_template import TEMPLATE

# --- Prompt Template ---
prompt = ChatPromptTemplate.from_template(TEMPLATE)

# --- Fallback LLM Chain ---
async def fallback_llm_chain(inputs: dict) -> str:
    """
    Primary chain with a fallback mechanism from OpenAI to Gemini models.
    Tries OpenAI first, then iterates through available Gemini keys on failure.
    """
    try:
        # Primary chain using OpenAI
        chain = prompt | RAG_LLM | StrOutputParser()
        return await chain.ainvoke(inputs)
    except (RateLimitError, APIError) as e:
        print(f"⚠️ OpenAI API error ({type(e).__name__}), trying Gemini keys...")
        if not GEMINI_API_KEYS:
            raise
        # Shuffle keys to distribute load in case of multiple failures
        keys_shuffled = random.sample(GEMINI_API_KEYS, len(GEMINI_API_KEYS))
        for key in keys_shuffled:
            try:
                gemini_llm = get_gemini_llm(key)
                chain = prompt | gemini_llm | StrOutputParser()
                return await chain.ainvoke(inputs)
            except Exception:
                # Try the next key if one fails
                continue
        raise
    except OpenAIError:
        raise

# Main RAG chain runnable with the defined fallback logic
rag_chain = RunnableLambda(fallback_llm_chain).with_config(run_name="rag_chain_with_fallback")

# --- Streaming Wrapper ---
async def stream_rag_chain(inputs: dict):
    """
    A wrapper to handle streaming output from the language model.
    If streaming is enabled and supported, it yields content chunks.
    Otherwise, it falls back to a single invocation.
    """
    try:
        formatted_messages = prompt.format_messages(**inputs)

        if LLM_STREAMING_ENABLED and hasattr(RAG_LLM, "astream"):
            async for chunk in RAG_LLM.astream(formatted_messages):
                yield chunk.content if hasattr(chunk, "content") else str(chunk)
            return
        
        # Fallback for non-streaming cases
        yield await rag_chain.ainvoke(inputs)
    except Exception as e:
        yield f"⚠️ Streaming failed: {e}"