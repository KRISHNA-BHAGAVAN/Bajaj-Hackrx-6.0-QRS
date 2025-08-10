import os
import asyncio
import aiohttp
import traceback
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# LangChain and LangGraph imports - using the latest package structures
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

# LLM imports
from langchain_openai import ChatOpenAI

load_dotenv(override=True)


llm = ChatOpenAI(api_key=os.getenv("GROQ_API_KEY"),
                 model_name="openai/gpt-oss-20b",
                 temperature=0,
                 base_url="https://api.groq.com/openai/v1/",
                 streaming=True)

# llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),temperature=0, streaming=True)


# --- Agent Tool Definition---
@tool
async def web_scraper_tool(url: str) -> str:
    """
    Fetches complete text content from a web URL, API endpoint, or a PDF document asynchronously.
    Use this single tool to get initial instructions from a PDF and to fetch data from any other external web source as needed.
    """
    print(f"\n--- TOOL: Async Web Scraper ---")
    print(f"Fetching content from: {url}")

    is_pdf = url.split('?')[0].lower().endswith('.pdf')

    if is_pdf:
        print("Detected PDF content.")
        try:
            loader = PyMuPDF4LLMLoader(url)
            docs = await loader.aload()
            
            if not docs:
                return "Successfully connected to the PDF URL, but no content could be loaded."
            
            full_text = "\n\n".join(doc.page_content for doc in docs)
            return full_text if full_text else "Successfully loaded the PDF, but no text was found."
        except Exception as e:
            traceback.print_exc()
            return f"An error occurred while loading or parsing the PDF: {e}"

    try:
        # Set a timeout to prevent the agent from hanging on unresponsive sites.
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
                response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)

                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    print("Detected JSON content.")
                    print(str(await response.json()))
                    return str(await response.json())
                else:
                    print("Detected HTML/Text content.")
                    html_text = await response.text()
                    soup = BeautifulSoup(html_text, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                    return text if text else "Successfully fetched URL, but no text content was found."

    except aiohttp.ClientError as e:
        print(f"Error fetching URL with aiohttp: {e}")
        return f"Error: Could not fetch the URL. A network error occurred: {e}"
    except asyncio.TimeoutError:
        print(f"Error: Timeout while trying to fetch URL: {url}")
        return f"Error: The request to the URL timed out after 15 seconds."
    except Exception as e:
        traceback.print_exc()
        return f"An unexpected error occurred during web scraping: {e}"


# --- Agent Definition (Prompt Reliability Improvement) ---
AGENT_SYSTEM_PROMPT = """You are a highly specialized reasoning agent. Your mission is to follow instructions from a given document to find a final answer.

**Your Workflow:**
1.  **Analyze the Mission:** Carefully read the 'INSTRUCTION DOCUMENT CONTENT' provided by the user. Understand the overall objective and all the required steps.
2.  **Execute Step-by-Step:** Follow the instructions precisely. For each step that requires fetching external data from a URL, you MUST use the `web_scraper_tool`.
3.  **Reason and Combine:** Analyze the information you've gathered. Think about how it helps you complete the current step and move to the next one.
4.  **Conclude:** Once you have followed all steps and found the final piece of information required by the 'USER QUERY', state the answer clearly and concisely. Your final output should only be the answer itself.

**Critical Instructions:**
- Do not guess or hallucinate. Rely *only* on the information provided in the instructions and gathered from your tool.
- You have one tool: `web_scraper_tool`. Use it to read the content of any URL.
- Announce when you are moving to a new step, e.g., "Now proceeding to Step 2."
"""

async def reasoning_agent(url:str, query: List[str]) -> List[str]:
    """
    Initializes and runs the reasoning agent with improved performance and reliability.
    """
    tools = [web_scraper_tool]

    print(f"--- Pre-fetching instruction document from: {url} ---")
    # Ainvoke the async tool to pre-fetch the instructions without blocking.
    instruction_content = await web_scraper_tool.ainvoke({"url": url})
    
    if "Error:" in instruction_content or "failed" in instruction_content:
        error_message = f"Failed to load the initial instruction document. Aborting. Reason: {instruction_content}"
        print(error_message)
        return [error_message]

    print(f"--- Instructions Loaded Successfully ---")

    agent_executor = create_react_agent(llm, tools)

    print(f"{'='*20} Agent Initialized. Starting Task. {'='*20}\n")
    
    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(
            content=f"""
                ### INSTRUCTION DOCUMENT CONTENT:
                ---
                {instruction_content}
                ---

                ### USER QUERY:
                {query}

                Please execute the mission based on the instructions above to answer the user's query.
                """
        )
    ]
    
    result = await agent_executor.ainvoke({"messages": messages})
    return [result["messages"][-1].content]

#Testing
async def main():
    url = "https://hackrx.blob.core.windows.net/hackrx/rounds/FinalRound4SubmissionPDF.pdf?sv=2023-01-03&spr=https&st=2025-08-07T14%3A23%3A48Z&se=2027-08-08T14%3A23%3A00Z&sr=b&sp=r&sig=nMtZ2x9aBvz%2FPjRWboEOZIGB%2FaGfNf5TfBOrhGqSv4M%3D"
    query = "what is my flight number?"

    print(f"{'='*20} Initializing Reasoning Agent {'='*20}")
    
    tools = [web_scraper_tool]

    agent_executor = create_react_agent(llm, tools)

    print(f"{'='*20} Agent Initialized. Starting Task. {'='*20}\n")
    
    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(
            content=f"The instruction document is loaded from '{url}'. Please execute the mission: {query}"
        )
    ]

    # We invoke the agent with the full list of messages.
    async for event in agent_executor.astream_events({"messages": messages}, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
        elif kind == "on_tool_end":
            # This formatting helps clearly distinguish tool output in the console.
            print(f"\n\n<tool_code>")
            print(f"Tool: {event['name']}")
            print(f"</tool_code>\n<tool_output>")
            print(event['data'].get('output'))
            print("</tool_output>\n")
        elif kind == "on_chat_model_end":
            print("\n")

    print(f"\n\n{'='*20} Task Finished {'='*20}")

if __name__ == "__main__":
    asyncio.run(main())
    