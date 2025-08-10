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
# In llm_services.py

async def stream_rag_chain(inputs: dict):
    """
    A wrapper to handle streaming output with a fallback mechanism.
    """
    formatted_messages = await prompt.ainvoke(inputs)

    # 1. Try streaming from the primary LLM first
    try:
        if LLM_STREAMING_ENABLED:
            print("Attempting to stream from primary LLM...")
            async for chunk in RAG_LLM.astream(formatted_messages):
                yield chunk.content
            return
        else: # Non-streaming path
             # This correctly uses your existing fallback chain
            yield await rag_chain.ainvoke(inputs)
            return

    except (RateLimitError, APIError) as e:
        print(f"⚠️ Primary LLM streaming failed ({type(e).__name__}), trying Gemini fallbacks...")
        if not GEMINI_API_KEYS:
            yield f"⚠️ Streaming failed: {e}. No fallback keys available."
            return
        
        # 2. If primary fails, iterate through fallback LLMs for streaming
        keys_shuffled = random.sample(GEMINI_API_KEYS, len(GEMINI_API_KEYS))
        for key in keys_shuffled:
            try:
                print(f"Attempting to stream from fallback Gemini key...")
                gemini_llm = get_gemini_llm(key)
                async for chunk in gemini_llm.astream(formatted_messages):
                    yield chunk.content
                return # Success, exit the function
            except Exception as gemini_e:
                print(f"⚠️ Gemini key failed. Error: {gemini_e}")
                continue # Try the next key

        yield f"⚠️ Streaming failed: All fallback models also failed."

    except Exception as e:
        yield f"⚠️ An unexpected streaming error occurred: {e}"