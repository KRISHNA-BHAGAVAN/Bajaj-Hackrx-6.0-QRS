# config.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


# Load environment variables from .env file
load_dotenv(override=True)

# --- API Keys & Tokens ---
# Make sure to set these in your .env file
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#These keys are used only for fallback which in our case is useful since our openai api is having ratelimit problem
GEMINI_API_KEYS = [os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 7) if os.getenv(f"GEMINI_API_KEY_{i}")]

# --- Application Constants ---
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))
LLM_STREAMING_ENABLED = os.getenv("LLM_STREAMING_ENABLED", "true").lower() in ("true", "1", "yes")
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "4"))
QA_CONCURRENCY = int(os.getenv("QA_CONCURRENCY", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "5000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CACHE_DIR = "./faiss_cache"
EMBED_CACHE_DIR = "./embed_cache"

# --- Embedding Model Configuration ---
# Recommended model but need an nvidia api key
embeddings = NVIDIAEmbeddings(
    model="nvidia/llama-3.2-nv-embedqa-1b-v2",
    api_key=NVIDIA_API_KEY,
    truncate="NONE",
)

#For local embedding, use the following model
# from langchain_huggingface import HuggingFaceEmbeddings
# import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# embeddings = HuggingFaceEmbeddings(
#     model_name="jinaai/jina-embeddings-v4",
#     model_kwargs={
#         'trust_remote_code': True,
#         'device': str(device)
#     },
#     encode_kwargs={
#         'normalize_embeddings': True
#     }
# )
# If you want to use any other embedding model make sure it supports the recommended embedding size of 2048 for better retrieval recall

EMBED_BATCH_API_AVAILABLE = hasattr(embeddings, "embed_documents")


# =================================================================================
# LLM SELECTION FOR THE MAIN RAG PIPELINE (llm_services.py)
# =================================================================================
# INSTRUCTIONS: Uncomment the model you want to use for the main question-answering task.
# OpenAI is the default.

# --- Option 1: OpenAI (Default) ---
RAG_LLM = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0, api_key=OPENAI_API_KEY)

# --- Option 2: Groq ---
# To use Groq, comment out the OpenAI model above and uncomment the following lines:
# RAG_LLM = ChatOpenAI(
#     api_key=GROQ_API_KEY,
#     model_name="llama3-70b-8192", #choose any model of your choice from groq
#     temperature=0,
#     base_url="https://api.groq.com/openai/v1/",
# )

# --- Option 3: Gemini ---
# To use Gemini, comment out the models above and uncomment the following lines.
# Note: The first available Gemini key from your .env file will be used.
# if GEMINI_API_KEYS:
#      RAG_LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEYS[0], temperature=0)
# else:
#      # Fallback or error if no Gemini key is found
#      RAG_LLM = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0, api_key=OPENAI_API_KEY)


# =================================================================================
# LLM SELECTION FOR THE REACT AGENT (react_agent.py)
# =================================================================================
# INSTRUCTIONS: Uncomment the model you want the ReAct agent to use for reasoning.

# --- Option 1: OpenAI (Default) ---
# To use OpenAI, comment out the Groq model above and uncomment the following lines:
AGENT_LLM = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0, api_key=OPENAI_API_KEY, streaming=True)


# --- Option 2: Groq ---
# AGENT_LLM = ChatOpenAI(
#     api_key=GROQ_API_KEY,
#     model_name="llama3-8b-8192",
#     temperature=0,
#     base_url="https://api.groq.com/openai/v1/",
#     streaming=True
# )

# --- Option 3: Gemini ---
# To use Gemini, comment out the models above and uncomment the following lines.
# if GEMINI_API_KEYS:
#      AGENT_LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEYS[0], temperature=0)
# else:
#      # Fallback or error if no Gemini key is found
#      AGENT_LLM = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0, api_key=OPENAI_API_KEY)


# Helper function for the fallback logic in llm_services.py
def get_gemini_llm(api_key):
    """Initializes a Gemini LLM instance with a specific API key."""
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
