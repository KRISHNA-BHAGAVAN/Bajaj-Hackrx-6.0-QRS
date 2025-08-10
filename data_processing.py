# data_processing.py
import os
import faiss
import hashlib
import asyncio
import json
import traceback
import aiohttp
import numpy as np
from typing import List
from uuid import uuid4

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.tools import tool

from document_parser import extract_image, extract_pptx, extract_xlsx
from config import (
    embeddings, EMBED_CONCURRENCY,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBED_CACHE_DIR, CACHE_DIR,
)


@tool
async def get_web_content(url: str) -> str:
    """Fetches text content from a URL asynchronously with a timeout."""
    print(f"Fetching web content from: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                text = await resp.text()
                soup = BeautifulSoup(text, 'html.parser')
                return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        return f"Error fetching URL: {e}"

def url_to_cache_path(url: str) -> str:
    """Generates a consistent cache path for a URL's FAISS index."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, url_hash)

def get_docs_from_url(doc_url: str) -> List[Document]:
    """Downloads and parses a document from a URL based on its extension."""
    print(f"Downloading document from: {doc_url}")
    url_path = doc_url.split('?')[0]
    ext = os.path.splitext(url_path)[1].lower()

    if ext in [".zip", ".bin"]:
        print(f"Unsupported file type: {ext}")
        return []
    try:
        if ext in [".pdf", ".docx"]:
            return PyMuPDFLoader(doc_url).load()
        elif ext == ".pptx":
            return [Document(page_content=extract_pptx(doc_url))]
        elif ext == ".xlsx":
            return [Document(page_content=str(extract_xlsx(doc_url)))]
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            return [Document(page_content=extract_image(doc_url))]
        else:
            print("No specific document extension found. Treating as a generic web page.")
            web_content = asyncio.run(get_web_content.ainvoke({"url": doc_url}))
            return [] if "Error:" in web_content else [Document(page_content=web_content)]
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        traceback.print_exc()
        return []

def _chunk_docs(docs: List[Document], doc_url: str) -> List[Document]:
    """Helper function to chunk documents based on file type."""
    ext = os.path.splitext(doc_url.split('?')[0])[1].lower()
    if ext in [".pdf", ".docx"]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(docs)
    return docs

async def get_chunks(docs: List[Document], doc_url: str) -> List[Document]:
    """Asynchronously chunks documents to avoid blocking."""
    return await asyncio.to_thread(_chunk_docs, docs, doc_url)


# --- Embedding Cache Helpers ---
def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _cache_paths(batch_index: int):
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    return os.path.join(EMBED_CACHE_DIR, f"batch_{batch_index}.npy"), os.path.join(EMBED_CACHE_DIR, f"batch_{batch_index}.json")

def _load_cached_embeddings(batch_index: int):
    npy_path, meta_path = _cache_paths(batch_index)
    if os.path.exists(npy_path) and os.path.exists(meta_path):
        try:
            return np.load(npy_path), json.load(open(meta_path))
        except Exception:
            return None, None
    return None, None

def _save_cached_embeddings(batch_index: int, vectors, meta):
    npy_path, meta_path = _cache_paths(batch_index)
    np.save(npy_path, vectors)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

# --- Async Batch Embedding ---
async def embed_batches_concurrently(vs, chunks, batch_size):
    """
    Coordinates concurrent batch embedding using a semaphore and per-batch caching.
    """
    sem = asyncio.Semaphore(EMBED_CONCURRENCY)
    tasks = []

    for i in range(0, len(chunks), batch_size):
        batch_index = (i // batch_size) + 1
        batch = chunks[i:i + batch_size]
        ids = [str(uuid4()) for _ in batch]
        print(f"📦 Scheduling batch {batch_index} with {len(batch)} chunks...")

        async def _run_batch(b=batch, ids_batch=ids, bi=batch_index):
            async with sem:
                cache_vectors, cache_meta = _load_cached_embeddings(bi)
                text_hashes = [_hash_text(doc.page_content) for doc in b]

                if cache_vectors is not None and cache_meta == text_hashes:
                    print(f"⚡ Loaded cached vectors for batch {bi}")
                    vs.add_documents(documents=b, ids=ids_batch, embeddings=cache_vectors.tolist())
                    return

                def _embed_sync():
                    texts = [d.page_content for d in b]
                    vectors = embeddings.embed_documents(texts)
                    _save_cached_embeddings(bi, np.array(vectors), text_hashes)
                    vs.add_documents(documents=b, ids=ids_batch, embeddings=vectors)

                await asyncio.to_thread(_embed_sync)
        
        tasks.append(asyncio.create_task(_run_batch()))

    if tasks:
        await asyncio.gather(*tasks)
