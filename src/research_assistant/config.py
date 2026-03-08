# src/research_assistant/config.py
"""
Configuration module — loads all settings from .env

Why use a config module?
- Single place to change settings
- Type-safe access to env variables
- Clear documentation of what each setting does
"""

import os
from dotenv import load_dotenv

# load_dotenv() reads your .env file and sets environment variables
# It does NOT overwrite variables already set in the OS environment
# This means you can override settings without changing .env
load_dotenv()


class Config:
    """
    Central config object.
    All settings are read once at startup.
    """

    # ── Ollama Settings ────────────────────────────────────────────────────
    # Ollama runs a local HTTP server — this is the default address
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # The model name — must match what you've pulled with `ollama pull`
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    # Temperature controls randomness:
    #   0.0 = deterministic, always same answer (good for factual Q&A)
    #   1.0 = very creative/random (good for creative writing)
    #   0.3 = slightly creative but mostly focused — our sweet spot for RAG
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))

    # ── LangSmith Settings ─────────────────────────────────────────────────
    # LangSmith reads these env vars AUTOMATICALLY — you don't need to pass
    # them to any LangChain object. Just setting them enables tracing globally.
    #
    # LANGCHAIN_TRACING_V2=true     → enables the new v2 tracing protocol
    # LANGCHAIN_API_KEY=ls__...     → authenticates with LangSmith
    # LANGCHAIN_PROJECT=name        → groups runs into a named project
    #
    # These are read directly by the langchain SDK, not by us explicitly.
    LANGSMITH_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "research-assistant")

    # ── Vector Store Settings ──────────────────────────────────────────────
    # ChromaDB is a local vector database — it persists to disk here
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # The "collection" is like a table in a SQL database
    CHROMA_COLLECTION_NAME: str = "research_docs"

    # ── Document Settings ──────────────────────────────────────────────────
    DOCS_DIR: str = os.getenv("DOCS_DIR", "./docs")

    # Chunk size: how many characters per text chunk
    # Why chunking? LLMs have a context window limit. We can't feed in
    # a 100-page PDF at once. We split it into chunks, embed each chunk,
    # and only retrieve the relevant ones.
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))

    # Overlap: how many characters two adjacent chunks share
    # This prevents losing context at chunk boundaries.
    # e.g., if a sentence spans the end of chunk 1 and start of chunk 2,
    # the overlap ensures both chunks contain it.
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # How many chunks to retrieve per query
    # More = more context for the LLM, but also more noise and slower
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))


# Create a single shared instance — import this in other modules
settings = Config()