# src/research_assistant/ingest.py
"""
Document Ingestion Pipeline
===========================

This module handles the "offline" part of RAG:
  1. Load documents from disk
  2. Split into chunks
  3. Embed each chunk into a vector
  4. Store vectors in ChromaDB

This runs ONCE (or when you add new docs).
The chat pipeline then queries this pre-built store at runtime.

RAG Architecture Overview:
──────────────────────────
Documents → Loader → Splitter → Embedder → VectorStore
                                                 ↑
                              Query → Embedder → Retriever → LLM
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
# Document is LangChain's core data structure.
# It has two fields:
#   .page_content  → the actual text string
#   .metadata      → dict of arbitrary info (source file, page number, etc.)

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
# Loaders read files and return List[Document]
# TextLoader   → reads .txt files, one Document per file
# PyPDFLoader  → reads PDFs, one Document per PAGE
# DirectoryLoader → scans a folder and uses the right loader per file type

from langchain_text_splitters import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter is the most commonly used splitter.
#
# HOW IT WORKS:
# It tries to split on these separators IN ORDER:
#   ["\n\n", "\n", " ", ""]
# First tries double newlines (paragraph breaks) — best split point.
# If chunk is still too big, tries single newlines.
# If still too big, splits on spaces (words).
# Last resort: splits mid-character.
#
# This is "recursive" because it keeps trying smaller separators
# until the chunk fits within the size limit.

from langchain_community.vectorstores import Chroma
# Chroma is a local vector database.
# It stores:
#   - The original text of each chunk
#   - The embedding vector for each chunk (a list of ~384 floats)
#   - Metadata for each chunk
#
# When you query it, it:
#   1. Embeds your query into a vector
#   2. Finds the N nearest vectors using cosine similarity
#   3. Returns the corresponding text chunks

from langchain_community.embeddings import HuggingFaceEmbeddings
# HuggingFaceEmbeddings uses a local sentence-transformer model.
# Default model: "all-MiniLM-L6-v2"
#   - 384-dimensional embeddings
#   - Very fast, runs on CPU
#   - ~80MB download (first run only, cached after)
#
# Why not use Ollama for embeddings?
#   - You could! But sentence-transformers are faster for batch embedding
#   - "all-MiniLM-L6-v2" is specifically trained for semantic similarity
#   - The embedding model and the chat model serve different purposes

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import settings

console = Console()


def load_documents(docs_dir: str) -> List[Document]:
    """
    Load all supported documents from the docs directory.

    Supports: .txt, .pdf
    Returns a flat list of Document objects.
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        console.print(f"[red]Docs directory not found: {docs_dir}[/red]")
        return []

    all_docs: List[Document] = []

    # ── Load .txt files ────────────────────────────────────────────────────
    txt_files = list(docs_path.glob("**/*.txt"))
    for txt_file in txt_files:
        loader = TextLoader(str(txt_file), encoding="utf-8")
        # .load() returns List[Document] — usually 1 doc per .txt file
        docs = loader.load()

        # Add the source filename to metadata so we can cite it later
        for doc in docs:
            doc.metadata["source"] = txt_file.name
            doc.metadata["file_type"] = "txt"

        all_docs.extend(docs)
        console.print(f"  [green]✓[/green] Loaded: {txt_file.name}")

    # ── Load .pdf files ────────────────────────────────────────────────────
    pdf_files = list(docs_path.glob("**/*.pdf"))
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        # PyPDFLoader returns one Document PER PAGE, each with metadata:
        #   {"source": "file.pdf", "page": 0}
        docs = loader.load()

        for doc in docs:
            doc.metadata["file_type"] = "pdf"

        all_docs.extend(docs)
        console.print(f"  [green]✓[/green] Loaded: {pdf_file.name} ({len(docs)} pages)")

    console.print(f"\n[bold]Total documents loaded:[/bold] {len(all_docs)}")
    return all_docs


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    Why split?
    1. LLMs have token limits (~2048-8192 tokens for llama3.2:3b)
    2. Smaller chunks = more precise retrieval (less noise)
    3. Embedding quality degrades with very long text

    Why RecursiveCharacterTextSplitter?
    It's "smart" — it tries to split at natural language boundaries
    (paragraphs > sentences > words) rather than just cutting at N chars.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,        # Max chars per chunk (not tokens!)
        chunk_overlap=settings.CHUNK_OVERLAP,  # Overlap between consecutive chunks
        length_function=len,                   # How to measure chunk size

        # add_start_index adds "start_index" to chunk metadata
        # so you know WHERE in the original document each chunk came from
        add_start_index=True,
    )

    # split_documents handles the list and preserves metadata from parent docs
    chunks = splitter.split_documents(documents)

    console.print(f"[bold]Chunks created:[/bold] {len(chunks)}")
    console.print(f"  Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    return chunks


def get_embeddings():
    """
    Initialize the embedding model.

    HuggingFaceEmbeddings downloads "all-MiniLM-L6-v2" on first use.
    After that it's cached in ~/.cache/huggingface/

    An embedding is a dense vector of floats that represents the
    "semantic meaning" of text. Similar texts have similar vectors.

    Example:
      "dog barks"  → [0.12, -0.34, 0.89, ...]  (384 numbers)
      "canine howls" → [0.11, -0.31, 0.91, ...]  (similar!)
      "pizza recipe" → [-0.72, 0.45, -0.12, ...]  (very different)
    """
    console.print("[dim]Loading embedding model (downloads once)...[/dim]")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},   # Use "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},
        # normalize_embeddings=True ensures vectors are unit length
        # This makes cosine similarity = dot product (faster math)
    )
    return embeddings


def build_vector_store(chunks: List[Document]) -> Chroma:
    """
    Create (or update) the ChromaDB vector store.

    ChromaDB stores data in PERSIST_DIR on disk.
    If you run ingest again, it recreates the collection.

    What happens internally:
    1. Each chunk's text is fed to the embedding model
    2. Embedding model returns a 384-dim vector
    3. ChromaDB stores: (id, vector, text, metadata) for each chunk
    4. The vectors are indexed for fast approximate nearest-neighbor search
    """
    embeddings = get_embeddings()

    console.print("[dim]Building vector store...[/dim]")

    # If the persist directory exists, delete and recreate
    # (to avoid duplicate chunks if you re-run ingest)
    import shutil
    if os.path.exists(settings.CHROMA_PERSIST_DIR):
        shutil.rmtree(settings.CHROMA_PERSIST_DIR)

    # Chroma.from_documents():
    # 1. Takes all chunks
    # 2. Calls embeddings.embed_documents() on all of them (batched)
    # 3. Stores everything in the persist directory
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

    # .persist() writes the in-memory store to disk
    # (in newer chromadb versions this is automatic, but doesn't hurt)
    vector_store.persist()

    count = vector_store._collection.count()
    console.print(f"[bold green]✓ Vector store built![/bold green] {count} vectors stored at {settings.CHROMA_PERSIST_DIR}")

    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an existing vector store from disk.
    Called at chat time — no re-embedding needed.
    """
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    return vector_store


def run_ingestion():
    """Main entry point for the ingestion pipeline."""
    console.print("\n[bold cyan]🔄 Starting Document Ingestion Pipeline[/bold cyan]\n")

    # Step 1: Load
    console.print("[bold]Step 1: Loading documents[/bold]")
    documents = load_documents(settings.DOCS_DIR)

    if not documents:
        console.print("[red]No documents found! Add .txt or .pdf files to the docs/ folder.[/red]")
        return

    # Step 2: Split
    console.print("\n[bold]Step 2: Splitting into chunks[/bold]")
    chunks = split_documents(documents)

    # Step 3: Embed + Store
    console.print("\n[bold]Step 3: Embedding and storing in ChromaDB[/bold]")
    build_vector_store(chunks)

    console.print("\n[bold green]✅ Ingestion complete! You can now run `chat`.[/bold green]")