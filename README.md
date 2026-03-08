🦜 Personal Research Assistant

Chat with your own documents using a local LLM.
Built with LangChain, LangSmith, Ollama, and ChromaDB.

This project allows you to:

Load PDF or text documents

Convert them into embeddings

Store them in a vector database

Ask questions about them using a local LLM

Requirements

Make sure you have the following installed:

Ollama

uv (Python package manager)

Python 3.10+

LangSmith account (free)

Install Ollama from:
https://ollama.com

Install uv:

pip install uv
Setup
1. Install Dependencies

Run the following commands:

uv add langchain langchain-ollama langchain-community langchain-core
uv add langchain-text-splitters chromadb sentence-transformers
uv add langsmith python-dotenv rich pypdf
2. Pull the LLM Model

Download the model using Ollama:

ollama pull llama3.2:3b
3. Create the .env File

Create a .env file in the root directory and add:

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=research-assistant

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TEMPERATURE=0.3

CHROMA_PERSIST_DIR=./chroma_db
DOCS_DIR=./docs

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4

Get your LangSmith API key from:

https://smith.langchain.com

Go to Settings → API Keys

4. Add Your Documents

Place any files you want to chat with inside the docs/ folder.

Supported formats:

.pdf

.txt

Example:

docs/
   research_paper.pdf
   notes.txt
Running the Project
1. Test Ollama Connection
uv run python -m research_assistant.main demo
2. Ingest Documents

Run this once (or whenever you add new documents):

uv run python -m research_assistant.main ingest

This will:

Split documents into chunks

Generate embeddings

Store them in ChromaDB

3. Start Chatting
uv run python -m research_assistant.main chat

Now you can ask questions about your documents.

Example:

> What is the main idea of the research paper?
Project Structure
research-assistant/
├── .env
├── pyproject.toml
├── docs/                # Put your PDFs or text files here
└── src/
    └── research_assistant/
        ├── config.py    # Configuration settings
        ├── ingest.py    # Document loading + vector store
        ├── chain.py     # RAG pipeline
        └── main.py      # CLI entry point
How It Works
1. Ingest

Documents are:

Split into chunks

Converted into embeddings using all-MiniLM-L6-v2

Stored locally in ChromaDB

2. Query

When you ask a question:

The question is converted into an embedding

The system retrieves the top 4 most similar chunks

Those chunks are sent to Llama 3.2 (3B) as context

The model generates an answer

3. Trace

Every step is logged in LangSmith, allowing you to inspect:

prompts

retrieved chunks

LLM responses

chain performance

✅ Result:
You get a local ChatGPT-like assistant for your own documents.
