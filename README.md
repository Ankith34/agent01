# 🦜 Personal Research Assistant

A **local AI research assistant** that allows you to chat with your own documents.  
Built using **LangChain, Ollama, ChromaDB, and LangSmith**.

This project uses **Retrieval Augmented Generation (RAG)** to retrieve relevant information from your documents and answer questions using a local LLM.

---

# ✨ Features

- Chat with your **PDF and text documents**
- Runs **locally with Ollama**
- Uses **ChromaDB vector database**
- Built with **LangChain RAG pipeline**
- Automatic **LangSmith tracing**
- Simple **CLI interface**

---

# 🛠️ Tech Stack

- **LangChain**
- **Ollama**
- **ChromaDB**
- **LangSmith**
- **Sentence Transformers**
- **Python**
- **uv (Python package manager)**

---

# 📋 Requirements

Make sure the following tools are installed:

- Python **3.10+**
- **Ollama**
- **uv**
- A **LangSmith account (free)**

### Install Ollama

Download from:

https://ollama.com

### Install uv

```bash
pip install uv
```

---

# ⚙️ Setup

## 1️⃣ Install Dependencies

Run the following commands:

```bash
uv add langchain langchain-ollama langchain-community langchain-core
uv add langchain-text-splitters chromadb sentence-transformers
uv add langsmith python-dotenv rich pypdf
```

---

## 2️⃣ Pull the LLM Model

Download the model with Ollama:

```bash
ollama pull llama3.2:3b
```

---

## 3️⃣ Create Environment File

Create a `.env` file in the project root:

```env
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
```

Get your **LangSmith API key** from:

https://smith.langchain.com → **Settings → API Keys**

---

# 📂 Add Your Documents

Place your files inside the **docs/** folder.

Supported formats:

- `.pdf`
- `.txt`

Example:

```
docs/
   research_paper.pdf
   notes.txt
```

---

# 🚀 Running the Project

## 1️⃣ Test Ollama Connection

```bash
uv run python -m research_assistant.main demo
```

---

## 2️⃣ Ingest Documents

Run this **once** (or whenever you add new documents):

```bash
uv run python -m research_assistant.main ingest
```

This will:

- Load the documents
- Split them into chunks
- Generate embeddings
- Store them in **ChromaDB**

---

## 3️⃣ Start Chatting

```bash
uv run python -m research_assistant.main chat
```

Example query:

```
What are the main findings of the research paper?
```

---

# 📁 Project Structure

```
research-assistant/
│
├── .env
├── pyproject.toml
├── docs/                 # Put PDFs and text files here
│
└── src/
    └── research_assistant/
        ├── config.py     # Configuration settings
        ├── ingest.py     # Document loading and vector store
        ├── chain.py      # RAG chain
        └── main.py       # CLI entry point
```

---

# ⚙️ How It Works

### 1️⃣ Document Ingestion

- Documents are split into smaller **text chunks**
- Chunks are converted into **embeddings**
- Stored in **ChromaDB vector database**

Embeddings model used:

```
all-MiniLM-L6-v2
```

---

### 2️⃣ Query Processing

When a user asks a question:

1. The question is converted into an **embedding**
2. The system retrieves the **top 4 most similar chunks**
3. The chunks are passed to the **LLM (Llama 3.2 3B)**
4. The model generates a response based on the context

---

### 3️⃣ LangSmith Tracing

Every step of the pipeline is automatically logged to **LangSmith**, allowing you to inspect:

- prompts
- retrieved chunks
- LLM responses
- pipeline performance

---

# 📌 Example Use Cases

- Research paper analysis
- Studying from PDFs
- Personal knowledge base
- Document Q&A assistant

---

# 📜 License

MIT License

---

# ⭐ If you found this project useful, consider giving it a star!
