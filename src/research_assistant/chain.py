# src/research_assistant/chain.py
"""
RAG Chain
=========

This is the core of the project — the LangChain Runnable chain
that powers the question-answering.

Chain Architecture:
───────────────────
User Question
     │
     ▼
┌─────────────────────┐
│  Retriever          │  ← Searches ChromaDB for relevant chunks
│  (ChromaDB)         │
└─────────────────────┘
     │
     │  Returns: List[Document] (4 most similar chunks)
     ▼
┌─────────────────────┐
│  ChatPromptTemplate │  ← Inserts chunks + question into prompt
└─────────────────────┘
     │
     │  Returns: formatted ChatPromptValue
     ▼
┌─────────────────────┐
│  ChatOllama         │  ← Sends prompt to llama3.2:3b via Ollama API
│  (llama3.2:3b)      │
└─────────────────────┘
     │
     │  Returns: AIMessage object
     ▼
┌─────────────────────┐
│  StrOutputParser    │  ← Extracts just the string from AIMessage
└─────────────────────┘
     │
     ▼
  Final Answer (str)

Every step above is automatically traced in LangSmith!
"""

from typing import List, Dict, Any
from operator import itemgetter

from langchain_ollama import ChatOllama
# ChatOllama connects to your locally running Ollama server.
# It uses the /api/chat endpoint on localhost:11434.
# Unlike OpenAI, there's no API key — Ollama runs fully locally.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# ChatPromptTemplate builds structured prompts with role labels:
#   System: ...   (instructions for the AI)
#   Human: ...    (the user's question)
#   AI: ...       (the assistant's turn)
#
# {placeholders} in curly braces are filled in at runtime.

from langchain_core.output_parsers import StrOutputParser
# StrOutputParser takes the AIMessage returned by the LLM
# and extracts just the .content string from it.
# Input:  AIMessage(content="The answer is 42", ...)
# Output: "The answer is 42"

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)
# These are the building blocks of LangChain's LCEL (LangChain Expression Language)
#
# RunnablePassthrough — passes input through unchanged (like a wire)
# RunnableLambda — wraps any Python function as a Runnable
# RunnableParallel — runs multiple Runnables in parallel and merges outputs
#
# LCEL uses the | (pipe) operator to chain Runnables:
#   chain = prompt | llm | parser
# This is equivalent to:
#   def chain(input):
#       return parser(llm(prompt(input)))

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langsmith import traceable
# @traceable is a decorator that wraps any Python function with LangSmith tracing.
# Even functions that don't use LangChain directly will appear in the trace tree!

from rich.console import Console

from .config import settings
from .ingest import load_vector_store

console = Console()


def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents into a single context string.

    Takes:  [Document("chunk1"), Document("chunk2"), ...]
    Returns: "Source: file.txt\nchunk1\n\n---\n\nSource: file.txt\nchunk2\n..."

    This string is injected into the {context} placeholder in our prompt.
    The LLM sees this as part of the prompt, not as a separate input.
    """
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        page_info = f" (page {page + 1})" if page != "" else ""
        formatted_parts.append(
            f"[Source {i}: {source}{page_info}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_parts)


def build_rag_chain(vector_store: Chroma):
    """
    Build the full RAG chain using LCEL (LangChain Expression Language).

    LCEL is LangChain's way of composing operations using the | operator.
    It's inspired by Unix pipes — output of one step becomes input of the next.

    Why LCEL instead of the older LLMChain / RetrievalQA classes?
    1. More transparent — you see every step
    2. Better streaming support
    3. Automatic LangSmith tracing of each component
    4. Easy to modify individual steps
    5. Supports async out of the box
    """

    # ── Step 1: LLM Setup ──────────────────────────────────────────────────
    llm = ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=settings.OLLAMA_TEMPERATURE,

        # num_predict limits max output tokens
        # llama3.2:3b can handle up to ~8192 context tokens
        num_predict=512,

        # Ollama-specific: don't keep model loaded between calls
        # (set to -1 to keep it loaded for faster repeated queries)
        keep_alive="5m",
    )

    # ── Step 2: Prompt Template ────────────────────────────────────────────
    # This is the system prompt + user question template.
    # The LLM will ONLY use the provided context — this prevents hallucination
    # by instructing it not to use its parametric knowledge.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful research assistant. Answer the user's question
based ONLY on the provided context below. If the answer is not in the context,
say "I don't have enough information in the provided documents to answer that."

Do not make up information. Cite which source(s) you used in your answer.

Context:
{context}"""
        ),
        (
            "human",
            "{question}"
        ),
    ])
    # The {context} and {question} placeholders will be filled by the chain.

    # ── Step 3: Retriever ──────────────────────────────────────────────────
    retriever = vector_store.as_retriever(
        search_type="similarity",
        # search_type options:
        #   "similarity"  → pure cosine similarity (default)
        #   "mmr"         → Maximum Marginal Relevance: balances
        #                   relevance with diversity (avoids returning
        #                   5 nearly identical chunks)
        #   "similarity_score_threshold" → only returns if score > threshold

        search_kwargs={
            "k": settings.RETRIEVAL_K,   # Return top-K most similar chunks
        }
    )

    # ── Step 4: Output Parser ──────────────────────────────────────────────
    output_parser = StrOutputParser()
    # Simple but important — extracts the string from AIMessage

    # ── Step 5: Assemble the Chain with LCEL ──────────────────────────────
    #
    # The chain needs two inputs: {question} and {context}
    # The user provides {question}; we derive {context} from the retriever.
    #
    # RunnableParallel runs both branches simultaneously:
    #   - "context"  branch: retriever → format_docs (runs the vector search)
    #   - "question" branch: passthrough (just passes the question through)
    #
    # After RunnableParallel, the output is:
    #   {"context": "formatted chunks...", "question": "user's question"}
    # This dict is then fed to the prompt template which fills in both.

    rag_chain = (
        RunnableParallel(
            {
                # Retrieve relevant chunks, then format them as a string
                "context": retriever | RunnableLambda(format_docs),

                # Just pass the question through unchanged
                "question": RunnablePassthrough(),
            }
        )
        | prompt          # Fill in {context} and {question}
        | llm             # Send to Ollama → returns AIMessage
        | output_parser   # Extract .content string
    )
    # The full type flow:
    #   str → {"context": str, "question": str} → ChatPromptValue → AIMessage → str

    return rag_chain, retriever


@traceable(name="Research Assistant Query")
def query_with_sources(
    question: str,
    rag_chain,
    retriever
) -> Dict[str, Any]:
    """
    Run a query and return both the answer and the source documents.

    @traceable wraps this function in a LangSmith span.
    Even though it's not a LangChain Runnable, it will appear
    in the LangSmith trace tree as a parent span containing
    all the LangChain sub-spans (retriever, LLM, etc.)

    Returns:
        {
            "answer": "The answer text...",
            "sources": [Document, Document, ...],
            "question": "original question"
        }
    """
    # Get the answer from the chain
    answer = rag_chain.invoke(question)

    # Also get the source docs for display (the chain already retrieved them,
    # but we re-retrieve here to show the user which sources were used)
    # In production you'd restructure the chain to return both in one call
    source_docs = retriever.invoke(question)

    return {
        "answer": answer,
        "sources": source_docs,
        "question": question,
    }


def build_conversational_chain(vector_store: Chroma):
    """
    BONUS: A conversational chain that remembers chat history.

    The difference from the basic chain:
    - Uses MessagesPlaceholder to inject past messages
    - Reformulates the question using history before retrieving
    - "What did you say about X?" → "What did the document say about X?"

    This is called a "Contextual Compression" or "History-Aware Retriever" pattern.
    """
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.messages import HumanMessage, AIMessage

    llm = ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=settings.OLLAMA_TEMPERATURE,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K})

    # Prompt to reformulate the question given chat history
    # Example: history has "tell me about black holes"
    #          new question: "how big are they?"
    #          reformulated: "how big are black holes?"
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, "
                   "reformulate it as a standalone question. "
                   "Do NOT answer it. Just reformulate if needed, else return as-is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Main QA prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful research assistant. Answer based only on:\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # create_stuff_documents_chain "stuffs" all retrieved docs into the context
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Full conversational chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
