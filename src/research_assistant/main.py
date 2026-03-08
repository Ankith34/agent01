# src/research_assistant/main.py
"""
CLI Entry Point
===============

Provides two commands:
  `uv run python -m research_assistant.main ingest`  → build vector store
  `uv run python -m research_assistant.main chat`    → interactive chat
"""

import sys
import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint

console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]🦜 Personal Research Assistant[/bold cyan]\n"
        "[dim]Powered by LangChain + LangSmith + Ollama (llama3.2:3b)[/dim]",
        border_style="cyan"
    ))


def run_ingest():
    """Ingest documents into the vector store."""
    print_banner()
    from .ingest import run_ingestion
    run_ingestion()


def run_chat():
    """Interactive chat loop."""
    print_banner()

    # ── Check vector store exists ──────────────────────────────────────────
    from .config import settings
    if not Path(settings.CHROMA_PERSIST_DIR).exists():
        console.print(
            "[red]❌ No vector store found![/red]\n"
            "Run [bold]`uv run python -m research_assistant.main ingest`[/bold] first."
        )
        sys.exit(1)

    # ── Load chain ─────────────────────────────────────────────────────────
    console.print("[dim]Loading vector store and building chain...[/dim]")

    from .ingest import load_vector_store
    from .chain import build_rag_chain, query_with_sources

    vector_store = load_vector_store()
    rag_chain, retriever = build_rag_chain(vector_store)

    console.print("[green]✓ Ready![/green] LangSmith tracing is [bold]ON[/bold].\n")
    console.print(f"  Project: [cyan]{settings.LANGSMITH_PROJECT}[/cyan]")
    console.print(f"  View traces at: [link=https://smith.langchain.com]https://smith.langchain.com[/link]\n")
    console.print("[dim]Type your question and press Enter. Type 'quit' to exit.[/dim]\n")
    console.print("─" * 60)

    # ── Chat loop ──────────────────────────────────────────────────────────
    while True:
        try:
            # Get user input
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye! 👋[/dim]")
                break

            # Show thinking indicator
            console.print("[dim]Thinking...[/dim]")

            # Run the chain (this call is traced in LangSmith!)
            result = query_with_sources(question, rag_chain, retriever)

            # ── Display Answer ─────────────────────────────────────────────
            console.print(f"\n[bold green]Assistant:[/bold green]")
            console.print(Panel(
                result["answer"],
                border_style="green",
                padding=(1, 2),
            ))

           

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'quit' to exit.[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


def run_demo():
    """
    Run a quick demo without needing documents.
    Just tests the Ollama connection with a simple prompt.
    """
    print_banner()
    console.print("[bold]Running connection test...[/bold]\n")

    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage
    from .config import settings

    try:
        llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0,
        )
        response = llm.invoke([HumanMessage(content="Say 'hello, I am working!' and nothing else.")])
        console.print(f"[green]✓ Ollama connection successful![/green]")
        console.print(f"  Model: {settings.OLLAMA_MODEL}")
        console.print(f"  Response: [cyan]{response.content}[/cyan]\n")
        console.print("LangSmith tracing should show this run at:")
        console.print(f"  [link=https://smith.langchain.com]https://smith.langchain.com[/link]")
    except Exception as e:
        console.print(f"[red]❌ Connection failed: {e}[/red]")
        console.print("\nMake sure Ollama is running:")
        console.print("  [cyan]ollama serve[/cyan]")
        console.print("  [cyan]ollama pull llama3.2:3b[/cyan]")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    commands = {
        "ingest": run_ingest,
        "chat": run_chat,
        "demo": run_demo,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        console.print(
            "[bold]Usage:[/bold]\n"
            "  [cyan]uv run python -m research_assistant.main ingest[/cyan]  "
            "→ Build vector store from docs/\n"
            "  [cyan]uv run python -m research_assistant.main chat[/cyan]    "
            "→ Start interactive chat\n"
            "  [cyan]uv run python -m research_assistant.main demo[/cyan]    "
            "→ Test Ollama connection\n"
        )
        sys.exit(1)

    commands[sys.argv[1]]()