#!/usr/bin/env python3
"""
OvernightPredict - Recursive Agent System with Self-Correction

Main entry point for the application.

Usage:
    python main.py                    # Interactive CLI
    python main.py --dashboard        # Launch TUI dashboard
    python main.py --api              # Start API server (future)

Environment Variables:
    OPENAI_API_KEY      - OpenAI API key
    OPENAI_ENABLED      - Enable OpenAI (true/false)
    DEEPSEEK_API_KEY    - DeepSeek API key
    DEEPSEEK_ENABLED    - Enable DeepSeek (true/false)
    CLAUDE_API_KEY      - Claude/Anthropic API key
    CLAUDE_ENABLED      - Enable Claude (true/false)
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OvernightPredict - Recursive Agent System with Self-Correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                          # Start interactive CLI
    python main.py --dashboard              # Launch TUI dashboard
    python main.py --create openai          # Quick create session
    python main.py --parallel openai,claude # Create parallel sessions
        """,
    )

    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch TUI dashboard",
    )

    parser.add_argument(
        "--create",
        type=str,
        metavar="PROVIDER",
        help="Quick create a session with specified provider",
    )

    parser.add_argument(
        "--parallel",
        type=str,
        metavar="PROVIDERS",
        help="Create parallel sessions (comma-separated providers)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Initial prompt for session creation",
    )

    parser.add_argument(
        "--workdir",
        type=str,
        default="",
        help="Working directory for sessions",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    return parser.parse_args()


async def quick_create_session(
    provider: str,
    prompt: str,
    workdir: str,
) -> None:
    """Quick create and start a session."""
    from src.core.config.settings import get_settings, LLMProvider
    from src.core.utils.logging import setup_logging
    from src.infrastructure.storage.sqlite_repository import SQLiteSessionRepository
    from src.infrastructure.storage.memory_event_bus import InMemoryEventBus
    from src.infrastructure.storage.memory_context_store import InMemoryContextStore
    from src.application.services.orchestrator import Orchestrator
    from rich.console import Console

    console = Console()
    setup_logging(level="INFO")

    console.print("[bold]Quick Session Creation[/bold]")

    settings = get_settings()

    # Initialize components
    db_path = settings.work_dir / "overnight.db"
    repository = SQLiteSessionRepository(db_path)
    await repository.initialize()

    event_bus = InMemoryEventBus()
    context_store = InMemoryContextStore()

    orchestrator = Orchestrator(
        settings=settings,
        repository=repository,
        context_store=context_store,
        event_bus=event_bus,
    )

    await orchestrator.initialize()
    await orchestrator.start()

    # Map provider
    provider_map = {
        "openai": LLMProvider.OPENAI,
        "deepseek": LLMProvider.DEEPSEEK,
        "claude": LLMProvider.CLAUDE,
    }

    llm_provider = provider_map.get(provider.lower())
    if not llm_provider:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        return

    # Create session
    session_id = await orchestrator.create_session(
        provider=llm_provider,
        name=f"Quick-{provider}",
        initial_prompt=prompt or "Ready for coding tasks.",
        working_directory=workdir or str(Path.cwd()),
    )

    console.print(f"[green]Session created: {session_id}[/green]")

    # Start session
    await orchestrator.start_session(session_id)
    console.print("[green]Session started.[/green]")

    # Interactive loop
    console.print("\n[dim]Enter questions (Ctrl+C to exit):[/dim]")

    try:
        while True:
            question = input("\n> ")
            if not question.strip():
                continue

            console.print("[dim]Processing...[/dim]")
            answer = await orchestrator.process_question(session_id, question)
            console.print(f"\n[bold]Answer:[/bold]\n{answer}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")

    await orchestrator.stop()


async def create_parallel_sessions(
    providers: str,
    prompt: str,
    workdir: str,
) -> None:
    """Create parallel sessions across multiple providers."""
    from src.core.config.settings import get_settings, LLMProvider
    from src.core.utils.logging import setup_logging
    from src.infrastructure.storage.sqlite_repository import SQLiteSessionRepository
    from src.infrastructure.storage.memory_event_bus import InMemoryEventBus
    from src.infrastructure.storage.memory_context_store import InMemoryContextStore
    from src.application.services.orchestrator import Orchestrator
    from rich.console import Console

    console = Console()
    setup_logging(level="INFO")

    console.print("[bold]Parallel Session Creation[/bold]")

    settings = get_settings()

    # Initialize components
    db_path = settings.work_dir / "overnight.db"
    repository = SQLiteSessionRepository(db_path)
    await repository.initialize()

    event_bus = InMemoryEventBus()
    context_store = InMemoryContextStore()

    orchestrator = Orchestrator(
        settings=settings,
        repository=repository,
        context_store=context_store,
        event_bus=event_bus,
    )

    await orchestrator.initialize()
    await orchestrator.start()

    # Parse providers
    provider_list = [p.strip() for p in providers.split(",")]
    provider_map = {
        "openai": LLMProvider.OPENAI,
        "deepseek": LLMProvider.DEEPSEEK,
        "claude": LLMProvider.CLAUDE,
    }

    llm_providers = []
    for p in provider_list:
        if p.lower() in provider_map:
            llm_providers.append(provider_map[p.lower()])
        else:
            console.print(f"[yellow]Unknown provider: {p}[/yellow]")

    if not llm_providers:
        console.print("[red]No valid providers specified.[/red]")
        return

    # Create group
    group_id = await orchestrator.create_session_group(
        providers=llm_providers,
        initial_prompt=prompt or "Ready for parallel coding tasks.",
        working_directory=workdir or str(Path.cwd()),
    )

    console.print(f"[green]Session group created: {group_id}[/green]")
    console.print(f"[green]Providers: {', '.join(p.value for p in llm_providers)}[/green]")

    # Start all
    await orchestrator.start_all_sessions()
    console.print("[green]All sessions started.[/green]")

    # Show status
    status = orchestrator.get_group_status(group_id)
    console.print(f"\n[bold]Group Status:[/bold]")
    console.print(f"Sessions: {status['session_count']}")

    console.print("\n[dim]Use dashboard for interactive control: python main.py --dashboard[/dim]")

    # Keep running
    console.print("\n[dim]Press Ctrl+C to stop all sessions.[/dim]")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass

    await orchestrator.stop()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.dashboard:
        # Launch dashboard
        from src.core.utils.logging import setup_logging
        setup_logging(level=args.log_level)

        from src.presentation.cli.main_cli import CLI
        cli = CLI()
        cli.run()

    elif args.create:
        # Quick create
        asyncio.run(quick_create_session(
            args.create,
            args.prompt,
            args.workdir,
        ))

    elif args.parallel:
        # Parallel sessions
        asyncio.run(create_parallel_sessions(
            args.parallel,
            args.prompt,
            args.workdir,
        ))

    else:
        # Default: Interactive CLI
        from src.core.utils.logging import setup_logging
        setup_logging(level=args.log_level)

        from src.presentation.cli.main_cli import CLI
        cli = CLI()
        cli.run()


if __name__ == "__main__":
    main()
