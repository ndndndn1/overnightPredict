"""
Main CLI interface for OvernightPredict.

Provides command-line interface for:
- Configuration setup
- Session creation and management
- Monitoring and control
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.live import Live
from rich import print as rprint

from src.core.config.settings import Settings, LLMProvider, ClaudeAuthType, get_settings
from src.core.utils.logging import setup_logging, get_logger


console = Console()


class CLI:
    """
    Command-line interface for OvernightPredict.

    Provides interactive setup and management of the recursive
    agent system.
    """

    def __init__(self):
        """Initialize the CLI."""
        self._settings: Optional[Settings] = None
        self._orchestrator = None
        self._logger = get_logger("cli")

    def run(self) -> None:
        """Run the CLI."""
        self._show_banner()

        try:
            asyncio.run(self._main_loop())
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Shutting down...[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    def _show_banner(self) -> None:
        """Display the welcome banner."""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    OvernightPredict                            ║
║      Recursive Agent System with Self-Correction               ║
║                                                                 ║
║  Meta-cognition enabled AI coding assistant                    ║
║  OODA Loop: Forecast → Execute → Evaluate → Tune               ║
╚═══════════════════════════════════════════════════════════════╝
        """
        console.print(Panel(banner, style="bold blue"))

    async def _main_loop(self) -> None:
        """Main interaction loop."""
        # Initial setup
        await self._setup_configuration()

        # Initialize system
        await self._initialize_system()

        # Main menu loop
        while True:
            action = await self._show_main_menu()

            if action == "create":
                await self._create_session_flow()
            elif action == "status":
                await self._show_status()
            elif action == "dashboard":
                await self._launch_dashboard()
            elif action == "process":
                await self._process_question_flow()
            elif action == "settings":
                await self._modify_settings()
            elif action == "quit":
                await self._shutdown()
                break

    async def _setup_configuration(self) -> None:
        """Set up initial configuration."""
        console.print("\n[bold]Initial Configuration[/bold]\n")

        # Check for existing configuration
        env_file = Path(".env")
        if env_file.exists():
            use_existing = Confirm.ask(
                "Found existing .env file. Use existing configuration?",
                default=True,
            )
            if use_existing:
                self._settings = get_settings()
                console.print("[green]Configuration loaded.[/green]")
                return

        # Interactive configuration
        console.print("Let's set up your LLM providers.\n")

        config_lines = []

        # OpenAI
        if Confirm.ask("Configure OpenAI?", default=True):
            api_key = Prompt.ask("OpenAI API Key", password=True)
            org_id = Prompt.ask("Organization ID (optional)", default="")
            config_lines.append(f"OPENAI_API_KEY={api_key}")
            config_lines.append("OPENAI_ENABLED=true")
            if org_id:
                config_lines.append(f"OPENAI_ORGANIZATION_ID={org_id}")

        # DeepSeek
        if Confirm.ask("Configure DeepSeek?", default=False):
            api_key = Prompt.ask("DeepSeek API Key", password=True)
            config_lines.append(f"DEEPSEEK_API_KEY={api_key}")
            config_lines.append("DEEPSEEK_ENABLED=true")

        # Claude/Anthropic
        if Confirm.ask("Configure Claude/Anthropic?", default=True):
            config_lines.append("CLAUDE_ENABLED=true")

            # Select authentication type
            console.print("\n[bold]Claude Authentication Options:[/bold]")
            console.print("  [1] API Key - Standard Anthropic API key")
            console.print("  [2] OAuth - Browser login (Claude Pro/Team subscription)")
            console.print("  [3] Session Key - From browser cookies after login")

            auth_choice = Prompt.ask(
                "Select authentication method",
                choices=["1", "2", "3"],
                default="1",
            )

            if auth_choice == "1":
                # API Key authentication
                config_lines.append("CLAUDE_AUTH_TYPE=api_key")
                api_key = Prompt.ask("Anthropic API Key", password=True)
                config_lines.append(f"CLAUDE_API_KEY={api_key}")
            elif auth_choice == "2":
                # OAuth authentication
                config_lines.append("CLAUDE_AUTH_TYPE=oauth")
                callback_port = Prompt.ask(
                    "OAuth callback port",
                    default="8080",
                )
                config_lines.append(f"CLAUDE_OAUTH_CALLBACK_PORT={callback_port}")
                console.print("[yellow]Note: Browser will open for login when Claude provider is used.[/yellow]")
            elif auth_choice == "3":
                # Session Key authentication
                config_lines.append("CLAUDE_AUTH_TYPE=session_key")
                console.print("\n[dim]To get your session key:[/dim]")
                console.print("[dim]1. Open claude.ai in your browser and log in[/dim]")
                console.print("[dim]2. Open Developer Tools (F12) > Application > Cookies[/dim]")
                console.print("[dim]3. Find 'sessionKey' cookie and copy its value[/dim]")
                session_key = Prompt.ask("Session Key", password=True)
                config_lines.append(f"CLAUDE_SESSION_KEY={session_key}")

            # Optional subscription info
            if auth_choice in ["2", "3"]:
                if Confirm.ask("Enter subscription details? (optional)", default=False):
                    email = Prompt.ask("Account email", default="")
                    if email:
                        config_lines.append(f"CLAUDE_ACCOUNT_EMAIL={email}")
                    sub_type = Prompt.ask(
                        "Subscription type",
                        choices=["free", "pro", "team", "enterprise"],
                        default="pro",
                    )
                    config_lines.append(f"CLAUDE_SUBSCRIPTION_TYPE={sub_type}")
                    daily_limit = Prompt.ask("Daily message limit (leave empty for auto)", default="")
                    if daily_limit:
                        config_lines.append(f"CLAUDE_DAILY_MESSAGE_LIMIT={daily_limit}")

        # Context sharing
        if Confirm.ask("Enable context sharing?", default=False):
            config_lines.append("CONTEXT_ENABLED=true")
            share_type = Prompt.ask(
                "Sharing type",
                choices=["file", "cloud_bucket"],
                default="file",
            )
            config_lines.append(f"CONTEXT_SHARING_TYPE={share_type}")

            if share_type == "file":
                share_path = Prompt.ask(
                    "Shared directory path",
                    default=".overnight/shared",
                )
                config_lines.append(f"CONTEXT_SHARED_PATH={share_path}")
            elif share_type == "cloud_bucket":
                bucket = Prompt.ask("S3 Bucket name")
                config_lines.append(f"CONTEXT_BUCKET_NAME={bucket}")

        # Save configuration
        if config_lines:
            with open(".env", "w") as f:
                f.write("\n".join(config_lines))
            console.print("[green]Configuration saved to .env[/green]")

        self._settings = get_settings()

    async def _initialize_system(self) -> None:
        """Initialize the orchestrator and services."""
        console.print("\n[bold]Initializing System...[/bold]")

        from src.infrastructure.storage.sqlite_repository import SQLiteSessionRepository
        from src.infrastructure.storage.memory_event_bus import InMemoryEventBus
        from src.infrastructure.storage.memory_context_store import InMemoryContextStore
        from src.infrastructure.context_sharing.sharing_factory import ContextSharingFactory
        from src.application.services.orchestrator import Orchestrator

        # Initialize repository
        db_path = self._settings.work_dir / "overnight.db"
        repository = SQLiteSessionRepository(db_path)
        await repository.initialize()

        # Initialize event bus and context store
        event_bus = InMemoryEventBus()
        context_store = InMemoryContextStore()

        # Initialize context sharing if enabled
        context_sync = await ContextSharingFactory.create_and_initialize(
            self._settings.context_sharing
        )

        # Create orchestrator
        self._orchestrator = Orchestrator(
            settings=self._settings,
            repository=repository,
            context_store=context_store,
            event_bus=event_bus,
            context_sync=context_sync,
        )

        await self._orchestrator.initialize()
        await self._orchestrator.start()

        providers = self._orchestrator.available_providers
        console.print(f"[green]System initialized with providers: {', '.join(providers)}[/green]")

    async def _show_main_menu(self) -> str:
        """Show main menu and get user choice."""
        console.print("\n[bold]Main Menu[/bold]")

        table = Table(show_header=False, box=None)
        table.add_row("[1]", "Create new session")
        table.add_row("[2]", "View status")
        table.add_row("[3]", "Launch dashboard")
        table.add_row("[4]", "Process question")
        table.add_row("[5]", "Settings")
        table.add_row("[q]", "Quit")
        console.print(table)

        choice = Prompt.ask(
            "Select option",
            choices=["1", "2", "3", "4", "5", "q"],
            default="2",
        )

        mapping = {
            "1": "create",
            "2": "status",
            "3": "dashboard",
            "4": "process",
            "5": "settings",
            "q": "quit",
        }

        return mapping[choice]

    async def _create_session_flow(self) -> None:
        """Flow for creating a new session."""
        console.print("\n[bold]Create New Session[/bold]\n")

        providers = self._orchestrator.available_providers
        if not providers:
            console.print("[red]No providers available. Check configuration.[/red]")
            return

        # Select provider
        console.print("Available providers:", ", ".join(providers))
        provider_name = Prompt.ask(
            "Select provider",
            choices=providers,
            default=providers[0],
        )

        # Provider enum mapping
        provider_map = {
            "openai": LLMProvider.OPENAI,
            "deepseek": LLMProvider.DEEPSEEK,
            "claude": LLMProvider.CLAUDE,
            "anthropic": LLMProvider.ANTHROPIC,
        }
        provider = provider_map.get(provider_name, LLMProvider.OPENAI)

        # Session name
        name = Prompt.ask("Session name", default="")

        # Initial prompt
        console.print("Enter initial prompt/requirements (empty line to finish):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        initial_prompt = "\n".join(lines)

        # Working directory
        working_dir = Prompt.ask(
            "Working directory",
            default=str(Path.cwd()),
        )

        # Create session
        session_id = await self._orchestrator.create_session(
            provider=provider,
            name=name,
            initial_prompt=initial_prompt,
            working_directory=working_dir,
        )

        console.print(f"[green]Session created: {session_id}[/green]")

        # Start session?
        if Confirm.ask("Start session now?", default=True):
            await self._orchestrator.start_session(session_id)
            console.print("[green]Session started.[/green]")

    async def _show_status(self) -> None:
        """Show status of all sessions."""
        console.print("\n[bold]Session Status[/bold]\n")

        statuses = self._orchestrator.get_all_sessions_status()

        if not statuses:
            console.print("[yellow]No active sessions.[/yellow]")
            return

        table = Table(title="Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Provider")
        table.add_column("Status", style="bold")
        table.add_column("Questions")
        table.add_column("Accuracy")
        table.add_column("Strategy")

        for status in statuses:
            status_color = {
                "running": "green",
                "paused": "yellow",
                "waiting_rate_limit": "yellow",
                "completed": "blue",
                "failed": "red",
            }.get(status["status"], "white")

            accuracy = status["metrics"]["prediction_accuracy"]
            accuracy_str = f"{accuracy:.1%}" if accuracy else "N/A"

            table.add_row(
                status["session_id"][:12],
                status.get("name", ""),
                status["provider"],
                f"[{status_color}]{status['status']}[/{status_color}]",
                str(status["metrics"]["questions_processed"]),
                accuracy_str,
                status["current_strategy"],
            )

        console.print(table)

    async def _launch_dashboard(self) -> None:
        """Launch the TUI dashboard."""
        console.print("\n[bold]Launching Dashboard...[/bold]")

        try:
            from src.presentation.dashboard.app import DashboardApp

            app = DashboardApp(self._orchestrator)
            await app.run_async()
        except ImportError:
            console.print("[yellow]Dashboard requires 'textual' package.[/yellow]")
            console.print("Install with: pip install textual")

    async def _process_question_flow(self) -> None:
        """Flow for processing a question in a session."""
        statuses = self._orchestrator.get_all_sessions_status()

        if not statuses:
            console.print("[yellow]No active sessions. Create one first.[/yellow]")
            return

        # Select session
        session_ids = [s["session_id"] for s in statuses]
        console.print("Available sessions:", ", ".join(sid[:8] for sid in session_ids))

        session_id = Prompt.ask("Session ID (first 8 chars)")

        # Find matching session
        match = None
        for sid in session_ids:
            if sid.startswith(session_id):
                match = sid
                break

        if not match:
            console.print("[red]Session not found.[/red]")
            return

        # Get question
        question = Prompt.ask("Enter your question")

        # Process
        console.print("\n[dim]Processing...[/dim]")
        answer = await self._orchestrator.process_question(match, question)

        console.print("\n[bold]Answer:[/bold]")
        console.print(Panel(answer))

    async def _modify_settings(self) -> None:
        """Modify settings."""
        console.print("\n[bold]Settings[/bold]")
        console.print("Settings modification not yet implemented.")
        console.print("Edit .env file directly for now.")

    async def _shutdown(self) -> None:
        """Shutdown the system."""
        console.print("\n[bold]Shutting down...[/bold]")

        if self._orchestrator:
            await self._orchestrator.stop()

        console.print("[green]Goodbye![/green]")


def main() -> None:
    """Entry point for CLI."""
    setup_logging(level="INFO")
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
