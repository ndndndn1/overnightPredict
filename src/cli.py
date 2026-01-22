"""Command-line interface for OvernightPredict."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table

from src.core.config import AuthMethod, Settings, get_settings
from src.core.models import ProjectContext
from src.sessions.orchestrator import SessionOrchestrator
from src.utils.ai_client import CredentialsManager
from src.utils.logging import setup_logging

app = typer.Typer(
    name="overnight",
    help="OvernightPredict - AI-powered autonomous code generation",
    add_completion=False,
)
console = Console()


@app.command()
def start(
    project_name: str = typer.Option(
        "MyProject",
        "--name", "-n",
        help="Project name",
    ),
    description: str = typer.Option(
        "Enterprise project built with AI",
        "--description", "-d",
        help="Project description",
    ),
    components: Optional[list[str]] = typer.Option(
        None,
        "--component", "-c",
        help="Components to implement (can be specified multiple times)",
    ),
    languages: Optional[list[str]] = typer.Option(
        None,
        "--language", "-l",
        help="Target languages (can be specified multiple times)",
    ),
    sessions: int = typer.Option(
        2,
        "--sessions", "-s",
        help="Number of parallel sessions",
        min=1,
        max=20,
    ),
    architecture: str = typer.Option(
        "microservices",
        "--arch", "-a",
        help="Architecture type",
    ),
) -> None:
    """Start overnight autonomous coding session."""
    settings = get_settings()
    setup_logging(settings)

    # Default components if not provided
    if not components:
        components = [
            "authentication",
            "user_management",
            "api_gateway",
            "database_layer",
            "caching",
            "logging",
            "monitoring",
        ]

    # Default languages if not provided
    if not languages:
        languages = ["python"]

    console.print(
        Panel.fit(
            f"[bold blue]OvernightPredict[/bold blue]\n"
            f"Starting autonomous coding for: [green]{project_name}[/green]\n"
            f"Sessions: {sessions} | Components: {len(components)}",
            title="Initializing",
        )
    )

    asyncio.run(
        run_overnight_session(
            settings=settings,
            project_name=project_name,
            description=description,
            components=list(components),
            languages=list(languages),
            num_sessions=sessions,
            architecture=architecture,
        )
    )


async def run_overnight_session(
    settings: Settings,
    project_name: str,
    description: str,
    components: list[str],
    languages: list[str],
    num_sessions: int,
    architecture: str,
) -> None:
    """Run the overnight coding session."""
    orchestrator = SessionOrchestrator(settings)

    # Create project context
    project = ProjectContext(
        name=project_name,
        description=description,
        target_languages=languages,
        architecture_type=architecture,
        pending_components=components,
    )

    await orchestrator.initialize_project(project)

    # Display initial status
    table = Table(title="Project Configuration")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Project Name", project_name)
    table.add_row("Description", description[:50] + "..." if len(description) > 50 else description)
    table.add_row("Architecture", architecture)
    table.add_row("Languages", ", ".join(languages))
    table.add_row("Components", str(len(components)))
    table.add_row("Parallel Sessions", str(num_sessions))

    console.print(table)
    console.print()

    # Start orchestrator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting sessions...", total=None)

        await orchestrator.start(initial_sessions=num_sessions)
        progress.update(task, description="Sessions running...")

        console.print("[green]Sessions started successfully![/green]")
        console.print("\nPress Ctrl+C to stop.\n")

        # Monitor loop
        try:
            while True:
                await asyncio.sleep(10)

                status = await orchestrator.get_orchestrator_status()
                metrics = await orchestrator.get_global_metrics()

                # Update progress display
                progress.update(
                    task,
                    description=f"Running | Sessions: {status['sessions']['running']} | "
                    f"Q&A: {metrics['total_questions']}/{metrics['total_answers']} | "
                    f"Artifacts: {metrics['total_artifacts']} | "
                    f"Accuracy: {metrics['overall_accuracy']:.1%}",
                )

                # Check if all done
                if (
                    not status["is_running"]
                    or status["sessions"]["running"] == 0
                ):
                    console.print("\n[green]All sessions completed![/green]")
                    break

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping sessions...[/yellow]")
            await orchestrator.stop(graceful=True)
            console.print("[green]Sessions stopped.[/green]")

    # Final summary
    metrics = await orchestrator.get_global_metrics()

    summary_table = Table(title="Session Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total Questions", str(metrics["total_questions"]))
    summary_table.add_row("Total Answers", str(metrics["total_answers"]))
    summary_table.add_row("Code Artifacts", str(metrics["total_artifacts"]))
    summary_table.add_row("Predictions Made", str(metrics["total_predictions"]))
    summary_table.add_row("Accurate Predictions", str(metrics["accurate_predictions"]))
    summary_table.add_row("Overall Accuracy", f"{metrics['overall_accuracy']:.1%}")
    summary_table.add_row("Sessions Completed", str(metrics["sessions_completed"]))

    console.print()
    console.print(summary_table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server host"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
) -> None:
    """Start the API server."""
    console.print(
        Panel.fit(
            f"[bold blue]OvernightPredict API Server[/bold blue]\n"
            f"Starting at http://{host}:{port}",
            title="Server",
        )
    )

    if reload:
        import uvicorn
        uvicorn.run(
            "src.api.server:create_app",
            host=host,
            port=port,
            reload=True,
            factory=True,
        )
    else:
        from src.api.server import run_server
        asyncio.run(run_server(host=host, port=port))


@app.command()
def status() -> None:
    """Show system status and configuration."""
    settings = get_settings()

    console.print(
        Panel.fit(
            "[bold blue]OvernightPredict[/bold blue]\n"
            "System Status and Configuration",
            title="Status",
        )
    )

    # Configuration table
    config_table = Table(title="Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Environment", settings.environment)
    config_table.add_row("AI Provider", settings.ai.primary_provider)
    config_table.add_row("AI Model", settings.ai.anthropic_model)
    config_table.add_row("Prediction Threshold", f"{settings.prediction.accuracy_threshold:.0%}")
    config_table.add_row("Lookahead Questions", str(settings.prediction.lookahead_count))
    config_table.add_row("Max Sessions", str(settings.sessions.max_sessions))
    config_table.add_row("Auto Scaling", "Enabled" if settings.sessions.auto_scale_enabled else "Disabled")
    config_table.add_row("Database", settings.storage.database_path)
    config_table.add_row("Log Level", settings.logging.level)

    console.print(config_table)

    # Authentication status
    creds = CredentialsManager.load_credentials()

    api_table = Table(title="Authentication")
    api_table.add_column("Provider", style="cyan")
    api_table.add_column("Method", style="magenta")
    api_table.add_column("Status", style="green")

    # Anthropic auth status
    if settings.anthropic_api_key:
        api_table.add_row("Anthropic", "API Key (env)", "[green]✓ Configured[/green]")
    elif settings.anthropic_session_token:
        api_table.add_row("Anthropic", "Session (env)", "[green]✓ Configured[/green]")
    elif creds.get("anthropic_api_key"):
        api_table.add_row("Anthropic", "API Key (stored)", "[green]✓ Configured[/green]")
    elif creds.get("anthropic_session_token"):
        api_table.add_row("Anthropic", "Session (stored)", "[green]✓ Configured[/green]")
    else:
        api_table.add_row("Anthropic", "-", "[red]✗ Not Set[/red]")

    # OpenAI auth status
    openai_status = "[green]✓ Configured[/green]" if settings.openai_api_key else "[red]✗ Not Set[/red]"
    api_table.add_row("OpenAI", "API Key", openai_status)

    console.print()
    console.print(api_table)

    console.print("\n[dim]Run 'overnight login' to configure authentication[/dim]")
    console.print("[dim]Run 'overnight auth-status' for detailed auth info[/dim]")


@app.command()
def init(
    output_dir: str = typer.Option(".", "--output", "-o", help="Output directory"),
) -> None:
    """Initialize a new project configuration."""
    import shutil
    from pathlib import Path

    output_path = Path(output_dir)

    # Copy example env file
    env_example = Path(__file__).parent.parent / ".env.example"
    if env_example.exists():
        shutil.copy(env_example, output_path / ".env")
        console.print("[green]Created .env file[/green]")

    # Create data directories
    (output_path / "data").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    (output_path / "data" / "checkpoints").mkdir(exist_ok=True)

    console.print("[green]Created data directories[/green]")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("1. Edit .env and add your API keys, or run 'overnight login' for session auth")
    console.print("2. Run 'overnight start' to begin coding")


@app.command()
def login(
    method: str = typer.Option(
        "session",
        "--method", "-m",
        help="Authentication method: 'api_key' or 'session'",
    ),
) -> None:
    """Login to Claude with API key or session credentials."""
    console.print(
        Panel.fit(
            "[bold blue]OvernightPredict Authentication[/bold blue]\n"
            "Configure your Claude API credentials",
            title="Login",
        )
    )

    if method == "api_key":
        _login_with_api_key()
    elif method == "session":
        _login_with_session()
    else:
        console.print(f"[red]Unknown method: {method}[/red]")
        console.print("Use 'api_key' or 'session'")
        raise typer.Exit(1)


def _login_with_api_key() -> None:
    """Login using API key."""
    console.print("\n[cyan]API Key Authentication[/cyan]")
    console.print("You can get an API key from: https://console.anthropic.com/\n")

    api_key = Prompt.ask(
        "Enter your Anthropic API key",
        password=True,
    )

    if not api_key:
        console.print("[red]API key cannot be empty[/red]")
        raise typer.Exit(1)

    # Validate API key format
    if not api_key.startswith("sk-ant-"):
        console.print("[yellow]Warning: API key doesn't match expected format (sk-ant-...)[/yellow]")
        if not Confirm.ask("Continue anyway?"):
            raise typer.Exit(1)

    # Save to credentials
    creds = CredentialsManager.load_credentials()
    creds["anthropic_api_key"] = api_key
    creds["auth_method"] = "api_key"
    CredentialsManager.save_credentials(creds)

    console.print("\n[green]✓ API key saved successfully![/green]")
    console.print(f"Credentials stored at: {CredentialsManager.get_credentials_path()}")


def _login_with_session() -> None:
    """Login using Claude subscription session token."""
    console.print("\n[cyan]Session Token Authentication[/cyan]")
    console.print("Use your Claude subscription account credentials.\n")

    console.print("[yellow]How to get your session token:[/yellow]")
    console.print("1. Go to https://claude.ai and log in")
    console.print("2. Open browser Developer Tools (F12)")
    console.print("3. Go to Application > Cookies > claude.ai")
    console.print("4. Copy the value of 'sessionKey' cookie\n")

    session_token = Prompt.ask(
        "Enter your session token (sessionKey cookie)",
        password=True,
    )

    if not session_token:
        console.print("[red]Session token cannot be empty[/red]")
        raise typer.Exit(1)

    # Optional: session key for additional auth
    console.print("\n[dim]Optional: You can also provide an Organization ID if applicable[/dim]")
    org_id = Prompt.ask(
        "Organization ID (press Enter to skip)",
        default="",
    )

    # Save to credentials
    creds = CredentialsManager.load_credentials()
    creds["anthropic_session_token"] = session_token
    creds["auth_method"] = "session_token"
    if org_id:
        creds["anthropic_org_id"] = org_id
    CredentialsManager.save_credentials(creds)

    console.print("\n[green]✓ Session credentials saved successfully![/green]")
    console.print(f"Credentials stored at: {CredentialsManager.get_credentials_path()}")

    # Test connection
    if Confirm.ask("\nTest connection now?", default=True):
        asyncio.run(_test_session_connection())


async def _test_session_connection() -> None:
    """Test the session connection."""
    from src.utils.ai_client import get_ai_client, clear_ai_client_cache

    console.print("\n[cyan]Testing connection...[/cyan]")

    try:
        clear_ai_client_cache()
        settings = get_settings()
        # Force session auth
        settings.ai.auth_method = AuthMethod.SESSION_TOKEN

        client = get_ai_client(settings)
        response = await client.generate(
            "Say 'Hello' in one word.",
            max_tokens=10,
        )

        console.print(f"[green]✓ Connection successful![/green]")
        console.print(f"Response: {response}")

    except Exception as e:
        console.print(f"[red]✗ Connection failed: {e}[/red]")
        console.print("\n[yellow]Please check your session token and try again.[/yellow]")


@app.command()
def logout() -> None:
    """Clear stored credentials."""
    console.print(
        Panel.fit(
            "[bold blue]OvernightPredict[/bold blue]\n"
            "Clearing stored credentials",
            title="Logout",
        )
    )

    creds_path = CredentialsManager.get_credentials_path()

    if not creds_path.exists():
        console.print("[yellow]No stored credentials found.[/yellow]")
        return

    if Confirm.ask("Are you sure you want to clear all stored credentials?"):
        CredentialsManager.clear_credentials()
        console.print("[green]✓ Credentials cleared successfully![/green]")
    else:
        console.print("[yellow]Cancelled.[/yellow]")


@app.command()
def auth_status() -> None:
    """Show current authentication status."""
    console.print(
        Panel.fit(
            "[bold blue]OvernightPredict[/bold blue]\n"
            "Authentication Status",
            title="Auth Status",
        )
    )

    settings = get_settings()
    creds = CredentialsManager.load_credentials()

    auth_table = Table(title="Authentication Configuration")
    auth_table.add_column("Source", style="cyan")
    auth_table.add_column("Type", style="magenta")
    auth_table.add_column("Status", style="green")

    # Check environment variables
    if settings.anthropic_api_key:
        auth_table.add_row(
            "Environment",
            "API Key",
            "[green]✓ ANTHROPIC_API_KEY set[/green]",
        )
    else:
        auth_table.add_row(
            "Environment",
            "API Key",
            "[dim]Not set[/dim]",
        )

    if settings.anthropic_session_token:
        auth_table.add_row(
            "Environment",
            "Session Token",
            "[green]✓ ANTHROPIC_SESSION_TOKEN set[/green]",
        )
    else:
        auth_table.add_row(
            "Environment",
            "Session Token",
            "[dim]Not set[/dim]",
        )

    # Check stored credentials
    if creds.get("anthropic_api_key"):
        auth_table.add_row(
            "Stored Credentials",
            "API Key",
            "[green]✓ Saved[/green]",
        )

    if creds.get("anthropic_session_token"):
        auth_table.add_row(
            "Stored Credentials",
            "Session Token",
            "[green]✓ Saved[/green]",
        )

    if not creds:
        auth_table.add_row(
            "Stored Credentials",
            "-",
            "[dim]No saved credentials[/dim]",
        )

    console.print(auth_table)

    # Show active auth method
    active_method = creds.get("auth_method", "api_key")
    console.print(f"\n[cyan]Active method:[/cyan] {active_method}")
    console.print(f"[cyan]Credentials file:[/cyan] {CredentialsManager.get_credentials_path()}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
