"""Command-line interface for OvernightPredict."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.core.config import Settings, get_settings
from src.core.models import ProjectContext
from src.sessions.orchestrator import SessionOrchestrator
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

    # API key status
    api_table = Table(title="API Keys")
    api_table.add_column("Provider", style="cyan")
    api_table.add_column("Status", style="green")

    anthropic_status = "[green]✓ Configured[/green]" if settings.anthropic_api_key else "[red]✗ Not Set[/red]"
    openai_status = "[green]✓ Configured[/green]" if settings.openai_api_key else "[red]✗ Not Set[/red]"

    api_table.add_row("Anthropic", anthropic_status)
    api_table.add_row("OpenAI", openai_status)

    console.print()
    console.print(api_table)


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
    console.print("1. Edit .env and add your API keys")
    console.print("2. Run 'overnight start' to begin coding")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
