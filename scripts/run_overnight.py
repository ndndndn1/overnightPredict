#!/usr/bin/env python3
"""
Run overnight autonomous coding session.

This script demonstrates the full OvernightPredict workflow:
1. Initialize a project with components
2. Start parallel sessions
3. Monitor progress
4. Generate code artifacts
"""

import asyncio
import signal
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from src.core.config import get_settings
from src.core.models import ProjectContext
from src.sessions.orchestrator import SessionOrchestrator
from src.utils.logging import setup_logging

console = Console()


async def main() -> None:
    """Run the overnight coding session."""
    settings = get_settings()
    setup_logging(settings)

    # Create project context
    project = ProjectContext(
        name="EnterpriseApp",
        description="A full-featured enterprise application with authentication, "
        "API gateway, and microservices architecture",
        target_languages=["python", "typescript"],
        architecture_type="microservices",
        requirements=[
            "High availability",
            "Scalable to 10k concurrent users",
            "RESTful API design",
            "OAuth2 authentication",
            "PostgreSQL database",
            "Redis caching",
            "Docker deployment",
        ],
        pending_components=[
            "authentication_service",
            "user_management_service",
            "api_gateway",
            "notification_service",
            "analytics_service",
            "config_service",
            "database_migrations",
            "caching_layer",
            "logging_middleware",
            "monitoring_dashboard",
        ],
    )

    console.print(
        Panel.fit(
            "[bold blue]OvernightPredict[/bold blue]\n"
            f"Project: [green]{project.name}[/green]\n"
            f"Components: {len(project.pending_components)}\n"
            f"Languages: {', '.join(project.target_languages)}",
            title="Starting Overnight Coding",
        )
    )

    # Create orchestrator
    orchestrator = SessionOrchestrator(settings)
    await orchestrator.initialize_project(project)

    # Handle shutdown gracefully
    shutdown_event = asyncio.Event()

    def signal_handler(sig: int, frame: object) -> None:
        console.print("\n[yellow]Received shutdown signal...[/yellow]")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start sessions
    await orchestrator.start(initial_sessions=3)

    console.print("[green]Sessions started! Monitoring progress...[/green]\n")

    # Monitor loop
    try:
        while not shutdown_event.is_set():
            # Get status
            status = await orchestrator.get_orchestrator_status()
            metrics = await orchestrator.get_global_metrics()

            # Build status table
            table = Table(title="Session Status")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Running Sessions", str(status["sessions"]["running"]))
            table.add_row("Total Sessions", str(status["sessions"]["total"]))
            table.add_row("Questions", str(metrics["total_questions"]))
            table.add_row("Answers", str(metrics["total_answers"]))
            table.add_row("Code Artifacts", str(metrics["total_artifacts"]))
            table.add_row("Predictions", str(metrics["total_predictions"]))
            table.add_row("Accuracy", f"{metrics['overall_accuracy']:.1%}")

            # Clear and print
            console.clear()
            console.print(
                Panel.fit(
                    f"[bold blue]OvernightPredict[/bold blue] - {project.name}",
                    title="Running",
                )
            )
            console.print(table)
            console.print("\n[dim]Press Ctrl+C to stop[/dim]")

            # Check completion
            if status["sessions"]["running"] == 0:
                console.print("\n[green]All sessions completed![/green]")
                break

            # Wait before next update
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

    finally:
        # Shutdown
        console.print("\n[yellow]Shutting down sessions...[/yellow]")
        await orchestrator.stop(graceful=True)

        # Final summary
        metrics = await orchestrator.get_global_metrics()

        console.print("\n")
        summary = Table(title="Final Summary")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")

        summary.add_row("Total Questions", str(metrics["total_questions"]))
        summary.add_row("Total Answers", str(metrics["total_answers"]))
        summary.add_row("Code Artifacts Generated", str(metrics["total_artifacts"]))
        summary.add_row("Predictions Made", str(metrics["total_predictions"]))
        summary.add_row("Accurate Predictions", str(metrics["accurate_predictions"]))
        summary.add_row("Overall Accuracy", f"{metrics['overall_accuracy']:.1%}")
        summary.add_row("Sessions Completed", str(metrics["sessions_completed"]))

        console.print(summary)
        console.print("\n[green]Overnight coding complete![/green]")


if __name__ == "__main__":
    asyncio.run(main())
