"""
TUI Dashboard Application using Textual.

Provides real-time monitoring and control of all sessions.
"""

from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, DataTable, Button, Input, Log
from textual.reactive import reactive
from textual.timer import Timer

from src.application.services.orchestrator import Orchestrator


class SessionPanel(Static):
    """Panel showing session status."""

    def __init__(self, session_data: dict, **kwargs):
        super().__init__(**kwargs)
        self.session_data = session_data

    def compose(self) -> ComposeResult:
        """Compose the session panel."""
        status = self.session_data.get("status", "unknown")
        metrics = self.session_data.get("metrics", {})

        status_color = {
            "running": "green",
            "paused": "yellow",
            "waiting_rate_limit": "orange",
            "completed": "blue",
            "failed": "red",
        }.get(status, "white")

        content = f"""[bold]{self.session_data.get('session_id', 'Unknown')[:12]}[/bold]
Provider: {self.session_data.get('provider', 'N/A')}
Status: [{status_color}]{status}[/{status_color}]
Questions: {metrics.get('questions_processed', 0)}
Predictions: {metrics.get('predictions_made', 0)}
Accuracy: {metrics.get('prediction_accuracy', 0):.1%}
Strategy: {self.session_data.get('current_strategy', 'default')}"""

        yield Static(content, classes="session-info")


class SessionsTable(DataTable):
    """Table showing all sessions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cursor_type = "row"

    def on_mount(self) -> None:
        """Set up the table columns."""
        self.add_columns(
            "ID", "Name", "Provider", "Status",
            "Questions", "Predictions", "Accuracy", "Strategy"
        )


class LogPanel(Log):
    """Panel for activity logs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs, highlight=True, markup=True)


class ControlPanel(Static):
    """Panel with control buttons."""

    def compose(self) -> ComposeResult:
        """Compose control panel."""
        yield Horizontal(
            Button("Start All", id="start-all", variant="success"),
            Button("Stop All", id="stop-all", variant="error"),
            Button("Refresh", id="refresh", variant="primary"),
            Button("New Session", id="new-session"),
            classes="button-row",
        )


class QuestionInput(Static):
    """Input panel for sending questions."""

    def compose(self) -> ComposeResult:
        """Compose question input."""
        yield Horizontal(
            Input(placeholder="Enter question...", id="question-input"),
            Button("Send", id="send-question", variant="primary"),
            classes="input-row",
        )


class DashboardApp(App):
    """
    TUI Dashboard for OvernightPredict.

    Provides real-time monitoring and control of sessions.
    """

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-gutter: 1;
    }

    #sessions-container {
        column-span: 2;
        row-span: 2;
        border: solid green;
    }

    #log-container {
        border: solid blue;
    }

    #control-container {
        border: solid yellow;
    }

    .button-row {
        height: 3;
        align: center middle;
    }

    .button-row Button {
        margin: 0 1;
    }

    .input-row {
        height: 3;
    }

    .input-row Input {
        width: 80%;
    }

    .session-info {
        padding: 1;
    }

    DataTable {
        height: 100%;
    }

    Log {
        height: 100%;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("n", "new_session", "New Session"),
        ("s", "start_all", "Start All"),
        ("x", "stop_all", "Stop All"),
    ]

    selected_session: reactive[Optional[str]] = reactive(None)

    def __init__(self, orchestrator: Orchestrator, **kwargs):
        super().__init__(**kwargs)
        self.orchestrator = orchestrator
        self._refresh_timer: Optional[Timer] = None

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        yield Header(show_clock=True)

        with Container(id="sessions-container"):
            yield Static("[bold]Sessions[/bold]", classes="title")
            yield SessionsTable(id="sessions-table")

        with Container(id="log-container"):
            yield Static("[bold]Activity Log[/bold]", classes="title")
            yield LogPanel(id="activity-log")

        with Container(id="control-container"):
            yield Static("[bold]Controls[/bold]", classes="title")
            yield ControlPanel()
            yield QuestionInput()

        yield Footer()

    def on_mount(self) -> None:
        """Set up when app mounts."""
        self._refresh_data()

        # Set up auto-refresh
        self._refresh_timer = self.set_interval(2.0, self._refresh_data)

    def _refresh_data(self) -> None:
        """Refresh session data."""
        table = self.query_one("#sessions-table", SessionsTable)
        log = self.query_one("#activity-log", LogPanel)

        # Clear and repopulate table
        table.clear()

        statuses = self.orchestrator.get_all_sessions_status()

        for status in statuses:
            metrics = status.get("metrics", {})
            accuracy = metrics.get("prediction_accuracy", 0)

            status_display = status.get("status", "unknown")
            if status_display == "running":
                status_display = "[green]running[/green]"
            elif status_display == "paused":
                status_display = "[yellow]paused[/yellow]"
            elif status_display == "failed":
                status_display = "[red]failed[/red]"

            table.add_row(
                status.get("session_id", "")[:12],
                status.get("name", ""),
                status.get("provider", ""),
                status_display,
                str(metrics.get("questions_processed", 0)),
                str(metrics.get("predictions_made", 0)),
                f"{accuracy:.1%}" if accuracy else "N/A",
                status.get("current_strategy", "default"),
            )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        log = self.query_one("#activity-log", LogPanel)

        if button_id == "start-all":
            log.write_line("[green]Starting all sessions...[/green]")
            await self.orchestrator.start_all_sessions()
            log.write_line("[green]All sessions started.[/green]")

        elif button_id == "stop-all":
            log.write_line("[yellow]Stopping all sessions...[/yellow]")
            await self.orchestrator.stop_all_sessions()
            log.write_line("[yellow]All sessions stopped.[/yellow]")

        elif button_id == "refresh":
            self._refresh_data()
            log.write_line("[blue]Data refreshed.[/blue]")

        elif button_id == "new-session":
            log.write_line("[cyan]New session creation not yet implemented in dashboard.[/cyan]")
            log.write_line("[cyan]Use CLI: python -m src.presentation.cli.main_cli[/cyan]")

        elif button_id == "send-question":
            await self._send_question()

    async def _send_question(self) -> None:
        """Send question to selected session."""
        log = self.query_one("#activity-log", LogPanel)
        input_widget = self.query_one("#question-input", Input)
        question = input_widget.value.strip()

        if not question:
            log.write_line("[red]Please enter a question.[/red]")
            return

        if not self.selected_session:
            log.write_line("[red]Please select a session first.[/red]")
            return

        log.write_line(f"[dim]Sending question to {self.selected_session[:8]}...[/dim]")

        try:
            answer = await self.orchestrator.process_question(
                self.selected_session,
                question,
            )
            log.write_line(f"[green]Answer:[/green] {answer[:200]}...")
            input_widget.value = ""
        except Exception as e:
            log.write_line(f"[red]Error: {e}[/red]")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in sessions table."""
        table = self.query_one("#sessions-table", SessionsTable)
        log = self.query_one("#activity-log", LogPanel)

        if event.row_key:
            row_data = table.get_row(event.row_key)
            if row_data:
                self.selected_session = row_data[0]  # Session ID
                log.write_line(f"[cyan]Selected session: {self.selected_session}[/cyan]")

    def action_quit(self) -> None:
        """Quit the app."""
        self.exit()

    def action_refresh(self) -> None:
        """Refresh data."""
        self._refresh_data()

    def action_new_session(self) -> None:
        """Create new session."""
        log = self.query_one("#activity-log", LogPanel)
        log.write_line("[cyan]Use CLI to create new sessions.[/cyan]")

    async def action_start_all(self) -> None:
        """Start all sessions."""
        await self.orchestrator.start_all_sessions()

    async def action_stop_all(self) -> None:
        """Stop all sessions."""
        await self.orchestrator.stop_all_sessions()


def main():
    """Run dashboard standalone (for testing)."""
    print("Dashboard requires an orchestrator instance.")
    print("Use: python -m src.presentation.cli.main_cli")


if __name__ == "__main__":
    main()
