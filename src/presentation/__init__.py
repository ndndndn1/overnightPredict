"""
Presentation Layer - CLI and Dashboard interfaces.

This layer contains:
- CLI: Command-line interface for session management
- Dashboard: Real-time TUI dashboard for monitoring
- Controllers: Interactive session control
"""

from src.presentation.cli.main_cli import CLI
from src.presentation.dashboard.app import DashboardApp

__all__ = [
    "CLI",
    "DashboardApp",
]
