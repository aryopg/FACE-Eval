"""Logging utilities with rich formatting."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table


class Logger:
    """Rich-based logger with verbose mode support."""

    def __init__(self, verbose: bool = True):
        """Initialize logger.

        Args:
            verbose: If True, show detailed logs. If False, only show important info.
        """
        self.verbose = verbose
        self.console = Console()

    def info(self, message: str, force: bool = False):
        """Log info message.

        Args:
            message: Message to log.
            force: Show even if not in verbose mode.
        """
        if self.verbose or force:
            self.console.print(f"[cyan]ℹ[/cyan] {message}")

    def success(self, message: str, force: bool = False):
        """Log success message.

        Args:
            message: Message to log.
            force: Show even if not in verbose mode.
        """
        if self.verbose or force:
            self.console.print(f"[green]✓[/green] {message}")

    def warning(self, message: str, force: bool = True):
        """Log warning message.

        Args:
            message: Message to log.
            force: Show even if not in verbose mode (default True for warnings).
        """
        if self.verbose or force:
            self.console.print(f"[yellow]⚠[/yellow] {message}")

    def error(self, message: str):
        """Log error message (always shown).

        Args:
            message: Message to log.
        """
        self.console.print(f"[red]✗[/red] {message}", style="bold red")

    def header(self, title: str):
        """Print a header panel.

        Args:
            title: Header title.
        """
        if self.verbose:
            self.console.print(Panel(f"[bold]{title}[/bold]", style="blue"))

    def table(self, title: str, data: dict, force: bool = False):
        """Print a table of key-value pairs.

        Args:
            title: Table title.
            data: Dictionary of data to display.
            force: Show even if not in verbose mode.
        """
        if self.verbose or force:
            table = Table(title=title, show_header=True, header_style="bold magenta")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")

            for key, value in data.items():
                table.add_row(str(key), str(value))

            self.console.print(table)

    def progress(self, description: str = "Processing...") -> Progress:
        """Create a progress bar.

        Args:
            description: Progress description.

        Returns:
            Rich Progress object.
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        )

    def print(self, *args, **kwargs):
        """Direct console print."""
        self.console.print(*args, **kwargs)

    def rule(self, title: str | None = None):
        """Print a horizontal rule.

        Args:
            title: Optional title for the rule.
        """
        if self.verbose:
            self.console.rule(title, style="dim")


# Global logger instance
_logger: Logger | None = None


def get_logger(verbose: bool = True) -> Logger:
    """Get or create the global logger instance.

    The first call creates the singleton; subsequent calls update `verbose` so
    that e.g. `get_logger(verbose=False)` after startup correctly silences logs.

    Args:
        verbose: Verbose mode setting.

    Returns:
        Logger instance.
    """
    global _logger
    if _logger is None:
        _logger = Logger(verbose=verbose)
    else:
        _logger.verbose = verbose
    return _logger


def set_verbose(verbose: bool):
    """Set global verbose mode.

    Args:
        verbose: Verbose mode setting.
    """
    global _logger
    if _logger is not None:
        _logger.verbose = verbose
    else:
        _logger = Logger(verbose=verbose)
