"""Progress reporting utilities for consistent output across CLI and MCP."""

import sys
from typing import Any, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

# Global console instance for consistent output
console = Console()
error_console = Console(stderr=True)


def create_progress_bar(description: str = "Processing", total: Optional[int] = None) -> tuple[Progress, TaskID]:
    """Create a progress bar for indexing operations.
    
    Args:
        description: Description text for the progress bar
        total: Total number of items (None for indeterminate)
        
    Returns:
        Tuple of (Progress instance, TaskID) for updating
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    
    task_id = progress.add_task(description, total=total)
    return progress, task_id


def update_progress(
    progress: Progress,
    task_id: TaskID,
    advance: int = 1,
    description: Optional[str] = None,
) -> None:
    """Update progress bar.
    
    Args:
        progress: Progress instance from create_progress_bar
        task_id: Task ID from create_progress_bar
        advance: Number of steps to advance (default 1)
        description: Optional new description text
    """
    if description:
        progress.update(task_id, description=description)
    progress.advance(task_id, advance)


def log_info(message: str, **kwargs: Any) -> None:
    """Log an info message.
    
    Args:
        message: Message to log
        **kwargs: Additional arguments passed to rich console
    """
    console.print(f"[blue]ℹ[/blue] {message}", **kwargs)


def log_warning(message: str, **kwargs: Any) -> None:
    """Log a warning message.
    
    Args:
        message: Warning message to log
        **kwargs: Additional arguments passed to rich console
    """
    console.print(f"[yellow]⚠[/yellow] {message}", **kwargs)


def log_error(message: str, **kwargs: Any) -> None:
    """Log an error message.
    
    Args:
        message: Error message to log
        **kwargs: Additional arguments passed to rich console
    """
    error_console.print(f"[red]✗[/red] {message}", **kwargs)


def log_success(message: str, **kwargs: Any) -> None:
    """Log a success message.
    
    Args:
        message: Success message to log
        **kwargs: Additional arguments passed to rich console
    """
    console.print(f"[green]✓[/green] {message}", **kwargs)