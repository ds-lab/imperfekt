from rich.console import Console
from rich.panel import Panel

console = Console()


def rich_error(msg):
    panel = Panel(f"[bold red]🚩 ERROR[/bold red]\n{msg}", border_style="red")
    console.print(panel)


def rich_warning(msg, surpress_warning=False):
    panel = Panel(f"[bold yellow]⚠️ WARNING[/bold yellow]\n{msg}", border_style="yellow")
    if not surpress_warning:
        console.print(panel)


def rich_info(msg):
    panel = Panel(f"[bold blue]ℹ️ INFO[/bold blue]\n{msg}", border_style="blue")
    console.print(panel)
