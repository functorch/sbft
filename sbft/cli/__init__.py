import rich
import typer
from rich.panel import Panel
from rich.table import Table

from sbft import REPO_ROOT, __version__
from sbft.cli import run

cli_app = typer.Typer(help="CLI interface for sbft.")
cli_app.add_typer(run.app, name="run")


@cli_app.command()
def version():
    """CLI interface for sbft."""
    typer.echo(f"sbft version {__version__}")
    typer.Exit()


@cli_app.command()
def list():
    """List available agents."""
    agents = Table(show_header=False, header_style="bold magenta", expand=True)
    agents.add_column("Agent", justify="left", style="cyan", no_wrap=True)
    for train_module in (REPO_ROOT / "sbft" / "agents").glob("*/train.py"):
        agent_module = train_module.parent
        if not agent_module.name.startswith("_"):
            agents.add_row(agent_module.name)
    rich.print(Panel(agents, title="Available Agents", title_align="center"))
    typer.Exit()
