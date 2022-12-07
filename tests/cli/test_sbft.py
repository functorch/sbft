"""Tests for the command `sbft`."""
from typer.testing import CliRunner

from sbft import __version__
from sbft.cli import cli_app


def test_sbft_version(cli_runner: CliRunner) -> None:
    """Test the command `sbft --version`."""
    result = cli_runner.invoke(cli_app, ["--version"])
    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip() == f"sbft version {__version__}"
