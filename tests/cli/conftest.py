import pytest
from typer.testing import CliRunner


@pytest.fixture()
def cli_runner() -> CliRunner:
    """Fixture to create a CliRunner for testing."""
    return CliRunner()
