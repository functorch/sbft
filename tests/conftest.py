import pytest


# Pytest session fixtures---------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    """Called to allow custom command line options to be added to pytest.

    Called once at the beginning of a test run.
    """
    # --all option skips any other options
    parser.addoption(
        "--all",
        action="store_true",
        dest="ALL",
        help="Skip all other options and run all tests.",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Called to perform the setup phase.

    Here are the rules:
    --all: Run all tests

    See pyproject.toml for the purpose of each mark.
    """
    # If --all is set, run all tests
    if item.config.getoption("--all"):
        # Do not process any other options
        return
