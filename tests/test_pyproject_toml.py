import typing as t
from itertools import chain

import pytest
import toml

from sbft import REPO_ROOT, __version__


@pytest.fixture()
def pyproject_config() -> dict[str, t.Any]:
    pyproject_file_path = REPO_ROOT / "pyproject.toml"
    return toml.load(pyproject_file_path)


@pytest.fixture()
def all_dependencies(pyproject_config: dict[str, t.Any]) -> dict[str, t.Any]:
    return pyproject_config["tool"]["poetry"]["dependencies"]


@pytest.fixture()
def all_extra_dependencies(pyproject_config: dict[str, t.Any]) -> list[str]:
    extras = pyproject_config["tool"]["poetry"]["extras"]
    return list(chain.from_iterable(extras.values()))


def test_pyproject_optional_present_in_option(
    all_dependencies: dict[str, t.Any], all_extra_dependencies: list[str]
) -> None:
    """Ensure that the optional dependencies are present in the at least one of the
    options section.

    If this test fails, it means that there is some dependency with optional = true, but
    will never get installed because no option in tool.poetry.extras requires it.
    """
    for dependency, options in all_dependencies.items():
        if isinstance(options, dict) and "optional" in options and options["optional"]:
            assert dependency in all_extra_dependencies


def test_pyproject_install_all(
    pyproject_config: dict[str, t.Any], all_dependencies: dict[str, t.Any]
) -> None:
    """Installing using `pip install 'sbft[all]'` should install all dependencies."""
    extras = pyproject_config["tool"]["poetry"]["extras"]
    all_extras: dict[str, str] = {k: v for k, v in extras.items() if k != "all"}
    all_extra_dependencies = chain.from_iterable(all_extras.values())
    extras_all = extras["all"]
    assert set(all_extra_dependencies) == set(extras_all)
    # If you specify a dependency in the extras section, make sure you mention it in
    # the actual dependencies section with the version number.
    for extra in all_extras.values():
        for dependency in extra:
            assert dependency in all_dependencies


def test_sbft_version(pyproject_config: dict[str, t.Any]) -> None:
    """Ensure that the version of sbft in sbft/__init__.py is the same as in
    pyproject.toml.
    """
    poetry = pyproject_config["tool"]["poetry"]
    sbft_version = poetry["version"]
    assert sbft_version == __version__
