"""Contains the subcommands for `sbft run`."""

import importlib

import typer
from omegaconf import DictConfig, OmegaConf

from sbft import external_utils as extu

app = typer.Typer()


@app.command()
def agent(
    agent_name: str = typer.Argument(
        ...,
        help="The name of the agent to run."
        + " Run `sbft list` for a list of available agents.",
    ),
    config_path: str = typer.Argument(
        ...,
        help="path to config file relative to current working directory",
    ),
    overrides: list[str] = typer.Option(
        [], "-o", "--override", help="override possibly nested keys in config"
    ),
):
    """Run an agent with the given configuration.

    Args:
    -----
        agent_name: The name of the agent to run.
            Run `sbft list` for a list of available agents.
        config_path: path to config file relative to current working directory
        overrides: override possibly nested keys in config. For example,
            `--override a.b.c=1` will override the value of the key `c` in the
            nested dictionary at the key `b` in the dictionary at the key `a`.

    Examples:
    ---------
    .. code-block:: bash
    >>> sbft run agent sac experiments/sac/humanoid.yml
    """
    config_ = OmegaConf.load(config_path)
    assert isinstance(config_, DictConfig)
    overrides_config = overrides_to_config(overrides, config_)
    config_ = OmegaConf.merge(config_, overrides_config)
    assert isinstance(config_, DictConfig)
    # Import the config module from the corresponding agent module
    config_module = importlib.import_module(f"sbft.agents.{agent_name}.config")
    # The config module should contain a function called load_config that takes in a
    # dictionary and returns the proper config object for the agent
    # Import the train module from the corresponding agent module
    load_config_func = getattr(config_module, "load_config")
    config_dict = OmegaConf.to_container(config_, resolve=True)
    config_obj = load_config_func(config_dict)
    train_module = importlib.import_module(f"sbft.agents.{agent_name}.train")
    # The train module must contain a function called `run` that takes an
    # ExperimentConfig as its only argument
    run_func = getattr(train_module, "run")
    # Run the experiment
    run_func(config_obj)


def overrides_to_config(overrides: list[str], config_: DictConfig) -> DictConfig:
    """Convert a list of overrides to a config.

    Args:
    -----
        overrides: A list of overrides. For example, `--override a.b.c=1` will
            override the value of the key `c` in the nested dictionary at the key
            `b` in the dictionary at the key `a`.
        config_: The config to override.

    Returns:
    --------
        The config with the overrides applied.

    Examples:
    ---------
    >>> overrides = ["a.b.c=1", "a.b.d=2"]
    >>> config_ = OmegaConf.create(dict(a=dict(b=dict(c=0, d=0))))
    >>> overrides_to_config(overrides, config_)
    {'a': {'b': {'c': 1, 'd': 2}}}
    """
    if overrides:
        for override in overrides:
            key = override.split("=")[0]
            assert extu.nested_dict_contains_dot_key(
                config_, key
            ), f"{key} not in config:\n{OmegaConf.to_yaml(config_)}"
        return OmegaConf.from_dotlist(overrides)
    else:
        return OmegaConf.create({})
