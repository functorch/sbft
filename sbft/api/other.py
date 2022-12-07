import abc
import typing as t

import torch
from gymnasium.core import Env as GymEnv

from sbft.api.memory import ExperienceT, Memory


class Policy(t.Protocol):
    """A Policy takes in the parameters of a function approximator and the states from
    an environment and returns the actions to take in those states."""

    def __call__(
        self,
        params: tuple[torch.nn.Parameter, ...],
        states: torch.Tensor,
        *,
        t: int | None = None,
        eval: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return actions to take in the given states for the given parameters.

        Args:
        -----
            params: The parameters of the function approximator.
            states: The states of the environment to take actions in.
            t: The global timestep. Can be used for epsilon-greedy policies, etc.
            eval: Whether or not to use the evaluation mode of the policy.
            generator: The generator to use for sampling random numbers.

        Returns:
        --------
            actions: The actions to take in the given states.
        """
        raise NotImplementedError


class Observer(t.Protocol[ExperienceT]):
    env: GymEnv[t.Any, t.Any]
    memory: Memory[ExperienceT]
    policy: Policy

    def observe(
        self,
        params: tuple[torch.nn.Parameter, ...],
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        raise NotImplementedError


class UI(abc.ABC):
    """A UI is used to display the training progress and results."""

    pass
