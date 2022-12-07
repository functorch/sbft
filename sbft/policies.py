"""Common utilities for policies / policy classes."""
import typing as t

import attrs
import numpy.typing as npt
import torch
from gymnasium import spaces as gym_spaces
from gymnasium.vector import VectorEnv

from sbft import api
from sbft import external_utils as extu


@attrs.define()
class ContinuousRandomPolicy(api.Policy):
    """A policy that generates random actions.

    Args:
    -----
        action_space_low: The lower bound of the action space.
        action_space_high: The upper bound of the action space.
        size: The size of the action space. If None, will be inferred from
            `action_space_high`.

    NOTE: action_space_high and action_space_low must be the spaces of a single env in
    the vectorized env.

    Example:
    --------
    >>> import torch
    ... from gymnasium.vector import make as vector_make
    ... from sbft.policies import ContinuousRandomPolicy
    ... env = vector_make("Pendulum-v1", 2)
    ... policy = ContinuousRandomPolicy.from_env(env)
    ... params = tuple()
    ... states, _ = env.reset(seed=0)
    ... generator = torch.Generator().manual_seed(0)
    ... actions = policy(params, states, generator=generator)
    ... assert torch.allclose(
    ...     actions, torch.tensor([[1.9850], [3.0729]]), atol=1e-4, rtol=0
    ... ), actions
    ... result = env.step(actions)
    ... assert isinstance(result, tuple)
    """

    action_space_high: float | npt.NDArray[t.Any]
    action_space_low: float | npt.NDArray[t.Any]

    size: tuple[int, ...] | torch.Size = attrs.field(default=None)

    _action_space_range: torch.Tensor | None = attrs.field(init=False, default=None)

    @classmethod
    def from_env(cls, env: VectorEnv) -> "ContinuousRandomPolicy":
        """Create a policy from an environment."""
        action_space: gym_spaces.Space[t.Any] = env.single_action_space
        assert isinstance(action_space, gym_spaces.Box)
        return cls(action_space.high, action_space.low, size=action_space.shape)

    def __attrs_post_init__(self) -> None:
        if self.size is None:
            if isinstance(self.action_space_high, float):
                self.size = torch.Size()
            else:
                self.size = torch.Size(self.action_space_high.shape)

    def __call__(
        self,
        params: tuple[torch.nn.Parameter, ...],
        states: torch.Tensor | t.Any,
        *,
        t: int | None = None,
        eval: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate random actions."""
        batch_size = states.shape[0]
        device = params[0].device if params else "cpu"
        random_unscaled_action = torch.rand(
            (batch_size, *self.size),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        if self._action_space_range is None:
            self._action_space_range = torch.as_tensor(
                self.action_space_high - self.action_space_low,
                device=device,
            )
        return extu.rescale_range(
            random_unscaled_action,
            old_range=1,
            new_range=self._action_space_range,
        )
