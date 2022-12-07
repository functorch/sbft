import typing as t

import attrs
import numpy as np
import numpy.typing as npt
import torch
from gymnasium.vector import VectorEnv

from sbft import external_utils as extu
from sbft.api import Memory, Observer, Policy
from sbft.experience import VanillaExperience


@attrs.define()
class EpisodeReturnRecorder:
    """Records episode returns during training and prints them to console."""

    ui: extu.CliUI = attrs.field(factory=lambda: extu.null_object(extu.CliUI))

    _step: int = attrs.field(init=False, default=0)
    _episode_returns: npt.NDArray[np.float32] = attrs.field(init=False, default=None)

    def track(
        self, rewards: npt.NDArray[np.float32], dones: npt.NDArray[np.bool_]
    ) -> None:
        """Track the episode returns.

        Args:
        -----
            rewards: The rewards received in the current step.
            dones: The dones received in the current step.

        Effects:
        --------
            Updates the internal state of the recorder.
            On each episode completion, prints the episode return to console.

        Example:
        --------
        >>> recorder = EpisodeReturnRecorder(Console())
        ... recorder.track(np.array([1, 2, 3]), np.array([False, False, False]))
        ... recorder.track(np.array([1, 2, 3]), np.array([False, False, True]))
        ... # Prints Step 2: -> Avg Return: 6.00
        ... assert (recorder._episode_returns == np.array([2, 4, 0])).all()
        ... recorder.track(np.array([1, 2, 3]), np.array([False, True, False]))
        ... # Prints Step 3: -> Avg Return: 6.00
        ... assert (recorder._episode_returns == np.array([3, 0, 3])).all()
        ... recorder.track(np.array([1, 2, 3]), np.array([True, False, False]))
        ... # Prints Step 4: -> Avg Return: 4.00
        ... assert (recorder._episode_returns == np.array([0, 2, 6])).all()
        """
        self._step += 1
        if self._episode_returns is None:
            self._episode_returns = np.zeros_like(rewards)
        self._episode_returns += rewards
        if not dones.any():
            return
        episode_returns = np.extract(dones, self._episode_returns)
        np.putmask(self._episode_returns, dones, 0)
        avg_return = episode_returns.mean()
        self.ui.track_return(step=self._step, avg_return=avg_return)


@attrs.define()
class TransitionObserver(Observer[VanillaExperience]):
    """Observe the environment and store the experience in the memory.

    Args:
        env: The environment to observe.
        memory: The memory to store the experience in.
        policy: The policy to use to select actions.

    NOTE: You must not interact with the environment outside of this Observer.
    NOTE: The environment must be able to reset itself. This is not checked.
    NOTE: We call `env.reset()` at initialization.

    Example:
    --------

    >>> import numpy as np
    ... import torch
    ... from gymnasium.vector import make as vector_make
    ... from sbft.memories import ExperienceReplay
    ... from sbft.observers import TransitionObserver
    ... from sbft.policies import ContinuousRandomPolicy
    ... env = vector_make("Pendulum-v1", 3)
    ... batch_size, capacity = 2, 3
    ... memory = ExperienceReplay.from_env(batch_size, capacity, env)
    ... policy = ContinuousRandomPolicy.from_env(env)
    ... observer = TransitionObserver(env, memory, policy)
    ... params = tuple()
    ... generator = torch.Generator().manual_seed(0)
    ... rng = np.random.default_rng(0)
    ... for _ in range(3):
    ...     observer.observe(params, generator=generator)
    ... assert len(memory) == 3
    ... exp = memory.get(rng=rng)
    ... assert np.allclose(
    ...     exp.states,
    ...     torch.tensor(
    ...         [
    ...             [ 0.63649213,  0.7712832,   0.40598112],
    ...             [ 0.9906156,   0.13667752,  1.2565843 ],
    ...             [ 0.01806341, -0.99983686, -1.0979359 ],
    ...             [ 0.59459233,  0.8040273,   1.0636618 ],
    ...             [ 0.9771289,   0.21264784,  1.5435461 ],
    ...             [-0.05929149, -0.9982407,  -1.5478135 ],
    ...         ],
    ...         dtype=torch.float32,
    ...     ),
    ... ), exp.states
    """

    env: VectorEnv
    memory: Memory[VanillaExperience]
    policy: Policy
    ui: extu.CliUI = attrs.field(factory=lambda: extu.null_object(extu.CliUI))

    _states: npt.NDArray[t.Any] | None = attrs.field(init=False, default=None)
    _default_generator: torch.Generator = attrs.field(init=False)
    _episode_return_recorder: EpisodeReturnRecorder = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        self._states = None
        self._default_generator = torch.Generator()
        self._episode_return_recorder = EpisodeReturnRecorder(self.ui)

    def observe(
        self,
        params: tuple[torch.nn.Parameter, ...],
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        generator = generator or self._default_generator
        if self._states is None:
            self._states, _ = self.env.reset(  # type: ignore
                seed=generator.initial_seed()
            )
        actions = (
            self.policy(
                params, self._as_tensor_like(self._states, params), generator=generator
            )
            .detach()
            .cpu()
            .numpy()
        )
        self.env.step_async(actions)
        transitions = self.env.step_wait()
        next_states, rewards, terminals, truncateds, _ = transitions  # type: ignore
        experience = dict(
            states=self._states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            terminals=terminals,
            truncateds=truncateds,
        )
        self.memory.add(**experience)
        self._episode_return_recorder.track(rewards, terminals | truncateds)
        self._states = next_states

    def _as_tensor_like(
        self, states: npt.NDArray[t.Any], params: tuple[torch.nn.Parameter, ...]
    ) -> torch.Tensor:
        return torch.as_tensor(
            states, device=params[0].device if params else "cpu", dtype=torch.float32
        )
