import typing as t

import attrs
import numpy as np
import numpy.typing as npt
from gymnasium.vector import VectorEnv

from sbft.api import InputTransform, Memory, OutputTransform
from sbft.experience import VanillaExperience
from sbft.memories.experience_replay.buffers import NumpyCircularBuffer
from sbft.memory_transforms import IdentityInputTransform, ToVanillaExperience


@attrs.define()
class ExperienceReplay(Memory[VanillaExperience]):
    """Experience replay memory.

    Used in Off-Policy algorithms like DQN, SAC etc.

    Args:
    -----
        batch_size: int, batch size.
        buffer: NumpyCircularBuffer, buffer to store the experiences.
        input_transform: InputTransform, function that transforms input to self.add().
        output_transform: OutputTransform, function that transforms output of self.get().  # noqa: E501

    Attributes:
    -----------
        batch_size: int, batch size.
        buffer: NumpyCircularBuffer, buffer to store the experiences.
        input_transform: InputTransform, function that transforms input to self.add().
        output_transform: OutputTransform, function that transforms output of self.get().  # noqa: E501
        rng: np.random.Generator, default random number generator when none is provided
            to self.get().

    Methods:
    --------
        add: add data to the buffer.
        get: get data from the buffer.
        flush: flush the buffer.
        __len__: get the length of the buffer.

    NOTE: If your VectorEnv has more than one environment, make sure to pass the number
    of envs to ToVanillaExperience.

    Examples:
    ---------
    >>> import numpy as np
    ... from gymnasium.vector import make as vector_make
    ... from sbft.memories.experience_replay import ExperienceReplay
    ... batch_size = 2
    ... n_envs = 2
    ... envs = vector_make("Pendulum-v1", num_envs=n_envs)
    ... memory = ExperienceReplay.from_env(
    ...     batch_size=batch_size,
    ...     capacity=3,
    ...     env=envs,
    ... )
    ... assert len(memory) == 0
    ... for _ in range(2):
    ...     states, _ = envs.reset()
    ...     actions = envs.action_space.sample()
    ...     next_states, rewards, terminals, truncateds, _ = envs.step(actions)
    ...     memory.add(
    ...         states=states,
    ...         actions=actions,
    ...         next_states=next_states,
    ...         rewards=rewards,
    ...         terminals=terminals,
    ...         truncateds=truncateds,
    ...     )
    ... assert len(memory) == 2
    ... experience = memory.get()
    ... assert experience.states.shape == (n_envs * batch_size, 3)
    ... assert experience.actions.shape == (n_envs * batch_size, 1)
    ... assert experience.next_states.shape == (n_envs * batch_size, 3)
    ... assert experience.rewards.shape == (n_envs * batch_size, 1)
    ... assert experience.terminals.shape == (n_envs * batch_size, 1)
    ... assert experience.truncateds.shape == (n_envs * batch_size, 1)
    """

    batch_size: int
    buffer: NumpyCircularBuffer

    input_transform: InputTransform = attrs.field(factory=IdentityInputTransform)
    output_transform: OutputTransform[VanillaExperience] = attrs.field(
        factory=lambda: ToVanillaExperience(1)
    )

    rng: np.random.Generator = attrs.field(init=False, repr=False)

    @classmethod
    def from_env(
        cls,
        batch_size: int,
        capacity: int,
        env: VectorEnv,
        input_transform: InputTransform | None = None,
        output_transform: OutputTransform[VanillaExperience] | None = None,
    ) -> "ExperienceReplay":
        buffer = NumpyCircularBuffer.from_env(env, capacity=capacity)
        return cls(
            batch_size=batch_size,
            buffer=buffer,
            input_transform=input_transform or IdentityInputTransform(),
            output_transform=output_transform or ToVanillaExperience(env.num_envs),
        )

    def __attrs_post_init__(self) -> None:
        self.rng = np.random.default_rng()

    @property
    def capacity(self) -> int:
        return self.buffer.capacity

    def add(self, **kwargs: npt.NDArray[t.Any]) -> None:
        # Ensure that the keys in kwargs are the same as those in VanillaExperience.
        kwargs = self.input_transform(**kwargs)
        if set(kwargs.keys()) != VanillaExperience.keys():
            raise ValueError(
                f"Invalid keys {set(kwargs.keys())}, expected {VanillaExperience.keys}"
            )
        self.buffer.add(kwargs)

    def get(self, rng: np.random.Generator | None = None) -> VanillaExperience:
        rng = rng or self.rng
        indices = rng.choice(len(self.buffer), size=self.batch_size, replace=False)
        return self.output_transform(**self.buffer[indices])

    def flush(self) -> None:
        return self.buffer.flush()

    def __len__(self):
        return len(self.buffer)
