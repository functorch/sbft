"""API followed by all replay buffers."""
import abc
import typing as t

import numpy as np
import numpy.typing as npt


class Experience(abc.ABC):
    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def keys(cls) -> set[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, key: str) -> npt.NDArray[t.Any]:
        raise NotImplementedError

    def __getattr__(self, name: str) -> npt.NDArray[t.Any]:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"{self.__class__} has no attribute {name}") from None


ExperienceT = t.TypeVar("ExperienceT", bound=Experience)
ExperienceT_co = t.TypeVar("ExperienceT_co", bound=Experience, covariant=True)
OutputTransformT = t.TypeVar("OutputTransformT", covariant=True)


class InputTransform(t.Protocol):
    """A protocol for transforming the input to a replay buffer.

    This protocol is used to transform the input to a replay buffer. Typically, the
    input is obtained from calling the environment step function. See the Memory
    subclasses for examples of how this protocol is used.
    """

    def __call__(self, **kwargs: npt.NDArray[t.Any]) -> dict[str, npt.NDArray[t.Any]]:
        raise NotImplementedError


class OutputTransform(t.Protocol[ExperienceT_co]):
    """A protocol for transforming the output of a replay buffer.

    The input is the data (or a subset of it) stored in the replay buffer. The output
    is an instance of Experience.  The transform is used to convert the data stored in
    the replay buffer to an instance of Experience.

    """

    def __call__(self, **kwargs: npt.NDArray[t.Any]) -> ExperienceT_co:
        raise NotImplementedError


class Memory(abc.ABC, t.Generic[ExperienceT]):
    input_transform: InputTransform
    output_transform: OutputTransform[ExperienceT]

    @abc.abstractmethod
    def add(self, **kwargs: npt.NDArray[t.Any]) -> None:
        """Add an experience to the replay buffer.

        Args:
            **kwargs: The experience to add to the replay buffer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, rng: np.random.Generator | None = None) -> ExperienceT:
        """Get a batch of experiences from the replay buffer.

        Args:
            rng: The random number generator to use.

        Returns:
            A batch of experiences.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def flush(self) -> None:
        """Flush the replay buffer."""
        raise NotImplementedError
