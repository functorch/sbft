"""Various InputTransform and OutputTransform implementations."""

import typing as t

import attrs
import numpy.typing as npt

from sbft.api import InputTransform, OutputTransform
from sbft.experience import VanillaExperience


class IdentityInputTransform(InputTransform):
    """Identity input transform."""

    def __call__(self, **kwargs: npt.NDArray[t.Any]) -> dict[str, npt.NDArray[t.Any]]:
        return kwargs


@attrs.define()
class ToVanillaExperience(OutputTransform[VanillaExperience]):
    """Transforms a dict of arrays to an Experience."""

    n_envs: int

    def __call__(self, **kwargs: npt.NDArray[t.Any]) -> VanillaExperience:
        def as_batch_shape(key: str, value: npt.NDArray[t.Any]) -> npt.NDArray[t.Any]:
            """Change shape from (batch_size, n_envs, ...) to (batch_size, ...)."""
            if key in {"rewards", "terminals", "truncateds"}:
                return value.reshape(-1, 1)
            return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])

        return VanillaExperience(
            **{
                k: as_batch_shape(k, v)
                for k, v in kwargs.items()
                if k in VanillaExperience.keys()
            }
        )
