import typing as t

import attrs
import numpy.typing as npt
import torch

from sbft.api import Experience
from sbft.external_utils import get_torch_dtype


@attrs.define()
class TensorVanillaExperience:
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    truncateds: torch.Tensor
    dones: torch.Tensor


@attrs.define()
class VanillaExperience(Experience):
    states: npt.NDArray[t.Any]
    actions: npt.NDArray[t.Any]
    next_states: npt.NDArray[t.Any]
    rewards: npt.NDArray[t.Any]
    terminals: npt.NDArray[t.Any]
    truncateds: npt.NDArray[t.Any]

    @classmethod
    def keys(cls) -> set[str]:
        return {
            "states",
            "actions",
            "next_states",
            "rewards",
            "terminals",
            "truncateds",
        }

    @property
    def dones(self) -> npt.NDArray[t.Any]:
        return self.terminals | self.truncateds

    def as_tensors(self, device: torch.device) -> TensorVanillaExperience:
        def as_tensor(x: npt.NDArray[t.Any]) -> torch.Tensor:
            return torch.as_tensor(x, device=device, dtype=get_torch_dtype(x))

        return TensorVanillaExperience(
            states=as_tensor(self.states),
            actions=as_tensor(self.actions),
            next_states=as_tensor(self.next_states),
            rewards=as_tensor(self.rewards),
            terminals=as_tensor(self.terminals),
            truncateds=as_tensor(self.truncateds),
            dones=as_tensor(self.dones),
        )

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, key: str) -> npt.NDArray[t.Any]:
        if key == "states":
            return self.states
        if key == "actions":
            return self.actions
        if key == "next_states":
            return self.next_states
        if key == "rewards":
            return self.rewards
        if key == "terminals":
            return self.terminals
        if key == "truncateds":
            return self.truncateds
        raise KeyError(f"Invalid key {key}")
