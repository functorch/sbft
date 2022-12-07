import typing as t

import numpy as np
import numpy.typing as npt
import torch


def soft_rescale_action(
    actions: torch.Tensor,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
) -> torch.Tensor:
    """Rescale the actions to the correct range."""
    return torch.tanh(actions) * action_scale + action_bias


def polyak_average(
    params: tuple[torch.nn.Parameter, ...],
    target_params: tuple[torch.nn.Parameter, ...],
    polyak_factor: float,
) -> tuple[torch.nn.Parameter, ...]:
    """Return updated target parameters using Polyak averaging."""
    assert len(params) == len(target_params)
    return tuple(
        t.cast(
            torch.nn.Parameter,
            param * polyak_factor + target_param * (1 - polyak_factor),
        )
        for param, target_param in zip(params, target_params)
    )


def get_torch_dtype(x: npt.NDArray[t.Any]) -> torch.dtype:
    """Return the torch dtype corresponding to the numpy dtype of x."""
    if np.issubdtype(x.dtype, np.bool_):
        return torch.bool
    if np.issubdtype(x.dtype, np.floating):
        return torch.float32
    if np.issubdtype(x.dtype, np.integer):
        return torch.int32
    raise ValueError(f"Unsupported dtype {x.dtype}")
