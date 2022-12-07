import typing as t

import torch

class FunctionalModule(torch.nn.Module):
    """Stub for FunctionalModule."""

    def __call__(
        self,
        parameters: tuple[torch.nn.Parameter, ...],
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any: ...

class FunctionalModuleWithBuffers(torch.nn.Module):
    """Stub for FunctionalModuleWithBuffers."""

    def __call__(
        self,
        parameters: tuple[torch.nn.Parameter, ...],
        buffers: tuple[torch.nn.Parameter, ...],
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any: ...

def make_functional(
    module: torch.nn.Module, disable_autograd_tracking: bool = ...
) -> tuple[FunctionalModule, tuple[torch.nn.Parameter, ...]]: ...
def make_functional_with_buffers(
    module: torch.nn.Module, disable_autograd_tracking: bool = ...
) -> tuple[
    FunctionalModuleWithBuffers,
    tuple[torch.nn.Parameter, ...],
    tuple[torch.nn.Parameter, ...],
]: ...
def grad(
    func: t.Callable[..., t.Any],
    argnums: int | tuple[int, ...] = ...,
    has_aux: bool = ...,
) -> t.Callable[..., t.Any]: ...
def grad_and_value(
    func: t.Callable[..., t.Any],
    argnums: int | tuple[int, ...] = ...,
    has_aux: bool = ...,
) -> t.Callable[..., t.Any]: ...
