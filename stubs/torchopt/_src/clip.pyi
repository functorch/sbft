"""
This type stub file was generated by pyright.
"""

from torchopt._src import base

ClipState = base.EmptyState
def clip_grad_norm(max_norm: float, norm_type: float = ..., error_if_nonfinite: bool = ...) -> base.GradientTransformation:
    """Clips gradient norm of an iterable of parameters.

    Args:
        max_delta: The maximum absolute value for each element in the update.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """
    ...

