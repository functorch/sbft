"""
This type stub file was generated by pyright.
"""

from typing import Callable, Optional
from torchopt._src.typing import TensorTree

def tree_add(tree_x: TensorTree, tree_y: TensorTree, alpha: float = ...) -> TensorTree:
    """Computes tree_x + alpha * tree_y."""
    ...

def solve_cg(matvec: Callable[[TensorTree], TensorTree], b: TensorTree, ridge: Optional[float] = ..., init: Optional[TensorTree] = ..., **kwargs) -> TensorTree:
    """Solves ``A x = b`` using conjugate gradient.

    It assumes that ``A`` is a Hermitian, positive definite matrix.

    Args:
        matvec: a function that returns the product between ``A`` and a vector.
        b: a tree of tensors.
        ridge: optional ridge regularization.
        init: optional initialization to be used by conjugate gradient.
        **kwargs: additional keyword arguments for solver.

    Returns:
        The solution with the same structure as ``b``.
    """
    ...

def solve_normal_cg(**kwargs): # -> partial[TensorTree]:
    """Wrapper for `solve_normal_cg`."""
    ...
