"""
This type stub file was generated by pyright.
"""

import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Union

ARGS = Tuple[Any, ...]
KWARGS = Dict[Any, Any]
def custom_root(optimality_fun: Callable, argnums: Union[int, Tuple[int, ...]] = ..., has_aux: bool = ..., solve: Callable = ..., reference_signature: Optional[Union[inspect.Signature, Callable]] = ...) -> Callable[[Callable], Callable]:
    """Decorator for adding implicit differentiation to a root solver.

    Args:
        optimality_fun: (callable)
            An equation function, ``optimality_fun(params, *args)``. The invariant is
            ``optimality_fun(sol, *args) == 0`` at the solution / root of ``sol``.
        argnums: (int or tuple of int, default: :const:`0`)
            Specifies arguments to compute gradients with respect to.
        has_aux: (default: :data:`False`)
            Whether the decorated solver function returns auxiliary data.
        solve: (callable, optional, default: :func:`linear_solve.solve_normal_cg`)
            a linear solver of the form ``solve(matvec, b)``.
        reference_signature: (function signature, optional)
            Function signature (i.e. arguments and keyword arguments), with which the solver and
            optimality functions are expected to agree. Defaults to ``optimality_fun``. It is
            required that solver and optimality functions share the same input signature, but both
            might be defined in such a way that the signature correspondence is ambiguous (e.g. if
            both accept catch-all ``**kwargs``). To satisfy ``custom_root``'s requirement, any
            function with an unambiguous signature can be provided here.

    Returns:
        A solver function decorator, i.e., ``custom_root(optimality_fun)(solver_fun)``.
    """
    ...

