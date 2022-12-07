"""
This type stub file was generated by pyright.
"""

from abc import abstractmethod
from typing import Callable, NamedTuple, Optional, Tuple
from torchopt._src.typing import Numeric, TensorTree

OptState = TensorTree
Params = TensorTree
Updates = Params
Schedule = Callable[[Numeric], Numeric]
class EmptyState(NamedTuple):
    """An empty state for the simplest stateless transformations."""
    ...


class TransformInitFn(Protocol):
    """A callable type for the :func:`init` step of a :class:`GradientTransformation`.

    The :func:`init` step takes a tree of ``params`` and uses these to construct an arbitrary
    structured initial ``state`` for the gradient transformation. This may hold statistics of the
    past updates or any other non static information.
    """
    @abstractmethod
    def __call__(self, params: Params) -> OptState:
        """The `init` function.

        Args:
            params:
                The initial value of the parameters.

        Returns:
            The initial state of the gradient transformation.
        """
        ...
    


class TransformUpdateFn(Protocol):
    """A callable type for the :func:`update` step of a :class:`GradientTransformation`.

    The :func:`update` step takes a tree of candidate parameter ``updates`` (e.g. their gradient
    with respect to some loss), an arbitrary structured ``state``, and the current ``params`` of the
    model being optimized. The ``params`` argument is optional, it must however be provided when
    using transformations that require access to the current values of the parameters. The
    ``inplace`` argument is optional, If :data:`True`, modify updates and state using inplace
    operations.
    """
    @abstractmethod
    def __call__(self, updates: Updates, state: OptState, *, params: Optional[Params] = ..., inplace: bool = ...) -> Tuple[Updates, OptState]:
        """The `update` function.

        Args:
            updates: A tree of candidate updates.
            state: The state of the gradient transformation.
            params: (optional)
                The current value of the parameters.
            inplace: (optional)
                If :data:`True`, modify updates and state using inplace operations.

        Returns:
            The transformed ``updates``, and the updated ``state``.
        """
        ...
    


class GradientTransformation(NamedTuple):
    """A pair of pure functions implementing a gradient transformation.

    TorchOpt optimizers are all implemented as *gradient transformations* like Optax. A gradient
    transformation is defined to be a pair of pure functions, which are combined together in a
    :class:`NamedTuple` so that they can be referred to by name.

    Since gradient transformations do not contain any internal state, all stateful optimizer
    properties (such as the current step count when using optimizer schedules, or momentum values)
    are passed through gradient transformations by using the optimizer *state* ``pytree``. Each time
    a gradient transformation is applied, the state is computed and returned, ready to be passed to
    the next call to the gradient transformation.

    Attributes:
        init:
            A pure function which, when called with an example instance of the parameters whose
            gradients will be transformed, returns a ``pytree`` containing the initial value for the
            optimizer state.
        update:
            A pure function which takes as input a pytree of updates (with the same tree structure
            as the original params ``pytree`` passed to :attr:`init`), the previous optimizer state
            (which may have been initialized using the :attr:`init` function), and optionally the
            ``inplace`` flag. The :attr:`update` function then returns the computed gradient
            updates, and a updates optimizer state. If the ``inplace`` flag is :data:`True`, the
            output results are the same instance as the input.
    """
    init: TransformInitFn
    update: TransformUpdateFn
    def chain(self, next: GradientTransformation) -> ChainedGradientTransformation:
        """Chain two gradient transformations together."""
        ...
    


class ChainedGradientTransformation(GradientTransformation):
    """A chain of gradient transformations.

    This class is a subclass of :class:`GradientTransformation` which allows for chaining of
    gradient transformations.
    """
    transformations: Tuple[GradientTransformation, ...]
    def __new__(cls, *transformations: GradientTransformation) -> ChainedGradientTransformation:
        """Creates a new chained gradient transformation."""
        ...
    
    def __str__(self) -> str:
        """Returns a string representation of the chained gradient transformation."""
        ...
    
    __repr__ = ...
    def __eq__(self, other: object) -> bool:
        """Returns whether two chained gradient transformations are equal."""
        ...
    
    def __hash__(self) -> int:
        """Returns the hash of the chained gradient transformation."""
        ...
    
    def __getstate__(self) -> Tuple[GradientTransformation, ...]:
        """Returns the state of the chained gradient transformation for serialization."""
        ...
    
    def __setstate__(self, state: Tuple[GradientTransformation, ...]) -> None:
        """Sets the state of the chained gradient transformation from serialization."""
        ...
    
    def __reduce__(self) -> Tuple[Callable, Tuple[Tuple[GradientTransformation, ...]]]:
        """Serialization support for chained gradient transformation."""
        ...
    


class IdentityGradientTransformation(GradientTransformation):
    """A gradient transformation that does nothing."""
    def __new__(cls): # -> Self@IdentityGradientTransformation:
        """Create a new gradient transformation that does nothing."""
        ...
    
    @staticmethod
    def init_fn(params: Params) -> OptState:
        """Returns empty state."""
        ...
    
    @staticmethod
    def update_fn(updates: Updates, state: OptState, *, params: Optional[Params] = ..., inplace: bool = ...) -> Tuple[Updates, OptState]:
        """Returns updates unchanged."""
        ...
    


def identity() -> IdentityGradientTransformation:
    """Stateless identity transformation that leaves input gradients untouched.

    This function passes through the *gradient updates* unchanged.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """
    ...

