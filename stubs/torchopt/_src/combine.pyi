"""
This type stub file was generated by pyright.
"""

from torchopt._src import base

def chain(*args: base.GradientTransformation) -> base.GradientTransformation:
    """Applies a list of chainable update transformations.

    Given a sequence of chainable transforms, :func:`chain` returns an :func:`init_fn` that
    constructs a ``state`` by concatenating the states of the individual transforms, and returns an
    :func:`update_fn` which chains the update transformations feeding the appropriate state to each.

    Args:
        *args:
            A sequence of chainable ``(init_fn, update_fn)`` tuples.

    Returns:
        A single ``(init_fn, update_fn)`` tuple.
    """
    ...

