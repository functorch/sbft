"""
This type stub file was generated by pyright.
"""

from torchopt._src import base

def apply_updates(params: base.Params, updates: base.Updates, *, inplace: bool = ...) -> base.Params:
    """Applies an update to the corresponding parameters.

    This is a utility functions that applies an update to a set of parameters, and then returns the
    updated parameters to the caller. As an example, the update may be a gradient transformed by a
    sequence of :class:`GradientTransformations`. This function is exposed for convenience, but it
    just adds updates and parameters; you may also apply updates to parameters manually, using
    :func:`tree_map` (e.g. if you want to manipulate updates in custom ways before applying them).

    Args:
        params: A tree of parameters.
        updates:
            A tree of updates, the tree structure and the shape of the leaf nodes must match that
            of ``params``.
        inplace: If :data:`True`, will update params in a inplace manner.

    Returns:
        Updated parameters, with same structure, shape and type as ``params``.
    """
    ...
