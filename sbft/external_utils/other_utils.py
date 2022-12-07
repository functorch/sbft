import typing as t
from unittest.mock import MagicMock

import torch
import typing_extensions as tx


class _SupportsRescaling(t.Protocol):
    def __add__(self: tx.Self, other: float | tx.Self) -> tx.Self:
        raise NotImplementedError

    def __radd__(self: tx.Self, other: float | tx.Self) -> tx.Self:
        raise NotImplementedError

    def __sub__(self: tx.Self, other: float | tx.Self) -> tx.Self:
        raise NotImplementedError

    def __rsub__(self: tx.Self, other: float | tx.Self) -> tx.Self:
        raise NotImplementedError

    def __mul__(self: tx.Self, other: float | tx.Self) -> tx.Self:
        raise NotImplementedError

    def __rmul__(self: tx.Self, other: float | tx.Self) -> tx.Self:
        raise NotImplementedError

    def __truediv__(self: tx.Self, other: float | tx.Self) -> tx.Self:
        raise NotImplementedError

    def __rtruediv__(self: tx.Self, other: float | tx.Self) -> tx.Self:
        raise NotImplementedError


T = t.TypeVar("T", bound=_SupportsRescaling)


def allclose(
    x: t.Any,
    y: t.Any,
    atol: float = 1e-8,
    rtol: float = 0.00001,
) -> bool:
    """Check if two values are close.

    Args:
    -----
        x: The first value.
        y: The second value.
        atol: The absolute tolerance.
        rtol: The relative tolerance.

    Returns:
    --------
        True if the values are close, False otherwise.

    Examples:
    ---------
    >>> import numpy as np
    ... import torch
    >>> allclose(1.0, 1.0)
    True
    >>> allclose(1.0, 1.0000001)
    True
    >>> allclose(torch.tensor(1.0), np.array(1.0))
    True
    >>> allclose(torch.tensor(1.0), 0)
    False
    >>> allclose(np.array(1.0), 0)
    False
    """
    x_ = torch.as_tensor(x, dtype=torch.float64)
    y_ = torch.as_tensor(y, dtype=torch.float64)
    return torch.allclose(x_, y_, atol=atol, rtol=rtol)


def rescale(
    x: T,
    old_min: float | T,
    old_max: float | T,
    new_min: float | T,
    new_max: float | T,
) -> T:
    """Rescale a value from one range to another.

    Args:
    -----
        x: The value to rescale.
        old_min: The minimum value of the old range.
        old_max: The maximum value of the old range.
        new_min: The minimum value of the new range.
        new_max: The maximum value of the new range.

    Returns:
    --------
        The rescaled value.

    Examples:
    ---------
    >>> import pytest
    >>> rescale(0.5, 0, 1, 0, 10)
    5.0
    >>> rescale(0.5, 0, 1, 10, 0)
    5.0
    >>> with pytest.raises(ValueError):
    ...     rescale(0.5, 0, 0, 0, 10)
    >>> with pytest.raises(ValueError):
    ...     rescale(0.5, 0, 1, 0, 0)
    """
    if allclose(old_min, old_max):
        raise ValueError("old_max must be different from old_min")
    if allclose(new_min, new_max):
        raise ValueError("new_max must be different from new_min")
    old_range = old_max - old_min
    new_range = new_max - new_min
    return t.cast(T, (((x - old_min) * new_range) / old_range) + new_min)


def rescale_range(
    x: T,
    old_range: float | T,
    new_range: float | T,
) -> T:
    """Rescale a value from one range to another.

    Args:
    -----
        x: The value to rescale.
        old_range: The old range.
        new_range: The new range.

    Returns:
    --------
        The rescaled value.

    Examples:
    ---------
    >>> import pytest
    >>> rescale_range(0.5, 1, 10)
    5.0
    >>> rescale_range(0.5, 1, -10)
    -5.0
    >>> with pytest.raises(ValueError):
    ...     rescale_range(0.5, 0, 10)
    >>> with pytest.raises(ValueError):
    ...     rescale_range(0.5, 1, 0)
    """
    if allclose(old_range, 0):
        raise ValueError("old_range must be different from 0")
    if allclose(new_range, 0):
        raise ValueError("new_range must be different from 0")
    return t.cast(T, (x * new_range) / old_range)


AnyT = t.TypeVar("AnyT")


def null_object(cls: t.Type[AnyT]) -> AnyT:
    """Create a null object following the Null object pattern.

    Args:
    -----
        cls: The class of the object for which a null object will be created.

    Returns:
    --------
        A dummy object having the same spec as `cls` that does nothing for any method
        call.

    Examples:
    ---------

    >>> import pytest
    ... class Foo:
    ...     def __init__(self, x: int):
    ...         self.x = x
    ...     def bar(self, y: list[int]) -> None:
    ...         y.append(self.x)
    ... actual_foo = Foo(1)
    ... l = []
    ... actual_foo.bar(l)
    ... assert l == [1]
    ... null_foo = null_object(Foo)
    ... new_l = []
    ... null_foo.bar(new_l)
    ... assert new_l == []
    ... with pytest.raises(AttributeError):
    ...     null_foo.x
    ... with pytest.raises(AttributeError):
    ...     null_foo.a_method_that_does_not_exist()

    Usage:
    ------
    Instead of the following pattern:
    >>> foo: Foo | None = None
    ... if foo is not None:
    ...     foo.bar([])

    Do:
    >>> foo = null_object(Foo)
    ... foo.bar([])  # This will do nothing
    """
    return t.cast(AnyT, MagicMock(spec=cls))
