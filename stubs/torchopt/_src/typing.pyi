"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable, Iterable, Mapping, TypeVar, Union
from torch import Tensor

Scalar = TypeVar('Scalar', float, int)
Numeric = Union[Tensor, Scalar]
Schedule = Callable[[Numeric], Numeric]
ScalarOrSchedule = Union[float, Schedule]
TensorTree = Union[Tensor, Iterable['TensorTree'], Mapping[Any, 'TensorTree']]
