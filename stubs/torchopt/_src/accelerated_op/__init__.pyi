"""
This type stub file was generated by pyright.
"""

import torch
from typing import Iterable, Optional, Union
from torchopt._src.accelerated_op.adam_op import AdamOp

def accelerated_op_available(devices: Optional[Union[str, torch.device, Iterable[Union[str, torch.device]]]] = ...) -> bool:
    """Check the availability of accelerated optimizer."""
    ...

