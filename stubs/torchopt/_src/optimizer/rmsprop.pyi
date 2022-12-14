"""
This type stub file was generated by pyright.
"""

import torch
from typing import Iterable
from torchopt._src.optimizer.base import Optimizer
from torchopt._src.typing import ScalarOrSchedule

class RMSProp(Optimizer):
    """The classic RMSProp optimizer.

    See Also:
        - The functional RMSProp optimizer: :func:`torchopt.rmsprop`.
        - The differentiable meta-RMSProp optimizer: :class:`torchopt.MetaRMSProp`.
    """
    def __init__(self, params: Iterable[torch.Tensor], lr: ScalarOrSchedule = ..., alpha: float = ..., eps: float = ..., weight_decay: float = ..., momentum: float = ..., centered: bool = ..., *, initial_scale: float = ..., nesterov: bool = ..., maximize: bool = ...) -> None:
        r"""The `init` function.

        Args:
            params: (iterable of torch.Tensor)
                An iterable of :class:`torch.Tensor`\s. Specifies what Tensors should be optimized.
            lr: (default: :const:`1e-2`)
                This is a fixed global scaling factor.
            alpha: (default: :const:`0.99`)
                Smoothing constant, the decay used to track the magnitude of previous gradients.
            eps: (default: :const:`1e-8`)
                A small numerical constant to avoid dividing by zero when rescaling.
            weight_decay: (default: :const:`0.0`)
                Weight decay, add L2 penalty to parameters.
            momentum: (default: :const:`0.0`)
                The decay rate used by the momentum term. The momentum is not used when it is set to
                :const:`0.0`.
            centered: (default: :data:`False`)
                If :data:`True`, use the variance of the past gradients to rescale the latest
                gradients.
            initial_scale: (default: :data:`0.0`)
                Initialization of accumulators tracking the magnitude of previous updates. PyTorch
                uses :data:`0.0`, TensorFlow 1.x uses :data:`1.0`. When reproducing results from a
                paper, verify the value used by the authors.
            nesterov: (default: :data:`False`)
                Whether to use Nesterov momentum.
            maximize: (default: :data:`False`)
                Maximize the params based on the objective, instead of minimizing.
        """
        ...
    


RMSprop = RMSProp
