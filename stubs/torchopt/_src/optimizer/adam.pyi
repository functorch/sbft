"""
This type stub file was generated by pyright.
"""

import torch
from typing import Iterable, Tuple
from torchopt._src.optimizer.base import Optimizer
from torchopt._src.typing import ScalarOrSchedule

class Adam(Optimizer):
    """The classic Adam optimizer.

    See Also:
        - The functional Adam optimizer: :func:`torchopt.adam`.
        - The differentiable meta-Adam optimizer: :class:`torchopt.MetaAdam`.
    """
    def __init__(self, params: Iterable[torch.Tensor], lr: ScalarOrSchedule, betas: Tuple[float, float] = ..., eps: float = ..., weight_decay: float = ..., *, eps_root: float = ..., maximize: bool = ..., use_accelerated_op: bool = ...) -> None:
        r"""The :meth:`init` function.

        Args:
            params: (iterable of torch.Tensor)
                An iterable of :class:`torch.Tensor`\s. Specifies what tensors should be optimized.
            lr: (default: :const:`1e-3`)
                This is a fixed global scaling factor.
            betas: (default: :const:`(0.9, 0.999)`)
                Coefficients used for computing running averages of gradient and its square.
            eps: (default: :const:`1e-8`)
                A small constant applied to denominator outside of the square root (as in the Adam
                paper) to avoid dividing by zero when rescaling.
            weight_decay: (default: :const:`0.0`)
                Weight decay, add L2 penalty to parameters.
            eps_root: (default: :data:`0.0`)
                A small constant applied to denominator inside the square root (as in RMSProp), to
                avoid dividing by zero when rescaling. This is needed for example when computing
                (meta-)gradients through Adam.
            maximize: (default: :data:`False`)
                Maximize the params based on the objective, instead of minimizing.
            use_accelerated_op: (default: :data:`False`)
                If :data:`True` use our implemented fused operator.
        """
        ...
    


