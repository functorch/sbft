"""Initializers for PyTorch similar to torch.nn.init, but for use with functorch.

All the initializers in this module take in a `generator` argument, which expects a
torch.Generator object. This is necessary because the initializers in this module
are used in conjunction with functorch, which requires that modules be stateless.

The typical use case is to create a generator object, seed it, and then pass it to
the initializers to generate the initial parameters for a module. For example:

```python
import torch
import functorch as ft

# Create a generator and seed it.
generator = torch.Generator(device='cpu')
generator.manual_seed(0)

# Create a functorch extracted module.
module = torch.nn.Sequential(...)
module_func, params, buffers = ft.make_functional_with_buffers(module)

# Create params for the module like this:
# TODO: Document the behaviour of how the generator's device affects the device of
# the generated parameters.
new_params = init_like_params(params, generator, uniform_like, a=-0.1, b=0.1)
x = torch.randn(3, 5)
output = module_func(new_params, buffers, x)
```
"""
import math
import typing as t

import numpy as np
import torch

P = t.ParamSpec("P")
TensorT = t.TypeVar("TensorT", torch.Tensor, torch.nn.Parameter)


class FunctionalInitializer(t.Protocol[P, TensorT]):
    def __call__(
        self,
        tensor: TensorT,
        generator: torch.Generator,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> TensorT:
        raise NotImplementedError()


def init_like_params(
    params: tuple[TensorT, ...],
    generator: torch.Generator,
    initializer: FunctionalInitializer[P, TensorT],
    *args: P.args,
    **kwargs: P.kwargs,
) -> t.Iterator[TensorT]:
    """Initializes a sequence of parameters like the parameters in `params`.

    Args:
        params: a sequence of tensors defining the shape of the output tensors
        generator: a `torch.Generator` object
        initializer: a function that takes in a tensor and returns a tensor of the
            same shape and dtype as the input tensor
        **kwargs: additional keyword arguments to pass to `initializer`

    Examples:
        >>> generator = torch.Generator(device='cpu')
        ... generator.manual_seed(0)
        ... params = [torch.empty(3, 5).to(device='cpu', dtype=torch.float16)]
        ... res = init_like_params(params, generator, uniform_like, a=-0.1, b=0.1)
        ... assert all(r.shape == p.shape for r, p in zip(res, params))
        ... assert all(r.dtype == p.dtype for r, p in zip(res, params))
        ... assert all(r.device == p.device for r, p in zip(res, params))
    """
    for param in params:
        yield initializer(tensor=param, generator=generator, *args, **kwargs)


def zeros_init_like(
    tensor: TensorT,
    generator: torch.Generator,
) -> TensorT:
    result = torch.zeros_like(tensor)
    if isinstance(tensor, torch.Tensor):
        return result
    return torch.nn.Parameter(result)


def constant_init_like(
    tensor: TensorT,
    generator: torch.Generator,
    value: float,
) -> TensorT:
    result = torch.full_like(tensor, value)
    if isinstance(tensor, torch.Tensor):
        return result
    return torch.nn.Parameter(result)


def init_feed_forward_like(
    weights_init: FunctionalInitializer[[], TensorT],
    bias_init: FunctionalInitializer[[], TensorT] = zeros_init_like,
) -> FunctionalInitializer[[], TensorT]:
    """Initialize either the weights or biases of a feed-forward layer.

    Args:
    -----
        weights_init: a function that takes in a tensor and returns a tensor of the
            same shape and dtype as the input tensor
        bias_init: a function that takes in a tensor and returns a tensor of the
            same shape and dtype as the input tensor
    Returns:
    --------
        A function that can initialize either the weights or the bias of a feed-forward
        layer.
        Signature of the returned function:
            (tensor: TensorT, generator: torch.Generator) -> TensorT
        where:
            tensor: a tensor defining the shape of the output tensor
            generator: a `torch.Generator` object for seeding the random number
                generator

    Examples:
    ---------

    >>> import torch
    ... import functorch as ft
    ... from functools import partial
    ... generator = torch.Generator(device='cpu').manual_seed(0)
    ... linear = torch.nn.Linear(3, 5)
    ... linear_func, params = ft.make_functional(linear)
    ... weights, bias = params
    ... initializer = init_feed_forward_like(
    ...     partial(constant_init_like, value=2)
    ... )
    ... new_weights, new_bias = init_like_params(params, generator, initializer)
    ... assert torch.allclose(new_weights, torch.full_like(weights, 2))
    ... assert torch.allclose(new_bias, torch.zeros_like(bias))
    """

    def init(tensor: TensorT, generator: torch.Generator) -> TensorT:
        if tensor.ndim == 2:
            return weights_init(tensor=tensor, generator=generator)
        if tensor.ndim == 1:
            return bias_init(tensor=tensor, generator=generator)
        raise ValueError(
            f"Expected tensor to have 1 or 2 dimensions, got {tensor.ndim}"
        )

    return init


def uniform(
    generator: torch.Generator,
    size: tuple[int, ...],
    a: float,
    b: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Returns a random floating point number from the uniform distribution
    :math:`\mathcal{U}(a, b)` on the interval :math:`[a, b)`

    Args:
        generator: a `torch.Generator` object
        size: a tuple defining the shape of the output tensor
        a: the lower bound of the range to sample from
        b: the upper bound of the range to sample from
        dtype: the desired data type of returned tensor
        device: the desired device of returned tensor

    Examples:
        >>> generator = torch.Generator(device='cpu')
        ... generator.manual_seed(0)
        ... size = (3, 5)
        ... dtype = torch.float16
        ... device = 'cpu'
        ... tensor = torch.empty(size, dtype=dtype, device=device)
        ... res = uniform_like(tensor, generator, a=-0.1, b=0.1)
        ... assert res.shape == tensor.shape
        ... assert res.dtype == tensor.dtype
        ... assert res.device == tensor.device
        ... assert res.min() >= -0.1, res.min()
        ... assert res.max() <= 0.1, res.max()
    """
    if a >= b:
        raise ValueError(f"uniform_ expects a < b, but got a={a} and b={b}")
    return (b - a) * torch.rand(*size, generator=generator, dtype=dtype) + a


def uniform_like(
    tensor: torch.Tensor,
    generator: torch.Generator,
    a: float,
    b: float,
) -> torch.Tensor:
    r"""Returns a tensor with the same characteristics as :attr:`tensor` filled with
    numbers from the uniform distribution :math:`\mathcal{U}(a, b)` on the interval
    :math:`[a, b)`.

    Args:
        tensor: a tensor defining the shape of the output tensor
        generator: a `torch.Generator` object
        a: the lower bound of the range to sample from
        b: the upper bound of the range to sample from

    Examples:
        >>> generator = torch.Generator(device='cpu')
        ... generator.manual_seed(0)
        ... size = (3, 5)
        ... dtype = torch.float16
        ... device = 'cpu'
        ... tensor = torch.empty(size, dtype=dtype, device=device)
        ... res = uniform_like(tensor, generator, a=-0.1, b=0.1)
        ... assert res.shape == tensor.shape
        ... assert res.dtype == tensor.dtype
        ... assert res.device == tensor.device
        ... assert res.min() >= -0.1, res.min()
        ... assert res.max() <= 0.1
    """
    if generator.device != tensor.device:
        # Raising RuntimeError because PyTorch does the same thing if you pass a gpu
        # generator and ask for a cpu tensor or vice-versa.
        raise RuntimeError(
            f"Generator device {generator.device} must match tensor device"
            + f" {tensor.device}"
        )
    return uniform(
        generator=generator,
        size=tensor.shape,
        a=a,
        b=b,
        dtype=tensor.dtype,
    )


# These no_grad_* functions are necessary as wrappers around the parts of these
# functions that use `with torch.no_grad()`. The JIT doesn't support context
# managers, so these need to be implemented as builtins. Using these wrappers
# lets us keep those builtins small and re-usable.
def _func_no_grad_uniform(
    generator: torch.Generator,
    size: tuple[int, ...],
    a: float,
    b: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Functional version of torch.nn.init._no_grad_uniform_."""
    with torch.no_grad():
        return uniform(generator=generator, size=size, a=a, b=b, dtype=dtype)


@t.overload
def _func_no_grad_normal(
    mean: float,
    std: float,
    generator: torch.Generator,
    size: tuple[int, ...] = (),
) -> torch.Tensor:
    ...


@t.overload
def _func_no_grad_normal(
    mean: torch.Tensor,
    std: torch.Tensor,
    generator: torch.Generator,
    size: tuple[int, ...] = (),
) -> torch.Tensor:
    ...


def _func_no_grad_normal(
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
    generator: torch.Generator,
    size: tuple[int, ...] = (),
) -> torch.Tensor:
    with torch.no_grad():
        if not size:
            assert isinstance(mean, torch.Tensor)
            assert isinstance(std, torch.Tensor)
            return torch.normal(mean=mean, std=std, generator=generator)
        else:
            assert isinstance(mean, float)
            assert isinstance(std, float)
            return torch.normal(mean=mean, std=std, size=size, generator=generator)


def _func_calculate_fan_in_and_fan_out(
    dimensions: int, shape: tuple[int, ...]
) -> tuple[int, int]:
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2"
            + " dimensions"
        )

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform(
    shape: tuple[int, ...],
    generator: torch.Generator,
    gain: float = 1.0,
) -> torch.Tensor:
    r"""Returns a new `Tensor` with given shape with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    This is the functional equivalent to calling
    `torch.nn.init.xavier_uniform_` with
    - shape=tensor.shape

    Args:
        shape: the shape of the resulting tensor
        generator: Generator used for random number generation
        gain: an optional scaling factor

    Examples:
        >>> import torch.nn as nn
        ... generator = torch.Generator(device='cpu')
        ... generator.manual_seed(0)
        ... res = xavier_uniform(
        ...     shape=(3, 5), generator=generator, gain=nn.init.calculate_gain('relu')
        ... )
        ... assert res.shape == (3, 5)
    """
    dimensions = len(shape)
    fan_in, fan_out = _func_calculate_fan_in_and_fan_out(
        dimensions=dimensions, shape=shape
    )
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _func_no_grad_uniform(generator=generator, size=shape, a=-a, b=a)


def xavier_uniform_like(
    tensor: torch.Tensor,
    generator: torch.Generator,
    gain: float = 1.0,
) -> torch.Tensor:
    r"""Returns a `Tensor` with same shape as with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        generator: Generator used for random number generation
        gain: an optional scaling factor

    Examples:
        >>> import torch.nn as nn
        ... generator = torch.Generator(device='cpu')
        ... w = torch.empty(3, 5)
        ... res = xavier_uniform_like(
        ...     w, gain=nn.init.calculate_gain('relu'), generator=generator
        ... )
        ... assert res.shape == (3, 5)

    """
    return xavier_uniform(shape=tensor.shape, generator=generator, gain=gain)


def xavier_normal(
    shape: tuple[int, ...],
    generator: torch.Generator,
    gain: float = 1.0,
) -> torch.Tensor:
    r"""Returns a new `Tensor` with given shape with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}  # noqa: E501

    Also known as Glorot initialization.

    Args:
        shape: shape of output tensor
        generator: Generator used for random number generation
        gain: an optional scaling factor

    Examples:
        >>> import torch.nn as nn
        ... generator = torch.Generator(device='cpu')
        ... generator.manual_seed(0)
        ... res = xavier_normal(
        ...     (3, 5), generator=generator, gain=nn.init.calculate_gain('relu')
        ... )
    """
    fan_in, fan_out = _func_calculate_fan_in_and_fan_out(
        dimensions=len(shape), shape=shape
    )
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _func_no_grad_normal(size=shape, generator=generator, mean=0.0, std=std)


def xavier_normal_like(
    tensor: torch.Tensor, generator: torch.Generator, gain: float = 1.0
) -> torch.Tensor:
    r"""Returns a new `Tensor` with shape = tensor.shape with values according to the
    method described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}  # noqa: E501

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        generator: Generator used for random number generation
        gain: an optional scaling factor

    Examples:
        >>> from torch import nn
        ... generator = torch.Generator(device='cpu')
        ... w = torch.empty(3, 5)
        ... res = xavier_normal_like(
        ...     w, gain=nn.init.calculate_gain('relu'), generator=generator
        ... )
        ... assert res.shape == (3, 5)
    """
    return xavier_normal(shape=tensor.shape, generator=generator, gain=gain)


def orthogonal(
    shape: tuple[int, ...],
    generator: torch.Generator,
    gain: float = 1,
):
    r"""Returns a new `Tensor` with given shape with a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        shape: Shape of output tensor
        generator: Generator used for random number generation
        gain: optional scaling factor

    Examples:
        >>> generator = torch.Generator(device='cpu')
        ... w = torch.empty(3, 5)
        ... res = orthogonal(w.shape, generator=generator)
    """
    ndimensions = len(shape)
    numel = int(np.prod(shape))
    if ndimensions < 2:
        print(shape)
        raise ValueError(
            f"Only tensors with 2 or more dimensions are supported, not {ndimensions}"
        )

    if numel == 0:
        # no-op
        return torch.as_tensor([])
    rows = shape[0]
    cols = numel // rows
    flattened = _func_no_grad_normal(
        mean=0.0, std=1.0, size=(rows, cols), generator=generator
    )

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = t.cast(tuple[torch.Tensor, torch.Tensor], torch.linalg.qr(flattened))
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()
    with torch.no_grad():
        result = q.view(shape).mul_(gain)
    return result


def orthogonal_like(
    tensor: torch.Tensor, generator: torch.Generator, gain: float = 1
) -> torch.Tensor:
    r"""Returns a new `Tensor` with given shape with a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        shape: Shape of output tensor
        generator: Generator used for random number generation
        gain: optional scaling factor

    Examples:
        >>> generator = torch.Generator(device='cpu')
        ... w = torch.empty(3, 5)
        ... res = orthogonal_like(w, generator=generator)
    """
    return orthogonal(shape=tensor.shape, generator=generator, gain=gain)
