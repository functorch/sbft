import itertools as it
import math
import typing as t
import warnings

import more_itertools as mit
import torch


class _FeedForwardModuleConstructor(t.Protocol):
    def __call__(
        self, input_features: int, output_features: int, *args: t.Any, **kwargs: t.Any
    ) -> torch.nn.Module:
        raise NotImplementedError()


class _ActivationLayerConstructor(t.Protocol):
    def __call__(self, *args: t.Any, **kwargs: t.Any) -> torch.nn.Module:
        raise NotImplementedError()


def as_feed_forward_module(
    module: str | _FeedForwardModuleConstructor,
    n_inputs: int,
    n_outputs: int,
    *args: t.Any,
    **kwargs: t.Any,
) -> torch.nn.Module:
    if isinstance(module, str):
        module = getattr(torch.nn, module)
    assert not isinstance(module, str)
    return module(n_inputs, n_outputs, *args, **kwargs)


def as_activation_module(
    module: str | _ActivationLayerConstructor,
    inplace: bool,
    *args: t.Any,
    **kwargs: t.Any,
) -> torch.nn.Module:
    if isinstance(module, str):
        module = getattr(torch.nn, module)
    assert not isinstance(module, str)
    try:
        return module(inplace=inplace, *args, **kwargs)
    except TypeError:
        return module(*args, **kwargs)


def simple_feed_forward(
    n_inputs: int = 1,
    hidden_layer_modules: str
    | _FeedForwardModuleConstructor
    | list[str | _FeedForwardModuleConstructor] = "Linear",
    hidden_layer_sizes: tuple[int, ...] = (256, 256),
    hidden_activations: str
    | _ActivationLayerConstructor
    | list[str | _ActivationLayerConstructor] = "ReLU",
    output_layer_module: str | _FeedForwardModuleConstructor = "Linear",
    n_outputs: int = 1,
    output_activation: str | _ActivationLayerConstructor = "Identity",
    dropout_rates: float | list[float] | tuple[float] = 0.0,
    layer_norms: bool | list[bool] | tuple[bool] = False,
    inplace_activation: bool = True,
) -> torch.nn.Module:
    """Construct a feed forward network.

    Args:
    -----
    n_inputs: int
        The number of inputs to the network.
    hidden_layer_modules: str | _FeedForwardModuleConstructor | list[str | _FeedForwardModuleConstructor]  # noqa: E501
        The module to use for the hidden layers. If a string, it must be the name of a
        module in torch.nn. If a list, and not the same length as hidden_layer_sizes,
        the values will be cycled through.
    hidden_layer_sizes: tuple[int, ...]
        The number of nodes in each hidden layer. Will return a network with number of
        layers equal to the length of this tuple + 1 (excluding activation, dropout and
        normalization layers).
    hidden_activations: str | _ActivationLayerConstructor | list[str | _ActivationLayerConstructor]  # noqa: E501
        The activation function to use for the hidden layers. If a string, it must be
        the name of a module in torch.nn. If a list, and not the same length as
        hidden_layer_sizes, the values will be cycled through.
    output_layer_module: str | _FeedForwardModuleConstructor
        The module to use for the output layer. If a string, it must be the name of a
        module in torch.nn.
    n_outputs: int
        The number of outputs from the network.
    output_activation: str | _ActivationLayerConstructor
        The activation function to use for the output layer. If a string, it must be
        the name of a module in torch.nn.
    dropout_rates: float | list[float]
        The dropout rate to use for each layer. If a list, and not the same length as
        hidden_layer_sizes, the values will be cycled through.
    layer_norms: bool | list[bool]
        Whether to use layer normalization for each layer. If a list, and not the same
        length as hidden_layer_sizes, the values will be cycled through.
    inplace_activation: bool
        Whether to use in-place activation functions where possible.

    Examples:
    1. A simple feed forward network with no hidden layers.
    >>> import functorch as ft
    ... module = simple_feed_forward(n_inputs=2, n_outputs=1, hidden_layer_sizes=())
    ... class EquivalentModule(torch.nn.Module):
    ...    def __init__(self):
    ...        super().__init__()
    ...        self.linear = torch.nn.Linear(2, 1)
    ...        self.activation = torch.nn.Identity()
    ...    def forward(self, x):
    ...        return self.activation(self.linear(x))
    ... equivalent_module = EquivalentModule()
    ... func_module, params = ft.make_functional(module)
    ... func_equivalent_module, _ = ft.make_functional(equivalent_module)
    ... x = torch.randn(3, 2)
    ... assert torch.allclose(func_module(params, x), func_equivalent_module(params, x))

    2. A simple feed forward network with one hidden layer.
    >>> import functorch as ft
    ... module = simple_feed_forward(n_inputs=2, n_outputs=1, hidden_layer_sizes=(3,))
    ... class EquivalentModule(torch.nn.Module):
    ...   def __init__(self):
    ...       super().__init__()
    ...       self.linear1 = torch.nn.Linear(2, 3)
    ...       self.activation1 = torch.nn.ReLU(inplace=True)
    ...       self.linear2 = torch.nn.Linear(3, 1)
    ...   def forward(self, x):
    ...       x = self.activation1(self.linear1(x))
    ...       return self.linear2(x)
    ... equivalent_module = EquivalentModule()
    ... func_module, params = ft.make_functional(module)
    ... func_equivalent_module, _ = ft.make_functional(equivalent_module)
    ... x = torch.randn(3, 2)
    ... assert torch.allclose(func_module(params, x), func_equivalent_module(params, x))

    3. A simple feed forward network with two hidden layers and dropout layers.
    >>> import functorch as ft
    ... module = simple_feed_forward(
    ...     n_inputs=2,
    ...     n_outputs=1,
    ...     hidden_layer_sizes=(3, 4),
    ...     dropout_rates=(0.1, 0.2),
    ... )
    ... class EquivalentModule(torch.nn.Module):
    ...   def __init__(self):
    ...       super().__init__()
    ...       self.linear1 = torch.nn.Linear(2, 3)
    ...       self.dropout1 = torch.nn.Dropout(0.1)
    ...       self.activation1 = torch.nn.ReLU(inplace=True)
    ...       self.linear2 = torch.nn.Linear(3, 4)
    ...       self.dropout2 = torch.nn.Dropout(0.2)
    ...       self.activation2 = torch.nn.ReLU(inplace=True)
    ...       self.linear3 = torch.nn.Linear(4, 1)
    ...   def forward(self, x):
    ...       x = self.activation1(self.dropout1(self.linear1(x)))
    ...       x = self.activation2(self.dropout2(self.linear2(x)))
    ...       return self.linear3(x)
    ... equivalent_module = EquivalentModule()
    ... func_module, params = ft.make_functional(module)
    ... func_equivalent_module, _ = ft.make_functional(equivalent_module)
    ... x = torch.randn(3, 2)
    ... func_module.eval()
    ... func_equivalent_module.eval()
    ... assert (
    ...     torch.allclose(func_module(params, x), func_equivalent_module(params, x))
    ... ), (module, equivalent_module)

    4. A simple feed forward network with two hidden layers and layer normalization.
    >>> import functorch as ft
    ... module = simple_feed_forward(
    ...     n_inputs=2,
    ...     n_outputs=1,
    ...     hidden_layer_sizes=(3, 4),
    ...     layer_norms=True,
    ... )
    ... class EquivalentModule(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.linear1 = torch.nn.Linear(2, 3)
    ...         self.activation1 = torch.nn.ReLU(inplace=True)
    ...         self.layer_norm1 = torch.nn.LayerNorm(3)
    ...         self.linear2 = torch.nn.Linear(3, 4)
    ...         self.activation2 = torch.nn.ReLU(inplace=True)
    ...         self.layer_norm2 = torch.nn.LayerNorm(4)
    ...         self.linear3 = torch.nn.Linear(4, 1)
    ...     def forward(self, x):
    ...         x = self.activation1(self.layer_norm1(self.linear1(x)))
    ...         x = self.activation2(self.layer_norm2(self.linear2(x)))
    ...         return self.linear3(x)
    ... equivalent_module = EquivalentModule()
    ... func_module, params = ft.make_functional(module)
    ... func_equivalent_module, _ = ft.make_functional(equivalent_module)
    ... func_module.eval()
    ... func_equivalent_module.eval()
    ... x = torch.randn(5, 2)
    ... assert (
    ...     torch.allclose(func_module(params, x), func_equivalent_module(params, x))
    ... ), (module, equivalent_module)

    5. Something like the DroQ network from the paper:
    [Dropout Q-Functions ...](https://openreview.net/forum?id=xCVJMsPv3RT)
    >>> import functorch as ft
    ... module = simple_feed_forward(
    ...     n_inputs=2,
    ...     n_outputs=1,
    ...     hidden_layer_sizes=(3, 4),
    ...     dropout_rates=0.5,
    ...     layer_norms=True,
    ...     inplace_activation=True,
    ... )
    ... class EquivalentModule(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.linear1 = torch.nn.Linear(2, 3)
    ...         self.activation1 = torch.nn.ReLU(inplace=True)
    ...         self.layer_norm1 = torch.nn.LayerNorm(3)
    ...         self.dropout1 = torch.nn.Dropout(0.5)
    ...         self.linear2 = torch.nn.Linear(3, 4)
    ...         self.activation2 = torch.nn.ReLU(inplace=True)
    ...         self.layer_norm2 = torch.nn.LayerNorm(4)
    ...         self.dropout2 = torch.nn.Dropout(0.5)
    ...         self.linear3 = torch.nn.Linear(4, 1)
    ...     def forward(self, x):
    ...         x = self.activation1(self.layer_norm1(self.dropout1(self.linear1(x))))
    ...         x = self.activation2(self.layer_norm2(self.dropout2(self.linear2(x))))
    ...         return self.linear3(x)
    ... equivalent_module = EquivalentModule()
    ... func_module, params = ft.make_functional(module)
    ... func_equivalent_module, _ = ft.make_functional(equivalent_module)
    ... x = torch.randn(5, 2)
    ... func_module.eval()
    ... func_equivalent_module.eval()
    ... assert (
    ...     torch.allclose(func_module(params, x), func_equivalent_module(params, x))
    ... ), (module, equivalent_module)
    """
    if not hidden_layer_sizes:
        if dropout_rates or layer_norms:
            warnings.warn(
                "dropout_rates and layer_norms are ignored when there are no hidden"
                + " layers."
            )
        return _no_hidden_layers_feed_forward(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            output_layer_module=output_layer_module,
            output_activation=output_activation,
            inplace_activation=inplace_activation,
        )
    hidden_layer_modules = list(mit.always_iterable(hidden_layer_modules))
    hidden_activations = list(mit.always_iterable(hidden_activations))
    dropout_rates = list(mit.always_iterable(dropout_rates))
    layer_norms = list(mit.always_iterable(layer_norms))
    return _with_hidden_layers_feed_forward(
        n_inputs=n_inputs,
        hidden_layer_modules=hidden_layer_modules,
        hidden_layer_sizes=hidden_layer_sizes,
        hidden_activations=hidden_activations,
        output_layer_module=output_layer_module,
        output_activation=output_activation,
        n_outputs=n_outputs,
        dropout_rates=dropout_rates,
        layer_norms=layer_norms,
        inplace_activation=inplace_activation,
    )


def _no_hidden_layers_feed_forward(
    n_inputs: int,
    n_outputs: int,
    output_layer_module: str | _FeedForwardModuleConstructor,
    output_activation: str | _ActivationLayerConstructor,
    inplace_activation: bool,
) -> torch.nn.Module:
    """Construct a feed forward network with no hidden layers."""
    return torch.nn.Sequential(
        as_feed_forward_module(output_layer_module, n_inputs, n_outputs),
        as_activation_module(output_activation, inplace_activation),
    )


def _with_hidden_layers_feed_forward(
    n_inputs: int,
    hidden_layer_modules: list[str | _FeedForwardModuleConstructor],
    hidden_layer_sizes: tuple[int, ...],
    hidden_activations: list[str | _ActivationLayerConstructor],
    output_layer_module: str | _FeedForwardModuleConstructor,
    output_activation: str | _ActivationLayerConstructor,
    n_outputs: int,
    dropout_rates: list[float],
    layer_norms: list[bool],
    inplace_activation: bool,
) -> torch.nn.Module:
    """Construct a feed forward network with hidden layers.

    Args:
    -----
    n_inputs: int
        The number of input features.
    hidden_layer_modules: list[str | _FeedForwardModuleConstructor]
        The modules to use for the hidden layers. If a string is given, it must be
        a valid module name in torch.nn.
    hidden_layer_sizes: tuple[int, ...]
        The number of hidden units in each hidden layer.
    hidden_activations: list[str | _ActivationLayerConstructor]
        The activation functions to use for the hidden layers. If a string is given,
        it must be a valid module name in torch.nn.
    output_layer_module: str | _FeedForwardModuleConstructor
        The module to use for the output layer. If a string is given, it must be
        a valid module name in torch.nn.
    output_activation: str | _ActivationLayerConstructor
        The activation function to use for the output layer. If a string is given,
        it must be a valid module name in torch.nn.
    n_outputs: int
        The number of output features.
    dropout_rates: list[float]
        The dropout rates to use for each hidden layer.
    layer_norms: list[bool]
        Whether to use layer normalization for each hidden layer.
    inplace_activation: bool
        Whether to use in-place activation functions, where possible, for the activation
        layers.

    NOTE: If any of the following sequences are not of the same length as the number of
    desired hidden layers, they will be repeated cyclically. For example, if you need 3
    hidden layers, but only provide ["ReLU", "Sigmoid"] for hidden_activations, then the
    actual hidden activations will be ["ReLU", "Sigmoid", "ReLU"].
    NOTE: hidden_layer_sizes is treated as the number of hidden layers the user desires.
    """
    layers = []
    all_but_one = [n_inputs] + list(hidden_layer_sizes)
    input_output_sizes = zip(all_but_one[:-1], all_but_one[1:])
    for module, (in_size, out_size), activation, dropout_rate, layer_norm in zip(
        it.cycle(hidden_layer_modules),
        input_output_sizes,
        it.cycle(hidden_activations),
        it.cycle(dropout_rates),
        it.cycle(layer_norms),
    ):
        # Hidden layer comes first
        layers.append(as_feed_forward_module(module, in_size, out_size))
        # Dropout layer if any, comes second
        if not math.isclose(dropout_rate, 0):
            layers.append(torch.nn.Dropout(dropout_rate))
        # Layer norm layer if any, comes third
        if layer_norm:
            layers.append(torch.nn.LayerNorm(out_size))
        # Activation layer comes last
        layers.append(as_activation_module(activation, inplace_activation))
    layers.extend(
        (
            as_feed_forward_module(
                output_layer_module, hidden_layer_sizes[-1], n_outputs
            ),
            as_activation_module(output_activation, inplace_activation),
        )
    )

    return torch.nn.Sequential(*layers)
