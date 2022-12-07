import torch
from torchsilk.distributions import NormalDistParams


class Concat(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(  # type: ignore[override]
        self, *args: torch.Tensor, **kwargs: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([*args, *kwargs.values()], dim=self.dim)


class SoftNormalDistParamsLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.mean_layer = torch.nn.Linear(in_features, out_features)
        self.log_std_layer = torch.nn.Linear(in_features, out_features)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x: torch.Tensor) -> NormalDistParams:  # type: ignore[override]
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1.0
        )
        std = log_std.exp()
        return NormalDistParams(loc=mean, scale=std)
