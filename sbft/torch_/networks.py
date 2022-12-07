"""Various torch.nn.Module subclasses that are not layers live here."""
import torch

from sbft.network_factories import simple_feed_forward


class ContinuousQNetwork(torch.nn.Module):
    def __init__(self, body: torch.nn.Module):
        super().__init__()
        self.body = body

    def forward(  # type: ignore
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.body(torch.cat([states, actions], dim=-1))


class QuantileEnsembleCritics(torch.nn.Module):
    def __init__(
        self,
        critics: list[torch.nn.Module],
        n_quantiles: int,
    ) -> None:
        super().__init__()
        self.critics = torch.nn.ModuleList(critics)
        self.n_quantiles = n_quantiles
        self.n_total_quantiles = n_quantiles * len(critics)
        self.n_critics = len(critics)

    @classmethod
    def feed_forward(
        cls,
        state_dim: int,
        action_dim: int,
        n_critics: int = 2,
        n_quantiles: int = 1,
        hidden_layer_sizes: tuple[int, ...] = (512, 512, 512),
        hidden_activation: str | type[torch.nn.Module] = "ReLU",
        output_activation: str | type[torch.nn.Module] = "Identity",
        dropout_rates: float | tuple[float] = 0.0,
        layer_norms: bool | tuple[bool] = False,
        device: torch.device | str = "cpu",
    ) -> "QuantileEnsembleCritics":
        device = device
        critics = [
            simple_feed_forward(
                n_inputs=state_dim + action_dim,
                n_outputs=n_quantiles,
                hidden_layer_sizes=hidden_layer_sizes,
                hidden_activations=hidden_activation,  # type: ignore
                output_activation=output_activation,  # type: ignore
                dropout_rates=dropout_rates,
                layer_norms=layer_norms,
            )
            for _ in range(n_critics)
        ]
        return cls(
            critics=critics,
            n_quantiles=n_quantiles,
        ).to(device)

    def forward(  # type: ignore
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        random_ensemble_size: int | None = None,
        force_random_ensemble: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, ...]:
        random_ensemble_size = random_ensemble_size or self.n_critics
        state_actions = torch.cat([states, actions], dim=-1)
        if force_random_ensemble or random_ensemble_size < self.n_critics:
            sample_idxs: list[int] = (
                torch.randint(
                    low=0,
                    high=self.n_critics,
                    size=(random_ensemble_size,),
                    generator=generator,
                )
                .cpu()
                .tolist()
            )
        else:
            sample_idxs = list(range(self.n_critics))
        return tuple(self.critics[idx](state_actions) for idx in sample_idxs)
