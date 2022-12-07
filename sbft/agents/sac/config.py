import typing as t

import pydantic as pdt

from sbft.config import ExperimentConfig


class SACConfig(ExperimentConfig):
    """Config specific to the SAC algorithm."""

    env_id: str = pdt.Field(
        default="Pendulum-v1", description="Name of the gymnasium environment."
    )
    learning_timesteps: int = pdt.Field(
        default=int(1e5),  # Sufficient for Pendulum-v1
        description="Total number of timesteps to train the neural network for.",
    )
    buffer_size: int = pdt.Field(
        default=10000, description="Size of the replay buffer."
    )
    discount_factor: float = pdt.Field(
        default=0.99,
        description="Discount factor, gamma, for the cumulative return.",
    )
    polyak_factor: float = pdt.Field(
        default=0.005,
        description="Soft target network update polyak averaging factor.",
    )
    batch_size: int = pdt.Field(
        default=256,
        description="Batch size for SAC training.",
    )
    learning_starts: int = pdt.Field(
        default=10000,
        description="Number of environment steps to collect before starting training.",
    )
    actor_log_std_min: float = pdt.Field(
        default=-5.0,
        description="Minimum value of the log standard deviation of the actor's "
        + "output distribution.",
    )
    q_network_hidden_layers: list[int] = pdt.Field(
        default=[256, 256],
        description="Hidden layer sizes for the SoftQNetwork.",
    )
    actor_hidden_layers: list[int] = pdt.Field(
        default=[256, 256],
        description="Hidden layer sizes for the ActorNetwork.",
    )
    actor_log_std_max: float = pdt.Field(
        default=2.0,
        description="Maximum value of the log standard deviation of the actor's "
        + "output distribution.",
    )
    policy_lr: float = pdt.Field(
        default=3e-4,
        description="Learning rate for the policy network.",
    )
    q_lr: float = pdt.Field(
        default=3e-4,
        description="Learning rate for the Q network.",
    )
    policy_update_freq: int = pdt.Field(
        default=2,
        description="Frequency of updating the policy network.",
    )
    target_network_update_freq: int = pdt.Field(
        default=1,
        description="Frequency of updating the target network.",
    )
    entropy_coef: float = pdt.Field(
        default=0.2,
        description="Entropy regularization coefficient.",
    )
    tune_entropy: bool = pdt.Field(
        default=False,
        description="Whether to automatically tune the entropy coefficient.",
    )


def load_config(data: dict[str, t.Any]) -> SACConfig:
    """Load a SAC config from a dictionary."""
    return SACConfig(**data)
