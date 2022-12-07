"""The main logic to run the training loop."""
import time
import typing as t
from unittest.mock import Mock

import torch
import wandb
from gymnasium import spaces as gym_spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector import make as vector_make  # type: ignore
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from wandb.sdk.wandb_run import Run

from sbft import external_utils as extu
from sbft.agents.sac.config import SACConfig
from sbft.agents.sac.models import SACTrainState, create_models_and_params
from sbft.memories import ExperienceReplay
from sbft.observers import TransitionObserver


def run(config: SACConfig) -> None:  # sourcery skip: low-code-quality
    """Run the experiment with SAC agent using the given configuration."""
    # Initialize experiment tracking
    if config.wandb:
        wandb.init(**config.wandb_init_config)
        wandb_run: Run = t.cast(Run, wandb.run)
    else:
        wandb_run: Run = Mock(spec=Run)

    cli_ui = extu.CliUI(config.exp_name, config=config)

    # Handle reproducibility
    extu.seed_all_randomness(
        seed=config.seed,
        cuda_deterministic=config.cuda_deterministic,
        cudnn_deterministic=config.cudnn_deterministic,
    )
    generator = torch.Generator().manual_seed(config.seed)
    # Create the environment
    envs: VectorEnv = vector_make(
        config.env_id,
        num_envs=1,
        asynchronous=False,
        wrappers=RecordEpisodeStatistics,
    )
    memory = ExperienceReplay.from_env(
        batch_size=config.batch_size,
        capacity=config.buffer_size,
        env=envs,
    )
    assert isinstance(envs.single_action_space, gym_spaces.Box)
    # Create the models and parameters
    model_graph, params = create_models_and_params(envs, config)
    observer = TransitionObserver(
        env=envs,
        memory=memory,
        policy=model_graph.policy,
        ui=cli_ui,
    )
    # Create the training state(s)
    # The train states will get updated over the course of training.
    actor_train_state, critic_train_state, alpha_train_state = SACTrainState.create(
        config=config,
        params=params,
        model_graph=model_graph,
    )
    actor_loss = alpha_loss = 0
    cli_ui.start()
    random_sampling_task = cli_ui.training_progress_bar.add_task(
        "[blue]Random exploration",
        total=config.learning_starts,
    )
    for _ in range(config.learning_starts):
        observer.observe(params.actor)
        cli_ui.training_progress_bar.update(random_sampling_task, advance=1)

    start_time = time.time()

    # Start the training loop
    train_progress_bar = cli_ui.training_progress_bar.add_task(
        "[green]Training",
        total=config.learning_timesteps,
    )
    for global_step in range(config.learning_timesteps):
        observer.observe(params.actor)
        # Sample a batch of experiences
        batch = memory.get().as_tensors(config.torch_device)
        # Update training states
        # Update the critic.
        critic_train_state, critic_loss = critic_train_state.step_and_value(
            params.actor,
            params.target_critic,
            batch,
            params.alpha,
            generator,
        )
        if global_step % config.policy_update_freq == 0:  # TD 3 Delayed update support
            # compensate for the delay by doing 'actor_update_interval' instead of 1
            for _ in range(config.policy_update_freq):
                # Update the actor
                actor_train_state, actor_loss = actor_train_state.step_and_value(
                    params.critic, batch, params.alpha, generator
                )
                # Update the entropy
                (
                    alpha_train_state,
                    alpha_loss,
                ) = alpha_train_state.step_and_value(params.actor, batch, generator)

        # update the target networks
        if global_step % config.target_network_update_freq == 0:
            target_critic_params = extu.polyak_average(
                critic_train_state.params,
                params.target_critic,
                config.polyak_factor,
            )
        else:
            target_critic_params = None
        params = params.evolve(
            actor=actor_train_state.params,
            critic=critic_train_state.params,
            target_critic=target_critic_params,
            entropy=alpha_train_state.params,
        )

        if global_step % 100 == 0:
            wandb_run.log(
                {
                    "vars/alpha": alpha_train_state.params[0],
                    "losses/critic_loss": critic_loss,
                    "losses/actor_loss": actor_loss,
                    "losses/alpha_loss": alpha_loss,
                    "throughputs/samples": int(
                        global_step / (time.time() - start_time)
                    ),
                }
            )
        cli_ui.training_progress_bar.update(train_progress_bar, advance=1)

    envs.close()
    cli_ui.stop()


if __name__ == "__main__":
    config = SACConfig()
    run(config)
