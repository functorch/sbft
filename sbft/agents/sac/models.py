import typing as t

import attrs
import numpy as np
import torch
import torch.nn.functional as F
import torchopt as tpt
import torchsilk as tsk
import typing_extensions as tx
from gymnasium import spaces as gym_spaces
from gymnasium.vector import VectorEnv

from sbft import external_utils as extu
from sbft.agents.sac import initializers as tsi
from sbft.agents.sac.config import SACConfig
from sbft.api import Policy
from sbft.experience import TensorVanillaExperience
from sbft.network_factories import simple_feed_forward
from sbft.torch_ import layers, networks

ActorParams: t.TypeAlias = tuple[torch.nn.Parameter, ...]
CriticParams: t.TypeAlias = tuple[torch.nn.Parameter, ...]
EntropyParams: t.TypeAlias = tuple[torch.nn.Parameter]


def create_critic(env: VectorEnv, config: SACConfig) -> torch.nn.Module:
    """Create the critic network.

    Args:
    -----
        env: Used for calculating action and state dimensions
        config: Config containing architecture of network

    Returns:
    --------
        The critic network that takes in states and actions and outputs Q values.

    Example:
    --------

    >>> import torch
    ... from gymnasium.vector import make as vector_make
    ... from sbft.agents.sac import SACConfig
    ... env = vector_make("Pendulum-v1", 3)
    ... config = SACConfig(
    ...     q_network_hidden_layers=[32, 32],
    ... )
    ... critic = create_critic(env, config)
    ... qf1, qf2 = critic(torch.randn(1, 3), torch.randn(1, 1))
    ... assert isinstance(qf1, torch.Tensor)
    ... assert isinstance(qf2, torch.Tensor)

    # This is typically used to create a functional version of the network, and the
    original network is not used for training. The returned parameters are used to
    initialize the parameters of the functional version of the network.
    >>> from sbft.agents.sac import initializers as tsi
    ... func_critic, params = tsk.make_functional(critic)
    ... generator = torch.Generator().manual_seed(0)
    ... qf1, qf2 = func_critic(params, torch.randn(1, 3), torch.randn(1, 1))
    ... assert isinstance(qf1, torch.Tensor)
    ... assert isinstance(qf2, torch.Tensor)
    ... feed_forward_init = tsi.init_feed_forward_like(
    ...     weights_init=tsi.orthogonal_like
    ... )
    ... new_params = tsi.init_like_params(
    ...     params, generator, initializer=feed_forward_init
    ... )
    ... new_qf1, new_qf2 = func_critic(new_params, torch.randn(1, 3), torch.randn(1, 1))
    ... assert not torch.allclose(qf1, new_qf1)
    ... assert not torch.allclose(qf2, new_qf2)
    """
    action_dim = np.prod(env.single_action_space.shape)  # type: ignore
    state_dim = np.prod(env.single_observation_space.shape)  # type: ignore
    return networks.QuantileEnsembleCritics.feed_forward(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layer_sizes=config.q_network_hidden_layers,
        hidden_activation="ReLU",
        device=config.torch_device,
    )


def create_actor(env: VectorEnv, config: SACConfig) -> torch.nn.Module:
    """Create the actor network.

    Args:
    -----
        env: Used for calculating action and state dimensions
        config: Config containing architecture of network

    Returns:
    --------
        The actor network that takes in states and outputs the parameters of the action
        distribution.

    Example:
    --------
    >>> import torch
    ... from gymnasium.vector import make as vector_make
    ... from sbft.agents.sac import SACConfig
    ... env = vector_make("Pendulum-v1", 2)
    ... config = SACConfig(
    ...     actor_hidden_layers=[32, 32],
    ... )
    ... actor = create_actor(env, config)
    ... assert isinstance(actor(torch.randn(1, 3)).mean, torch.Tensor)

    # This is typically used to create a functional version of the network, and the
    original network is not used for training. The returned parameters are used to
    initialize the parameters of the functional version of the network.

    >>> from sbft.agents.sac import initializers as tsi
    ... func_actor, params = tsk.make_functional(actor)
    ... generator = torch.Generator().manual_seed(0)
    ... init_result = func_actor(params, torch.randn(1, 3))
    ... assert isinstance(init_result.mean, torch.Tensor)
    ... feed_forward_init = tsi.init_feed_forward_like(
    ...     weights_init=tsi.orthogonal_like
    ... )
    ... new_params = tsi.init_like_params(
    ...     params, generator=generator, initializer=feed_forward_init
    ... )
    ... new_result = func_actor(new_params, torch.randn(1, 3))
    ... assert isinstance(new_result.mean, torch.Tensor)
    ... assert not torch.allclose(init_result.mean, new_result.mean)
    """
    action_dim = np.prod(env.single_action_space.shape)  # type: ignore
    state_dim = np.prod(env.single_observation_space.shape)  # type: ignore
    return torch.nn.Sequential(
        simple_feed_forward(
            n_inputs=state_dim,
            n_outputs=config.actor_hidden_layers[-1],
            hidden_layer_sizes=config.actor_hidden_layers[:-1],
            hidden_activations="ReLU",
        ),
        layers.SoftNormalDistParamsLayer(config.actor_hidden_layers[-1], action_dim),
    ).to(config.torch_device)


class _ActionResult(t.NamedTuple):
    """Result of taking an action in the environment."""

    actions: torch.Tensor
    log_probs: torch.Tensor


@attrs.define()
class SACPolicy(Policy):
    """The policy for SAC.

    Attributes:
    -----------
        actor: The actor network
        action_bias: The bias to add to the actions
        action_scale: The scale to multiply the actions by

    Examples:
    ---------

    >>> import torch
    ... from gymnasium.vector import make as vector_make
    ... from sbft.agents.sac import SACConfig
    ... env = vector_make("Pendulum-v1")
    ... config = SACConfig(
    ...     actor_hidden_layers=[32, 32],
    ... )
    ... actor = create_actor(env, config)
    ... func_actor, params = tsk.make_functional(actor)
    ... policy = SACPolicy(
    ...     actor=func_actor,
    ...     action_scale=torch.tensor(2.0),
    ...     action_bias=torch.tensor(0.0),
    ... )
    ... assert isinstance(policy(params, torch.randn(1, 3)), torch.Tensor)
    ... assert isinstance(policy(params, torch.randn(1, 3), eval=True), torch.Tensor)
    ... generator = torch.Generator().manual_seed(0)
    ... feed_forward_init = tsi.init_feed_forward_like(
    ...     weights_init=tsi.orthogonal_like
    ... )
    ... new_params = tsi.init_like_params(
    ...     params, generator=generator, initializer=feed_forward_init
    ... )
    ... actions = policy(new_params, torch.randn(1, 3, generator=generator), eval=True)
    ... assert torch.allclose(
    ...     actions, torch.tensor([[-0.4081]]), atol=1e-4, rtol=0
    ... ), actions
    """

    actor: tsk.FunctionalModule[[torch.Tensor], tsk.distributions.NormalDistParams]
    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __call__(
        self,
        params: tuple[torch.nn.Parameter, ...],
        states: torch.Tensor,
        *,
        t: int | None = None,
        eval: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return actions to take in the given states for the given parameters.

        Args:
        -----
            params: The parameters of the actor (function approximator).
            states: The states of the environment to take actions in.
            t: The global timestep. Unused.
            eval: Whether or not to use the evaluation mode of the policy.
                For Gaussian policies, this means using the mean instead of sampling.
            generator: The generator to use for sampling random numbers.

        Returns:
        --------
            actions: The actions to take in the given states.
        """
        dist_params: tsk.distributions.NormalDistParams = self.actor(params, states)
        unscaled_actions = (
            dist_params.mean
            if eval
            else tsk.distributions.Normal().rsample(
                params=dist_params, generator=generator
            )
        )
        return extu.soft_rescale_action(
            unscaled_actions,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
        )

    def actions_and_log_probs(
        self,
        params: tuple[torch.nn.Parameter, ...],
        states: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> _ActionResult:
        """Return log probabilities along with actions.

        Useful for computing the policy loss.

        Args:
        -----
            params: The parameters of the actor (function approximator).
            states: The states of the environment to take actions in.
            generator: The generator to use for sampling random numbers.

        Returns:
        --------
            actions: The actions to take in the given states.
            log_probs: The log probabilities of the actions.
            means: The evaluation actions (means) of the policy.
        """
        dist_params: tsk.distributions.NormalDistParams = self.actor(params, states)
        normal = tsk.distributions.Normal()
        x_t = normal.rsample(params=dist_params, generator=generator)
        actions = extu.soft_rescale_action(
            x_t, action_scale=self.action_scale, action_bias=self.action_bias
        )
        log_probs = normal.log_prob(params=dist_params, value=x_t)
        # Enforcing Action Bound
        log_probs -= torch.log(self.action_scale * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        return _ActionResult(actions=actions, log_probs=log_probs)


class SACParams(t.NamedTuple):
    """The parameters for SAC."""

    actor: ActorParams
    critic: CriticParams
    target_critic: CriticParams
    entropy: EntropyParams

    @property
    def alpha(self) -> float:
        """Return the current value of alpha."""
        return self.entropy[0].exp().item()

    def evolve(
        self: tx.Self,
        *,
        actor: ActorParams | None = None,
        critic: CriticParams | None = None,
        target_critic: CriticParams | None = None,
        entropy: EntropyParams | None = None,
    ) -> tx.Self:
        """Return a new SACParams with the given parameters updated."""
        return self.__class__(
            actor=actor if actor is not None else self.actor,
            critic=critic if critic is not None else self.critic,
            target_critic=target_critic
            if target_critic is not None
            else self.target_critic,
            entropy=entropy if entropy is not None else self.entropy,
        )


@attrs.define()
class SACModelGraph:
    critic: tsk.FunctionalModule[[torch.Tensor, torch.Tensor], torch.Tensor]
    policy: SACPolicy
    config: SACConfig
    target_entropy: float

    @property
    def critic_loss_fn(
        self,
    ) -> tsk.LossFunction[
        CriticParams,
        [
            ActorParams,
            CriticParams,
            TensorVanillaExperience,
            float,
            torch.Generator | None,
        ],
        torch.Tensor,
    ]:
        """Create the critic loss function.

        Returns:
        --------
            A function that computes the critic loss. The signature of the function is:
            (
                params, actor_params, target_critic_params, exp, alpha, generator
            ) -> torch.Tensor
            where:
                params: Parameters of the two Q funtions.
                actor_params: The parameters of the actor.
                target_critic_params: The parameters of the target critic.
                exp: The experience to use to compute the loss.
                alpha: The current value of alpha.
                generator: The generator to use for sampling random numbers.
            returns:
                The critic loss for both q functions combined.
        """

        @torch.no_grad()
        def get_next_q_values(
            actor_params: ActorParams,
            target_critic_params: CriticParams,
            exp: TensorVanillaExperience,
            alpha: float,
            generator: torch.Generator | None = None,
        ) -> torch.Tensor:
            actions, log_probs = self.policy.actions_and_log_probs(
                params=actor_params, states=exp.next_states, generator=generator
            )
            next_qf1, next_qf2 = self.critic(
                target_critic_params, exp.next_states, actions
            )
            min_qf_next_target = torch.min(next_qf1, next_qf2) - alpha * log_probs
            return exp.rewards.flatten() + (
                ~exp.dones.flatten()
            ) * self.config.discount_factor * min_qf_next_target.view(-1)

        def get_qf_loss(
            params: CriticParams,
            actor_params: ActorParams,
            target_critic_params: CriticParams,
            exp: TensorVanillaExperience,
            alpha: float,
            generator: torch.Generator | None = None,
        ) -> torch.Tensor:
            qf1_values, qf2_values = self.critic(params, exp.states, exp.actions)
            next_q_values = get_next_q_values(
                actor_params=actor_params,
                target_critic_params=target_critic_params,
                exp=exp,
                alpha=alpha,
                generator=generator,
            )
            qf1_loss = F.mse_loss(qf1_values.view(-1), next_q_values)
            qf2_loss = F.mse_loss(qf2_values.view(-1), next_q_values)
            return qf1_loss + qf2_loss

        return get_qf_loss

    @property
    def actor_loss_fn(
        self,
    ) -> tsk.LossFunction[
        ActorParams,
        [CriticParams, TensorVanillaExperience, float, torch.Generator | None],
        torch.Tensor,
    ]:
        """Create the actor loss function.

        Returns:
        --------
            A function that computes the actor loss. The signature of the function is:
            (
                params, critic_params, exp, alpha, generator
            ) -> torch.Tensor
            where:
                params: The parameters of the actor.
                critic_params: The parameters of the critic.
                exp: The experience to use to compute the loss.
                alpha: The current value of alpha.
                generator: The generator to use for sampling random numbers.
        """

        def get_actor_loss(
            params: ActorParams,
            critic_params: CriticParams,
            exp: TensorVanillaExperience,
            alpha: float,
            generator: torch.Generator | None = None,
        ) -> torch.Tensor:
            actions, log_probs = self.policy.actions_and_log_probs(
                params=params, states=exp.next_states, generator=generator
            )
            qf1, qf2 = self.critic(critic_params, exp.states, actions)
            min_qf = torch.min(qf1, qf2).view(-1)
            return (alpha * log_probs - min_qf).mean()

        return get_actor_loss

    @property
    def entropy_loss_fn(
        self,
    ) -> tsk.LossFunction[
        tuple[torch.nn.Parameter, ...],
        [
            tuple[torch.nn.Parameter, ...],
            TensorVanillaExperience,
            torch.Generator | None,
        ],
        torch.Tensor,
    ]:
        """Create the alpha loss function.

        Returns:
        --------
            A function that computes the alpha loss. The signature of the function is:
            (
                params, actor_params, exp, generator
            ) -> torch.Tensor
            where:
                params: The parameters of the alpha (log_alpha).
                actor_params: The parameters of the actor.
                exp: The experience to use to compute the loss.
                generator: The generator to use for sampling random numbers.
            returns:
                The alpha loss.
        """

        def get_alpha_loss(
            params: tuple[torch.nn.Parameter, ...],
            actor_params: tuple[torch.nn.Parameter, ...],
            exp: TensorVanillaExperience,
            generator: torch.Generator | None = None,
        ) -> torch.Tensor:
            alpha = params[0].exp()
            with torch.no_grad():
                _, log_probs = self.policy.actions_and_log_probs(
                    params=actor_params, states=exp.next_states, generator=generator
                )
            return (alpha * (-log_probs - self.target_entropy)).mean()

        return get_alpha_loss


def create_models_and_params(
    envs: VectorEnv,
    config: SACConfig,
) -> tuple[SACModelGraph, SACParams]:
    """Create the initial training state and parameters.

    Args:
    -----
        envs: The vectorized environment.
        config: The configuration.
    Returns:
    --------
        SACModelGraph: The model graph containing the models and loss functions for SAC.
        SACParams: The initial parameters of the models.

    Example:
    --------

    >>> import torch
    ... from gymnasium.vector import make as vector_make
    ... from sbft.agents.sac import SACConfig
    ... from sbft.agents.sac.models import create_models_and_params
    ... config = SACConfig()
    ... envs = vector_make("Pendulum-v1", num_envs=2)
    ... model_graph, params = create_models_and_params(envs, config)
    ... states, _ = envs.reset(seed=0)
    ... states = torch.as_tensor(states)
    ... generator = torch.Generator().manual_seed(config.seed)
    ... actions = model_graph.policy(params.actor, states, generator=generator)
    ... assert torch.allclose(
    ...     actions, torch.tensor([[ 0.0843], [-0.0948]]), atol=1e-4, rtol=0
    ... ), actions
    """
    assert isinstance(envs.single_action_space, gym_spaces.Box)
    action_scale = torch.tensor(
        (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
        dtype=torch.float32,
    )
    action_bias = torch.tensor(
        (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
        dtype=torch.float32,
    )

    actor_module = create_actor(envs, config)
    critic_module = create_critic(envs, config)
    func_actor, orig_actor_params = tsk.make_functional(actor_module)
    func_critic, orig_critic_params = tsk.make_functional(critic_module)
    generator = torch.Generator().manual_seed(config.seed)
    feed_forward_init = tsi.init_feed_forward_like(weights_init=tsi.xavier_uniform_like)
    actor_params = tuple(
        tsi.init_like_params(
            orig_actor_params, generator=generator, initializer=feed_forward_init
        )
    )
    critic_params = tuple(
        tsi.init_like_params(
            orig_critic_params, generator=generator, initializer=feed_forward_init
        )
    )
    target_entropy = -np.prod(envs.single_action_space.shape).item()
    log_alpha = torch.zeros(1, requires_grad=False, device=config.torch_device)
    sac_params = SACParams(
        actor=actor_params,
        critic=critic_params,
        target_critic=critic_params,
        entropy=(log_alpha,),
    )
    sac_model_graph = SACModelGraph(
        critic=func_critic,
        policy=SACPolicy(func_actor, action_scale, action_bias),
        config=config,
        target_entropy=target_entropy,
    )
    return sac_model_graph, sac_params


class SACTrainState(t.NamedTuple):
    """The training state for SAC."""

    actor: tsk.TrainState[
        # The parameters of the actor.
        ActorParams,
        # Other parameters of the loss function
        [
            # The parameters of the critic.
            CriticParams,
            # The experience to use to compute the loss.
            TensorVanillaExperience,
            # The current value of alpha.
            float,
            # The generator to use for sampling random numbers.
            torch.Generator | None,
        ],
        # The return value of the loss function.
        torch.Tensor,
        # The actor optimizer state.
        tuple[torch.Tensor, ...],
    ]
    critic: tsk.TrainState[
        # The parameters of the critic.
        CriticParams,
        # Other parameters of the loss function
        [
            # The parameters of the actor.
            ActorParams,
            # The parameters of the target critics.
            CriticParams,
            # The experience to use to compute the loss.
            TensorVanillaExperience,
            # The current value of alpha.
            float,
            # The generator to use for sampling random numbers.
            torch.Generator | None,
        ],
        # The return value of the loss function.
        torch.Tensor,
        # The critic optimizer state.
        tuple[torch.Tensor, ...],
    ]
    entropy: tsk.TrainState[
        # The parameters of the alpha (log_alpha).
        EntropyParams,
        # Other parameters of the loss function
        [
            # The parameters of the actor.
            ActorParams,
            # The experience to use to compute the loss.
            TensorVanillaExperience,
            # The generator to use for sampling random numbers.
            torch.Generator | None,
        ],
        # The return value of the loss function.
        torch.Tensor,
        # The entropy coefficient optimizer state.
        tuple[torch.Tensor, ...],
    ]

    @classmethod
    def create(
        cls,
        config: SACConfig,
        params: SACParams,
        model_graph: SACModelGraph,
    ) -> "SACTrainState":
        """Create an instance of SACTrainState.

        Args:
        -----
            config: The configuration.
            params: The parameters of the models.
            model_graph: The model graph containing models and loss functions.
        Returns:
        --------
            SACTrainState: The created training state.

        Example:
        --------

        >>> import torch
        ... from gymnasium.vector import make as vector_make
        ... from sbft.memories import ExperienceReplay
        ... from sbft.observers import TransitionObserver
        ... from sbft.agents.sac import SACConfig
        ... from sbft.agents.sac.models import create_models_and_params, SACTrainState
        ... config = SACConfig()
        ... envs = vector_make("Pendulum-v1", num_envs=2)
        ... model_graph, params = create_models_and_params(envs, config)
        ... generator = torch.Generator("cpu").manual_seed(config.seed)
        ... memory = ExperienceReplay.from_env(capacity=1000, batch_size=8, env=envs)
        ... observer = TransitionObserver(envs, memory, model_graph.policy)
        ... for _ in range(10):  # Initial samples
        ...     observer.observe(params.actor)
        ... state = SACTrainState.create(config, params, model_graph)
        ... for _ in range(15):  # Training loop
        ...     observer.observe(params.actor)
        ...     batch = memory.get().as_tensors(config.torch_device)
        ...     critic_state = state.critic.step(
        ...         params.actor,
        ...         params.target_critic,
        ...         batch,
        ...         params.alpha,
        ...         generator,
        ...     )
        ...     actor_state = state.actor.step(
        ...         params.critic,
        ...         batch,
        ...         params.alpha,
        ...         generator,
        ...     )
        ...     entropy_state = state.entropy.step(
        ...         params.actor,
        ...         batch,
        ...         generator,
        ...     )
        """
        actor_opt = tpt.adam(config.policy_lr)
        critic_opt = tpt.adam(config.q_lr)
        alpha_opt = tpt.adam(config.q_lr)
        return cls(
            actor=tsk.TrainState(
                params=params.actor,
                has_aux=False,
                loss_fn=model_graph.actor_loss_fn,
                opt_func=actor_opt,
            ),
            critic=tsk.TrainState(
                params=params.critic,
                has_aux=False,
                loss_fn=model_graph.critic_loss_fn,
                opt_func=critic_opt,
            ),
            entropy=tsk.TrainState(
                params=params.entropy,
                has_aux=False,
                loss_fn=model_graph.entropy_loss_fn,
                opt_func=alpha_opt,
            ),
        )
