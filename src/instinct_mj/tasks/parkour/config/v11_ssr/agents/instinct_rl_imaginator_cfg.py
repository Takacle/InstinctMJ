"""V11 SSR parkour agent configuration.

Composes the existing V11 parkour MoE policy + WASABI AMP algorithm with
the ``ImaginatorAlgoMixin`` (provided by instinct_rl). The imagination
MLP is owned by the algorithm and trained with Gaussian NLL on the
``foothold_privilege`` -> ``foothold_target`` supervision pair.

Reference: Yu et al., SSR, 2026, Section 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from instinct_mj.rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlImaginatorAlgoMixinCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@dataclass(kw_only=True)
class DepthEncoderConv2dCfg(InstinctRlConv2dHeadCfg):
    """Depth-image encoder (mirrors the V11 parkour baseline)."""

    output_size: int = 128
    channels: list = field(default_factory=lambda: [4])
    kernel_sizes: list = field(default_factory=lambda: [3])
    strides: list = field(default_factory=lambda: [1])
    hidden_sizes: list = field(default_factory=lambda: [256, 256])
    paddings: list = field(default_factory=lambda: [1])
    nonlinearity: str = "ReLU"
    use_maxpool: bool = True
    component_names: list = field(default_factory=lambda: ["depth_image"])


@dataclass(kw_only=True)
class EncoderConfigs:
    depth_encoder: object = field(default_factory=lambda: DepthEncoderConv2dCfg())


@dataclass(kw_only=True)
class MoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    """Same MoE encoder policy as the V11 parkour baseline.

    The imagination MLP lives on the algorithm side, so the policy only
    needs the encoder + MoE backbone.
    """

    init_noise_std: float = 1.0
    num_moe_experts: int = 4
    actor_hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    critic_hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    activation: str = "elu"
    encoder_configs: object = field(default_factory=lambda: EncoderConfigs())
    critic_encoder_configs: object = field(default_factory=lambda: EncoderConfigs())


@dataclass(kw_only=True)
class ImaginatorAmpAlgoCfg(InstinctRlPpoAlgorithmCfg, InstinctRlImaginatorAlgoMixinCfg):
    """WASABI AMP + Imaginator algorithm configuration.

    ``class_name="ImaginatorWasabiPPO"`` composes (ImaginatorAlgoMixin,
    WasabiAlgoMixin, PPO) so the imagination MLP is trained alongside the
    AMP discriminator and the PPO policy.

    Stage 2a: ``foothold_reward_coef=0.0`` disables the imagined auxiliary
    reward; only the contact-time ``foothold_support_deficiency`` env reward
    is active. Stage 2b will set this to a positive value once the
    imagination model is validated.
    """

    class_name: str = "ImaginatorWasabiPPO"
    # WASABI discriminator configuration (mirrors V11 parkour baseline).
    discriminator_kwargs: dict = field(
        default_factory=lambda: {
            "hidden_sizes": [1024, 512],
            "nonlinearity": "ReLU",
        }
    )
    discriminator_reward_coef: float = 0.25
    discriminator_reward_type: str = "quad"
    discriminator_loss_func: str = "MSELoss"
    discriminator_gradient_penalty_coef: float = 5.0
    discriminator_optimizer_class_name: str = "AdamW"
    discriminator_weight_decay_coef: float = 3e-4
    discriminator_logit_weight_decay_coef: float = 0.04
    discriminator_optimizer_kwargs: dict = field(
        default_factory=lambda: {
            "lr": 1.0e-4,
            "betas": [0.9, 0.999],
        }
    )
    # PPO configuration (mirrors V11 parkour baseline).
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.004
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    # Imaginator configuration (stage 2a defaults).
    imaginator_privilege_key: str = "foothold_privilege"
    imaginator_target_key: str = "foothold_target"
    imaginator_hidden_sizes: tuple[int, ...] = (256, 128)
    imaginator_nonlinearity: str = "ELU"
    imaginator_loss_coef: float = 1.0
    imaginator_optimizer_class_name: str = "AdamW"
    imaginator_optimizer_kwargs: dict = field(
        default_factory=lambda: {"lr": 5e-4}
    )
    imaginator_num_feet: int = 2
    imaginator_target_dim_per_foot: int = 2
    # Stage 2a: imagined auxiliary reward disabled (set >0 in stage 2b).
    foothold_reward_coef: float = 0.0
    sigma_f: float = 0.0625


@dataclass(kw_only=True)
class V11SsrPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    policy_observation_group: str = "policy"
    critic_observation_group: str = "critic"
    max_iterations: int = 50000
    save_interval: int = 1000
    experiment_name: str = "v11_ssr"
    resume: bool = False
    load_run: str = "^(?!_play$).*"
    empirical_normalization: bool = False
    policy: object = field(default_factory=lambda: MoEPolicyCfg())
    algorithm: object = field(default_factory=lambda: ImaginatorAmpAlgoCfg())
