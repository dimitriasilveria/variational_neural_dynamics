from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn


class FiLMBlock(nn.Module):
    """Dense block modulated by a latent code using FiLM conditioning."""

    hidden_dim: int
    nonlinearity: Callable = nn.gelu
    initial_scale: float = 1.0

    @nn.compact
    def __call__(self, x: jax.Array, z: jax.Array) -> jax.Array:
        hidden = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
            name="dense",
        )(x)

        # Zero initialization makes FiLM an identity transformation initially:
        # hidden * (1 + scale) + shift == hidden.
        scale_shift = nn.Dense(
            2 * self.hidden_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="film",
        )(z)
        scale, shift = jnp.split(scale_shift, 2, axis=-1)
        return self.nonlinearity(hidden * (1.0 + scale) + shift)


class LatentConditionedResidualDynamics(nn.Module):
    """Latent-conditioned quadrotor residual model from the VND paper.

    The network uses a shared MLP backbone with FiLM conditioning and separate
    heads for position, velocity, and (optionally) orientation corrections.
    Orientation is represented as a three-dimensional rotation vector.

    Inputs:
        state_action: Current state and action, with the feature dimension set by
            ``state_action_dim``.
        z: A 12-dimensional latent condition by default.

    Output:
        Corrections concatenated as ``[delta_position, delta_velocity,
        delta_orientation]``. The output dimension is 9 when orientation is
        enabled and 6 otherwise.

    Example:
        >>> model = LatentConditionedResidualDynamics(state_action_dim=14)
        >>> params = model.initialize(jax.random.key(0))
        >>> prediction = model.apply(params, jnp.zeros(14), jnp.zeros(12))
        >>> prediction.shape
        (9,)
    """

    state_action_dim: int
    latent_dim: int = 12
    hidden_dim: int = 256
    num_blocks: int = 2
    predict_orientation: bool = True
    nonlinearity: Callable = nn.gelu
    initial_scale: float = 1.0

    @nn.compact
    def __call__(self, state_action: jax.Array, z: jax.Array) -> jax.Array:
        if state_action.shape[-1] != self.state_action_dim:
            raise ValueError(
                f"Expected {self.state_action_dim} state-action features, "
                f"got {state_action.shape[-1]}"
            )
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected a {self.latent_dim}-D latent, got {z.shape[-1]}"
            )

        x = state_action
        for block_idx in range(self.num_blocks):
            x = FiLMBlock(
                hidden_dim=self.hidden_dim,
                nonlinearity=self.nonlinearity,
                initial_scale=self.initial_scale,
                name=f"film_block_{block_idx}",
            )(x, z)

        head_init = nn.initializers.variance_scaling(
            self.initial_scale, mode="fan_avg", distribution="normal"
        )
        delta_position = nn.Dense(
            3,
            kernel_init=head_init,
            bias_init=nn.initializers.zeros,
            name="position_head",
        )(x)
        delta_velocity = nn.Dense(
            3,
            kernel_init=head_init,
            bias_init=nn.initializers.zeros,
            name="velocity_head",
        )(x)

        predictions = [delta_position, delta_velocity]
        if self.predict_orientation:
            delta_orientation = nn.Dense(
                3,
                kernel_init=head_init,
                bias_init=nn.initializers.zeros,
                name="orientation_head",
            )(x)
            predictions.append(delta_orientation)

        return jnp.concatenate(predictions, axis=-1)

    def initialize(self, key: jax.Array):
        """Initializes parameters using dummy state-action and latent inputs."""
        state_action = jnp.zeros((self.state_action_dim,), dtype=jnp.float32)
        z = jnp.zeros((self.latent_dim,), dtype=jnp.float32)
        return self.init(key, state_action, z)
