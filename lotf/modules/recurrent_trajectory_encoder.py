from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn


class RecurrentTrajectoryEncoder(nn.Module):
    """Recurrent state-action history encoder used by VND.

    Each normalized state-action vector is projected independently through a
    dense layer and GELU activation. A GRU then aggregates the projected
    context, and a final dense layer maps its last hidden state to the latent
    dynamics code.

    This is a deterministic encoder. Distribution-level alignment of its
    outputs with a sampling prior (for example, via MMD) belongs in the
    training objective rather than in this module.

    Args:
        state_action_dim: Number of features in one state-action vector.
        context_length: Number of state-action vectors in the history window.
        latent_dim: Dimension of the output dynamics code.
        projection_dim: Width of the per-timestep input projection.
        hidden_dim: GRU hidden-state width.
        nonlinearity: Activation after the input projection.

    Input:
        A normalized context with shape ``(..., context_length,
        state_action_dim)``. Leading batch dimensions are optional.

    Output:
        A latent code with shape ``(..., latent_dim)``.

    Example:
        >>> encoder = RecurrentTrajectoryEncoder(state_action_dim=14)
        >>> params = encoder.initialize(jax.random.key(0))
        >>> context = jnp.zeros((8, 20, 14))
        >>> encoder.apply(params, context).shape
        (8, 12)
    """

    state_action_dim: int
    context_length: int = 20
    latent_dim: int = 12
    projection_dim: int = 128
    hidden_dim: int = 128
    nonlinearity: Callable = nn.gelu

    @nn.compact
    def __call__(self, context: jax.Array) -> jax.Array:
        if context.ndim < 2:
            raise ValueError(
                "Expected context shape (..., context_length, "
                f"state_action_dim), got {context.shape}"
            )
        if context.shape[-2] != self.context_length:
            raise ValueError(
                f"Expected a {self.context_length}-step context, "
                f"got {context.shape[-2]} steps"
            )
        if context.shape[-1] != self.state_action_dim:
            raise ValueError(
                f"Expected {self.state_action_dim} state-action features, "
                f"got {context.shape[-1]}"
            )

        projected_context = nn.Dense(
            self.projection_dim,
            name="input_projection",
        )(context)
        projected_context = self.nonlinearity(projected_context)

        final_hidden, _ = nn.RNN(
            nn.GRUCell(features=self.hidden_dim, name="cell"),
            return_carry=True,
            name="history_gru",
        )(projected_context)

        return nn.Dense(self.latent_dim, name="latent_projection")(
            final_hidden
        )

    def initialize(self, key: jax.Array):
        """Initializes parameters using one dummy context window."""
        context = jnp.zeros(
            (self.context_length, self.state_action_dim), dtype=jnp.float32
        )
        return self.init(key, context)
