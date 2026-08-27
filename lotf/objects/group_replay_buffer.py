"""Host-side replay buffer for grouped latent-dynamics training data."""

import jax
import jax.numpy as jnp
import numpy as np


class GroupReplayBuffer:
    """Growing replay buffer with incremental normalization statistics.

    Each entry contains one state-action context, the consecutive dynamics
    inputs associated with its shared latent, and their physical residual
    targets. Arrays are kept on the host so the accumulated deployment buffer
    does not occupy accelerator memory.
    """

    def __init__(
        self,
        *,
        context_length: int,
        dynamics_horizon: int,
        state_action_dim: int,
        residual_dim: int,
    ):
        self.context_length = context_length
        self.dynamics_horizon = dynamics_horizon
        self.state_action_dim = state_action_dim
        self.residual_dim = residual_dim

        self.context_chunks = []
        self.dynamics_input_chunks = []
        self.target_chunks = []
        self.target_mask_chunks = []
        self.cumulative_groups = np.empty((0,), dtype=np.int64)
        self.num_groups = 0

        self.sa_count = 0
        self.sa_sum = np.zeros(state_action_dim, dtype=np.float64)
        self.sa_square_sum = np.zeros(state_action_dim, dtype=np.float64)
        self.target_count = 0
        self.target_sum = np.zeros(residual_dim, dtype=np.float64)
        self.target_square_sum = np.zeros(residual_dim, dtype=np.float64)

    def append(self, contexts, dynamics_inputs, targets, target_masks=None):
        """Append fixed-size groups with optional valid-target masks."""
        contexts = np.asarray(jax.device_get(contexts), dtype=np.float32)
        dynamics_inputs = np.asarray(
            jax.device_get(dynamics_inputs), dtype=np.float32
        )
        targets = np.asarray(jax.device_get(targets), dtype=np.float32)
        if target_masks is None:
            target_masks = np.ones(targets.shape[:2], dtype=bool)
        else:
            target_masks = np.asarray(
                jax.device_get(target_masks), dtype=bool
            )

        expected_context_shape = (
            self.context_length,
            self.state_action_dim,
        )
        expected_dynamics_shape = (
            self.dynamics_horizon,
            self.state_action_dim,
        )
        expected_target_shape = (
            self.dynamics_horizon,
            self.residual_dim,
        )
        expected_mask_shape = (self.dynamics_horizon,)
        if contexts.shape[1:] != expected_context_shape:
            raise ValueError(
                f"Expected context shape (*, {expected_context_shape}), "
                f"got {contexts.shape}"
            )
        if dynamics_inputs.shape[1:] != expected_dynamics_shape:
            raise ValueError(
                f"Expected dynamics-input shape (*, {expected_dynamics_shape}), "
                f"got {dynamics_inputs.shape}"
            )
        if targets.shape[1:] != expected_target_shape:
            raise ValueError(
                f"Expected target shape (*, {expected_target_shape}), "
                f"got {targets.shape}"
            )
        if target_masks.shape[1:] != expected_mask_shape:
            raise ValueError(
                f"Expected target-mask shape (*, {expected_mask_shape}), "
                f"got {target_masks.shape}"
            )
        if not (
            contexts.shape[0]
            == dynamics_inputs.shape[0]
            == targets.shape[0]
            == target_masks.shape[0]
        ):
            raise ValueError("Every replay array must contain the same groups")
        if contexts.shape[0] == 0:
            raise ValueError("Collection round produced no valid groups")
        if not np.all(target_masks.any(axis=1)):
            raise ValueError("Every replay group needs at least one valid target")

        self.context_chunks.append(contexts)
        self.dynamics_input_chunks.append(dynamics_inputs)
        self.target_chunks.append(targets)
        self.target_mask_chunks.append(target_masks)
        self.num_groups += contexts.shape[0]
        self.cumulative_groups = np.append(
            self.cumulative_groups, self.num_groups
        )

        context_values = contexts.reshape(-1, self.state_action_dim)
        dynamics_values = dynamics_inputs[target_masks]
        self.sa_count += context_values.shape[0] + dynamics_values.shape[0]
        self.sa_sum += context_values.sum(axis=0, dtype=np.float64)
        self.sa_sum += dynamics_values.sum(axis=0, dtype=np.float64)
        self.sa_square_sum += np.square(
            context_values, dtype=np.float64
        ).sum(axis=0)
        self.sa_square_sum += np.square(
            dynamics_values, dtype=np.float64
        ).sum(axis=0)

        target_values = targets[target_masks]
        self.target_count += target_values.shape[0]
        self.target_sum += target_values.sum(axis=0, dtype=np.float64)
        self.target_square_sum += np.square(
            target_values, dtype=np.float64
        ).sum(axis=0)

    def statistics(self):
        """Return accumulated input and residual-target mean/std arrays."""
        if self.num_groups == 0:
            raise ValueError("Cannot compute statistics from an empty buffer")

        input_mean = self.sa_sum / self.sa_count
        input_variance = np.maximum(
            self.sa_square_sum / self.sa_count - np.square(input_mean),
            0.0,
        )
        target_mean = self.target_sum / self.target_count
        target_variance = np.maximum(
            self.target_square_sum / self.target_count
            - np.square(target_mean),
            0.0,
        )
        return tuple(
            jnp.asarray(value, dtype=jnp.float32)
            for value in (
                input_mean,
                np.maximum(np.sqrt(input_variance), 1e-6),
                target_mean,
                np.maximum(np.sqrt(target_variance), 1e-8),
            )
        )

    def sample(self, key: jax.Array, sample_size: int):
        """Sample groups uniformly with replacement using a JAX-derived seed."""
        if self.num_groups == 0:
            raise ValueError("Cannot sample from an empty buffer")

        seed_bits = jax.random.bits(key, shape=(), dtype=jnp.uint32)
        host_seed = int(np.asarray(jax.device_get(seed_bits)))
        rng = np.random.default_rng(host_seed)
        global_indices = rng.integers(
            0, self.num_groups, size=sample_size, endpoint=False
        )
        chunk_indices = np.searchsorted(
            self.cumulative_groups, global_indices, side="right"
        )
        previous_ends = np.concatenate(
            [
                np.array([0], dtype=np.int64),
                self.cumulative_groups[:-1],
            ]
        )
        local_indices = global_indices - previous_ends[chunk_indices]

        contexts = np.empty(
            (
                sample_size,
                self.context_length,
                self.state_action_dim,
            ),
            dtype=np.float32,
        )
        dynamics_inputs = np.empty(
            (
                sample_size,
                self.dynamics_horizon,
                self.state_action_dim,
            ),
            dtype=np.float32,
        )
        targets = np.empty(
            (
                sample_size,
                self.dynamics_horizon,
                self.residual_dim,
            ),
            dtype=np.float32,
        )
        target_masks = np.empty(
            (sample_size, self.dynamics_horizon), dtype=bool
        )
        for chunk_index in np.unique(chunk_indices):
            output_mask = chunk_indices == chunk_index
            selected = local_indices[output_mask]
            contexts[output_mask] = self.context_chunks[chunk_index][selected]
            dynamics_inputs[output_mask] = (
                self.dynamics_input_chunks[chunk_index][selected]
            )
            targets[output_mask] = self.target_chunks[chunk_index][selected]
            target_masks[output_mask] = (
                self.target_mask_chunks[chunk_index][selected]
            )

        return tuple(
            jnp.asarray(value)
            for value in (contexts, dynamics_inputs, targets, target_masks)
        )
