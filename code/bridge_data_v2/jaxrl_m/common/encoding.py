import flax.linen as nn
from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp
from einops import rearrange, repeat


class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    use_proprio: bool
    stop_gradient: bool

    def __call__(self, observations: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        encoding = self.encoder(observations["image"])
        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)
        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)
        return encoding


class GCEncodingWrapper(nn.Module):
    """
    Encodes observations and goals into a single flat encoding. Handles all the
    logic about when/how to combine observations and goals.

    Takes a tuple (observations, goals) as input.

    Args:
        encoder: The encoder network for observations.
        goal_encoder: The encoder to use for goals (optional). If None, early
            goal concatenation is used, i.e. the goal is concatenated to the
            observation channel-wise before passing it through the encoder.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    goal_encoder: Optional[nn.Module]
    use_proprio: bool
    stop_gradient: bool

    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
    ) -> jnp.ndarray:
        observations, goals = observations_and_goals

        if len(observations["image"].shape) == 5:
            # obs history case
            batch_size, obs_horizon = observations["image"].shape[:2]
            # fold batch_size into obs_horizon to encode each frame separately
            obs_image = rearrange(observations["image"], "B T H W C -> (B T) H W C")
            # repeat goals so that there's a goal for each frame
            goal_image = repeat(
                goals["image"], "B H W C -> (B repeat) H W C", repeat=obs_horizon
            )
        else:
            obs_image = observations["image"]
            goal_image = goals["image"]

        if self.goal_encoder is None:
            # early goal concat
            encoder_inputs = jnp.concatenate([obs_image, goal_image], axis=-1)
            encoding = self.encoder(encoder_inputs)
        else:
            # late fusion
            encoding = self.encoder(obs_image)
            goal_encoding = self.goal_encoder(goals["image"])
            encoding = jnp.concatenate([encoding, goal_encoding], axis=-1)

        if len(observations["image"].shape) == 5:
            # unfold obs_horizon from batch_size
            encoding = rearrange(
                encoding, "(B T) F -> B (T F)", B=batch_size, T=obs_horizon
            )

        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)

        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)

        return encoding
