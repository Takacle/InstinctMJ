"""Foothold buffer for SSR Imagined Foothold Guidance.

Tracks per-foot contact events and exposes the "first future contact location"
supervision target for the foothold imagination model.

Target semantics:
  - For stance feet: target = current foot position (in base frame, xy).
  - For swing feet:  target = position of the *next* foot contact, in base
    frame xy. Because the next contact happens in the future, this module
    delays publication of the target by ``max_delay_steps`` env steps so
    that a future contact can be back-filled to past swing steps.

The buffer is a sliding window of size ``max_delay_steps``. At every env step:
  1. Current foot position (base frame xy) is appended to the window.
  2. For feet that newly make contact (``is_first_contact=True``), the
     current foot position is written into the target slots of the preceding
     ``steps_since_contact`` frames within the window.
  3. ``steps_since_contact`` is incremented for swing feet, reset to 0 on contact.
  4. The target published for the *current* env step is the oldest frame in
     the window (delayed by ``max_delay_steps``). ``target_valid`` is True
     only if a contact has been observed during the window for that foot.

This module is intentionally stateless w.r.t. the RL algorithm: it lives in
the env and exposes its target via an observation term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class FootholdBuffer:
    """Per-env sliding-window buffer for future-contact supervision targets.

    Args:
        num_envs: Number of parallel environments.
        num_feet: Number of feet tracked (typically 2 for biped).
        max_delay_steps: Window length in env steps. A contact is back-filled
          to at most this many preceding swing steps.
        device: Torch device.
    """

    def __init__(
        self,
        num_envs: int,
        num_feet: int,
        max_delay_steps: int,
        device: str,
    ):
        self.num_envs = num_envs
        self.num_feet = num_feet
        self.max_delay_steps = max_delay_steps
        self.device = device

        # Ring buffer of foot positions in base frame xy.
        # Shape: (max_delay_steps, num_envs, num_feet, 2)
        self._foot_pos_b_buf = torch.zeros(
            max_delay_steps, num_envs, num_feet, 2, device=device
        )
        # Back-filled target (future contact location in base frame xy).
        self._target_buf = torch.zeros(
            max_delay_steps, num_envs, num_feet, 2, device=device
        )
        # Validity flag: True if a contact was observed within the window for
        # this (env, foot) at the indexed step.
        self._target_valid_buf = torch.zeros(
            max_delay_steps, num_envs, num_feet, dtype=torch.bool, device=device
        )

        # Number of consecutive swing steps since the last contact, per env/foot.
        # Clipped to ``max_delay_steps - 1`` so back-fill stays inside the window.
        self._steps_since_contact = torch.zeros(
            num_envs, num_feet, dtype=torch.long, device=device
        )

        # Ring buffer write head.
        self._write_idx = 0

        # Last published target and validity (delayed by max_delay_steps).
        self._published_target = torch.zeros(
            num_envs, num_feet, 2, device=device
        )
        self._published_valid = torch.zeros(
            num_envs, num_feet, dtype=torch.bool, device=device
        )

    @torch.no_grad()
    def update(
        self,
        foot_pos_b: torch.Tensor,
        is_first_contact: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the buffer by one env step.

        Args:
            foot_pos_b: (num_envs, num_feet, 2) current foot xy position in
              base frame.
            is_first_contact: (num_envs, num_feet) bool. True where a foot
              newly establishes contact within the last env step.

        Returns:
            target: (num_envs, num_feet, 2) delayed supervision target.
            valid:  (num_envs, num_feet) bool validity flag.
        """
        cur = self._write_idx

        # 1. Append current foot position.
        self._foot_pos_b_buf[cur] = foot_pos_b

        # 2. Back-fill target for feet that just made contact.
        #    The target for the preceding ``steps_since_contact`` swing steps
        #    is the current (contact) position.
        if is_first_contact.any():
            contact_pos = foot_pos_b  # (num_envs, num_feet, 2)
            # For each env/foot with first contact, write contact_pos back into
            # the past _steps_since_contact frames plus the current frame.
            steps_back = self._steps_since_contact.clone()
            # Include the current frame in the back-fill.
            # Build (T, num_envs, num_feet) write mask:
            #   mask[t, e, f] = is_first_contact[e, f] and (cur - t) mod L <= steps_back[e, f]
            T = self.max_delay_steps
            t_indices = torch.arange(T, device=self.device)
            # Distance from current write idx going backwards for each slot.
            # offset[t] = (cur - t) mod L  -> 0 for current frame, 1 for one step back, ...
            offsets = (cur - t_indices) % T  # (T,)
            # mask shape (T, num_envs, num_feet)
            offset_ok = offsets.view(T, 1, 1) <= steps_back.unsqueeze(0)  # (T, E, F)
            contact_mask = is_first_contact.unsqueeze(0) & offset_ok  # (T, E, F)
            # Broadcast contact_pos to (T, num_envs, num_feet, 2) where mask is True.
            contact_pos_expanded = contact_pos.unsqueeze(0).expand(T, -1, -1, -1)
            self._target_buf = torch.where(
                contact_mask.unsqueeze(-1),
                contact_pos_expanded,
                self._target_buf,
            )
            self._target_valid_buf = self._target_valid_buf | contact_mask

        # 3. Update steps_since_contact: reset on contact, else increment.
        self._steps_since_contact = torch.where(
            is_first_contact,
            torch.zeros_like(self._steps_since_contact),
            self._steps_since_contact + 1,
        ).clamp_max(self.max_delay_steps - 1)

        # 4. Advance write head and publish the delayed frame.
        #    The delayed frame is the next slot to be overwritten (oldest in FIFO).
        #    Equivalent to: publish_idx = (cur + 1) mod L  (the frame that will be
        #    overwritten next step, i.e., the oldest currently in the buffer).
        publish_idx = (cur + 1) % self.max_delay_steps
        self._published_target = self._target_buf[publish_idx].clone()
        self._published_valid = self._target_valid_buf[publish_idx].clone()

        # 5. Invalidate the slot about to be overwritten next step.
        next_idx = publish_idx
        self._target_buf[next_idx].zero_()
        self._target_valid_buf[next_idx] = False

        # 6. Advance write head.
        self._write_idx = (self._write_idx + 1) % self.max_delay_steps

        return self._published_target, self._published_valid

    def get_target(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the most recently published (target, valid) pair.

        Call after :meth:`update`.
        """
        return self._published_target, self._published_valid

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset buffers for the given env ids (or all if None)."""
        if env_ids is None:
            env_ids = slice(None)
        self._foot_pos_b_buf[:, env_ids] = 0.0
        self._target_buf[:, env_ids] = 0.0
        self._target_valid_buf[:, env_ids] = False
        self._steps_since_contact[env_ids] = 0
        self._published_target[env_ids] = 0.0
        self._published_valid[env_ids] = False


# ---------------------------------------------------------------------------
# Env-attached buffer registry
# ---------------------------------------------------------------------------

_BUFFERS_ATTR = "_foothold_buffers"


def get_or_create_foothold_buffer(
    env: ManagerBasedRlEnv,
    buffer_name: str,
    num_feet: int,
    max_delay_steps: int,
) -> FootholdBuffer:
    """Get (or lazily create) a named FootholdBuffer on the env.

    Multiple terms can share a buffer by using the same ``buffer_name``.
    """
    if not hasattr(env, _BUFFERS_ATTR):
        setattr(env, _BUFFERS_ATTR, {})
    registry: dict[str, FootholdBuffer] = getattr(env, _BUFFERS_ATTR)
    if buffer_name not in registry:
        registry[buffer_name] = FootholdBuffer(
            num_envs=env.num_envs,
            num_feet=num_feet,
            max_delay_steps=max_delay_steps,
            device=env.device,
        )
    return registry[buffer_name]


def reset_foothold_buffers(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None = None
) -> None:
    """Reset all foothold buffers attached to the env (call on env reset)."""
    if not hasattr(env, _BUFFERS_ATTR):
        return
    registry: dict[str, FootholdBuffer] = getattr(env, _BUFFERS_ATTR)
    for buf in registry.values():
        buf.reset(env_ids)
