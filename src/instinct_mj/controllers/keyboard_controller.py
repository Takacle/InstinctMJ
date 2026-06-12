"""Keyboard velocity controller for robot locomotion.

Provides a standalone ``KeyboardVelocityController`` that maintains velocity
command state from keyboard input, and a ``KeyboardPlayViewer`` that
replaces the environment's velocity command each step with the keyboard
input.  The environment's command-area random sampling is completely
suppressed — the policy only ever sees keyboard-driven commands.

This module works with any command term that exposes a ``vel_command_b``
tensor, including ``UniformVelocityCommand`` (locomotion) and
``PoseVelocityCommand`` (parkour).

Usage
-----
    from instinct_mj.controllers.keyboard_controller import (
        KeyboardVelocityController,
        KeyboardPlayViewer,
    )

    # Create env, vec_env, policy as usual
    ...

    kb = KeyboardVelocityController(device=device)
    viewer = KeyboardPlayViewer(viewer_env, policy, kb)
    viewer.run()

Keyboard Controls
-----------------
    ↑ / I      Forward      (lin_vel_x += delta)
    ↓ / K      Backward     (lin_vel_x -= delta)
    ← / J      Turn left    (ang_vel_z += delta)
    → / L      Turn right   (ang_vel_z -= delta)
    U          Strafe left  (lin_vel_y += delta)
    O          Strafe right (lin_vel_y -= delta)
    X          Stop (all velocities -> 0)

    Terrain (parkour tasks):
    1-9, 0     Switch terrain type (10 types)
    [          Decrease difficulty (row - 1)
    ]          Increase difficulty (row + 1)

    Built-in viewer keys:
    Enter      Reset environment
    R          Toggle debug visualization
    Space      Pause / resume
    +/-        Speed up / slow down
    , / .      Switch environment
    P          Toggle reward plots
    A          Toggle show all envs
"""

from __future__ import annotations

import numpy as np
import torch
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.keys import (
  KEY_0,
  KEY_1,
  KEY_2,
  KEY_3,
  KEY_4,
  KEY_5,
  KEY_6,
  KEY_7,
  KEY_8,
  KEY_9,
  KEY_A,
  KEY_COMMA,
  KEY_DOWN,
  KEY_ENTER,
  KEY_EQUAL,
  KEY_I,
  KEY_J,
  KEY_K,
  KEY_L,
  KEY_LEFT,
  KEY_LEFT_BRACKET,
  KEY_MINUS,
  KEY_O,
  KEY_P,
  KEY_PERIOD,
  KEY_R,
  KEY_RIGHT,
  KEY_RIGHT_BRACKET,
  KEY_SPACE,
  KEY_U,
  KEY_UP,
  KEY_X,
)

_DIGIT_KEYS = [KEY_1, KEY_2, KEY_3, KEY_4, KEY_5,
               KEY_6, KEY_7, KEY_8, KEY_9, KEY_0]


class KeyboardVelocityController:
  """Maintains velocity command state updated by keyboard callbacks.

  The controller is always active.  On creation all velocities are zero
  (the robot stands still) and only keyboard input changes them.  The
  environment's command-area random sampling is never used.
  """

  def __init__(
    self,
    device: str,
    vel_delta: float = 0.1,
    max_lin_vel: float = 1.5,
    max_ang_vel: float = 3.0,
    ang_vel_factor: float = 2.0,
  ):
    self.device = device
    self.vel_delta = vel_delta
    self.max_lin_vel = max_lin_vel
    self.max_ang_vel = max_ang_vel
    self.ang_vel_factor = ang_vel_factor
    self.lin_vel_x = 0.0
    self.lin_vel_y = 0.0
    self.ang_vel_z = 0.0
    self._changed = False
    self._cleared = False

  def handle_key(self, key: int) -> None:
    """Process a key press and update velocity state."""
    self._changed = False
    self._cleared = False

    if key in (KEY_I, KEY_UP):
      self.lin_vel_x += self.vel_delta
      self._changed = True
    elif key in (KEY_K, KEY_DOWN):
      self.lin_vel_x -= self.vel_delta
      self._changed = True
    elif key in (KEY_J, KEY_LEFT):
      self.ang_vel_z += self.vel_delta * self.ang_vel_factor
      self._changed = True
    elif key in (KEY_L, KEY_RIGHT):
      self.ang_vel_z -= self.vel_delta * self.ang_vel_factor
      self._changed = True
    elif key == KEY_U:
      self.lin_vel_y += self.vel_delta
      self._changed = True
    elif key == KEY_O:
      self.lin_vel_y -= self.vel_delta
      self._changed = True
    elif key == KEY_X:
      self.lin_vel_x = 0.0
      self.lin_vel_y = 0.0
      self.ang_vel_z = 0.0
      self._cleared = True
      return

    if not self._changed:
      return

    self.lin_vel_x = max(-self.max_lin_vel, min(self.max_lin_vel, self.lin_vel_x))
    self.lin_vel_y = max(-self.max_lin_vel, min(self.max_lin_vel, self.lin_vel_y))
    self.ang_vel_z = max(-self.max_ang_vel, min(self.max_ang_vel, self.ang_vel_z))

  def get_command(self, num_envs: int) -> torch.Tensor:
    """Return the velocity command tensor (always active)."""
    cmd = torch.zeros(num_envs, 3, device=self.device)
    cmd[:, 0] = self.lin_vel_x
    cmd[:, 1] = self.lin_vel_y
    cmd[:, 2] = self.ang_vel_z
    return cmd

  def reset(self) -> None:
    """Reset all velocities to zero."""
    self.lin_vel_x = 0.0
    self.lin_vel_y = 0.0
    self.ang_vel_z = 0.0


class KeyboardPlayViewer(NativeMujocoViewer):
  """Native viewer that replaces velocity commands with keyboard input.

  The environment's command-area random sampling is completely suppressed.
  Every frame, the keyboard velocity is written into ``vel_command_b`` both
  before policy inference and after ``env.step()``, and the observation
  cache is invalidated so the policy always sees the keyboard command.

  For parkour tasks with curriculum terrain, terrain type and difficulty
  can be switched at runtime via keyboard.
  """

  def __init__(
    self,
    env,
    policy,
    vel_controller: KeyboardVelocityController,
    command_name: str = "base_velocity",
    **kwargs,
  ):
    self._vel_controller = vel_controller
    self._command_name = command_name
    self._command_override_error_reported = False
    self._terrain_column_map: dict[int, str] = {}
    self._terrain_type_names: list[str] = []
    self._current_terrain_col: int = 0
    self._current_terrain_level: int = 0
    super().__init__(env, policy, **kwargs)
    self._validate_command_override()
    self._build_terrain_map()

  def _build_terrain_map(self) -> None:
    """Build terrain-name → column-index mapping from terrain generator cfg.

    Uses the same proportional column assignment as
    ``TerrainGenerator._generate_curriculum_terrains`` and
    ``PoseVelocityCommand.__init__``.
    """
    terrain = self._get_terrain()
    if terrain is None:
      return
    gen_cfg = getattr(terrain.cfg, "terrain_generator", None)
    if gen_cfg is None:
      return
    sub_terrains = getattr(gen_cfg, "sub_terrains", None)
    if not sub_terrains:
      return

    proportions = np.array(
      [cfg.proportion for cfg in sub_terrains.values()]
    )
    proportions = proportions / proportions.sum()
    num_cols = gen_cfg.num_cols
    names = list(sub_terrains.keys())

    col_map: dict[int, str] = {}
    for col in range(num_cols):
      idx = int(
        np.min(
          np.where(
            col / num_cols + 0.001 < np.cumsum(proportions)
          )[0]
        )
      )
      col_map[col] = names[idx]

    self._terrain_column_map = col_map
    self._terrain_type_names = [
      col_map[c] for c in range(num_cols)
    ]
    unwrapped = self.env.unwrapped
    if hasattr(terrain, "terrain_types") and terrain.terrain_types is not None:
      self._current_terrain_col = int(terrain.terrain_types[0])
    if hasattr(terrain, "terrain_levels") and terrain.terrain_levels is not None:
      self._current_terrain_level = int(terrain.terrain_levels[0])

  def _get_terrain(self):
    """Return the terrain entity, or None if not available."""
    unwrapped = self.env.unwrapped
    scene = getattr(unwrapped, "scene", None)
    if scene is None:
      return None
    try:
      return scene["terrain"]
    except (KeyError, AttributeError):
      return None

  def _safe_key_callback(self, key: int) -> None:
    """Route keys: viewer actions first, then terrain, then velocity."""
    if key == KEY_ENTER:
      self.request_reset()
    elif key == KEY_SPACE:
      self.request_toggle_pause()
    elif key == KEY_MINUS:
      self.request_speed_down()
    elif key == KEY_EQUAL:
      self.request_speed_up()
    elif key == KEY_COMMA:
      self.request_action("PREV_ENV")
    elif key == KEY_PERIOD:
      self.request_action("NEXT_ENV")
    elif key == KEY_P:
      self.request_action("TOGGLE_PLOTS")
    elif key == KEY_R:
      self.request_action("TOGGLE_DEBUG_VIS")
    elif key == KEY_A:
      self.request_action("TOGGLE_SHOW_ALL_ENVS")
    elif key in _DIGIT_KEYS:
      self._handle_terrain_switch(key)
    elif key == KEY_LEFT_BRACKET:
      self._adjust_difficulty(-1)
    elif key == KEY_RIGHT_BRACKET:
      self._adjust_difficulty(1)
    else:
      try:
        self._vel_controller.handle_key(key)
      except Exception as exc:
        self.log(f"[WARN] keyboard velocity callback raised: {exc}")
      if self._vel_controller._changed:
        c = self._vel_controller
        self.log(
          f"[KB] vel_cmd = ({c.lin_vel_x:+.2f}, {c.lin_vel_y:+.2f}, "
          f"{c.ang_vel_z:+.2f})"
        )
      elif self._vel_controller._cleared:
        self.log("[KB] Velocity command cleared (zero)")

  def _execute_step(self) -> bool:
    """Run one obs/policy/step cycle with keyboard velocity command."""
    try:
      with torch.no_grad():
        self._override_velocity_command()
        self._clear_obs_cache()
        obs = self.env.get_observations()
        actions = self.policy(obs)
        self.env.step(actions)
        self._override_velocity_command()
        self._step_count += 1
        self._stats_steps += 1
        return True
    except Exception:
      import traceback

      self._last_error = traceback.format_exc()
      self.log(f"[ERROR] Exception during step:\n{self._last_error}")
      self.pause()
      return False

  def _clear_obs_cache(self) -> None:
    """Invalidate cached observations so the next compute() recalculates."""
    unwrapped = self.env.unwrapped
    if hasattr(unwrapped, "observation_manager"):
      obs_mgr = unwrapped.observation_manager
      if hasattr(obs_mgr, "_obs_buffer"):
        obs_mgr._obs_buffer = None

  def _override_velocity_command(self) -> None:
    """Write the keyboard velocity into the command tensor and history."""
    cmd = self._vel_controller.get_command(self.env.num_envs)
    try:
      term = self._get_command_term()
      term.vel_command_b.copy_(cmd)
      self._fill_velocity_obs_history(cmd)
    except Exception as exc:
      if not self._command_override_error_reported:
        self._command_override_error_reported = True
        self.log(f"[ERROR] Velocity command override failed: {exc}")

  def _fill_velocity_obs_history(self, cmd: torch.Tensor) -> None:
    """Fill all history buffer slots for velocity command observations."""
    unwrapped = self.env.unwrapped
    obs_mgr = getattr(unwrapped, "observation_manager", None)
    if obs_mgr is None:
      return
    history_buffers = getattr(obs_mgr, "_group_obs_term_history_buffer", None)
    if history_buffers is None:
      return
    term_cfgs_map = getattr(obs_mgr, "_group_obs_term_cfgs", None)
    if term_cfgs_map is None:
      return
    term_names_map = getattr(obs_mgr, "_group_obs_term_names", None)
    if term_names_map is None:
      return

    for group_name, term_cfgs in term_cfgs_map.items():
      term_names = term_names_map[group_name]
      group_history = history_buffers.get(group_name, {})
      for idx, term_cfg in enumerate(term_cfgs):
        if term_cfg.history_length <= 0:
          continue
        term_name = term_names[idx]
        circ_buf = group_history.get(term_name)
        if circ_buf is None or not circ_buf.is_initialized:
          continue
        params = getattr(term_cfg, "params", {}) or {}
        if params.get("command_name") != self._command_name:
          continue
        for slot in range(circ_buf.max_length):
          circ_buf._buffer[slot].copy_(cmd)

  def _get_command_term(self):
    """Return the command term that provides ``vel_command_b``."""
    unwrapped = self.env.unwrapped
    if not hasattr(unwrapped, "command_manager"):
      raise AttributeError("environment has no command_manager")
    mgr = unwrapped.command_manager
    try:
      term = mgr.get_term(self._command_name)
    except KeyError as exc:
      active = getattr(mgr, "active_terms", [])
      raise KeyError(
        f"command '{self._command_name}' not found; active_terms={active}"
      ) from exc
    if not hasattr(term, "vel_command_b"):
      raise AttributeError(
        f"command term '{self._command_name}' ({type(term).__name__}) "
        f"has no vel_command_b tensor"
      )
    return term

  def _validate_command_override(self) -> None:
    """Fail early if the command term is unreachable or lacks vel_command_b."""
    self._get_command_term()

  def _handle_terrain_switch(self, key: int) -> None:
    """Switch terrain type based on digit key and reset the environment."""
    if not self._terrain_type_names:
      self.log("[WARN] No terrain types available for this task")
      return
    idx = _DIGIT_KEYS.index(key)
    if idx >= len(self._terrain_type_names):
      self.log(f"[WARN] No terrain mapped to key {idx + 1}")
      return
    self._switch_terrain(idx)

  def _switch_terrain(self, col: int) -> None:
    """Move env 0 to the terrain at the given column and reset."""
    terrain = self._get_terrain()
    if terrain is None or terrain.terrain_origins is None:
      self.log("[WARN] Terrain not available")
      return

    num_rows = terrain.terrain_origins.shape[0]
    num_cols = terrain.terrain_origins.shape[1]
    if col < 0 or col >= num_cols:
      self.log(f"[WARN] Column {col} out of range [0, {num_cols})")
      return

    level = max(0, min(self._current_terrain_level, num_rows - 1))
    env_ids = torch.tensor([0], device=terrain.terrain_types.device)

    terrain.terrain_types[env_ids] = col
    terrain.terrain_levels[env_ids] = level
    terrain.env_origins[env_ids] = terrain.terrain_origins[level, col]

    self._current_terrain_col = col
    self._current_terrain_level = level

    name = self._terrain_column_map.get(col, f"col_{col}")
    self.log(
      f"[Terrain] Switching to '{name}' (col={col}, difficulty={level})"
    )
    self.reset_environment()
    self._sync_terrain_state()

  def _adjust_difficulty(self, delta: int) -> None:
    """Change terrain difficulty by *delta* rows and reset the environment."""
    terrain = self._get_terrain()
    if terrain is None or terrain.terrain_origins is None:
      self.log("[WARN] Terrain not available")
      return

    num_rows = terrain.terrain_origins.shape[0]
    new_level = max(
      0, min(self._current_terrain_level + delta, num_rows - 1)
    )
    if new_level == self._current_terrain_level:
      self.log(
        f"[Terrain] Difficulty already at "
        f"{'min' if delta < 0 else 'max'} ({new_level})"
      )
      return

    col = self._current_terrain_col
    env_ids = torch.tensor([0], device=terrain.terrain_types.device)

    terrain.terrain_levels[env_ids] = new_level
    terrain.env_origins[env_ids] = terrain.terrain_origins[new_level, col]

    self._current_terrain_level = new_level

    name = self._terrain_column_map.get(col, f"col_{col}")
    self.log(
      f"[Terrain] '{name}' difficulty {new_level} "
      f"(of {num_rows - 1})"
    )
    self.reset_environment()
    self._sync_terrain_state()

  def _sync_terrain_state(self) -> None:
    """Re-read terrain_types/levels after reset (curriculum may modify them)."""
    terrain = self._get_terrain()
    if terrain is None:
      return
    if hasattr(terrain, "terrain_types") and terrain.terrain_types is not None:
      self._current_terrain_col = int(terrain.terrain_types[0])
    if hasattr(terrain, "terrain_levels") and terrain.terrain_levels is not None:
      self._current_terrain_level = int(terrain.terrain_levels[0])
    name = self._terrain_column_map.get(
      self._current_terrain_col, "?"
    )
    self.log(
      f"[Terrain] Now at '{name}' difficulty={self._current_terrain_level}"
    )

  def reset_environment(self) -> None:
    """Reset the environment and zero the velocity controller."""
    self._vel_controller.reset()
    super().reset_environment()


def build_keyboard_play_viewer(
  env,
  policy,
  device: str,
  command_name: str = "base_velocity",
  vel_delta: float = 0.1,
  max_lin_vel: float = 1.5,
  max_ang_vel: float = 3.0,
  ang_vel_factor: float = 2.0,
  **kwargs,
) -> KeyboardPlayViewer:
  """Convenience factory: create a keyboard controller and viewer in one call."""
  ctrl = KeyboardVelocityController(
    device=device,
    vel_delta=vel_delta,
    max_lin_vel=max_lin_vel,
    max_ang_vel=max_ang_vel,
    ang_vel_factor=ang_vel_factor,
  )
  return KeyboardPlayViewer(env, policy, ctrl, command_name=command_name, **kwargs)
