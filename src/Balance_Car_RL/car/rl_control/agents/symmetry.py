"""Mirror symmetry of the RL cascade, for rsl_rl's symmetry augmentation (``RslRlSymmetryCfg``), like Isaac Lab's
``compute_symmetric_states`` of the ANYmal velocity task (``isaaclab_tasks/core/velocity/mdp/symmetry``).

The Balboa is (nearly) the same mirrored front to back (x -> -x) and left to right (y -> -y). In a mirror an angular
velocity flips with the axes it does not lie on twice over (a pseudovector), so:

    front-back: pitch, pitch rate, speed, turn rate, v_x and w_z commands, common and turning torque, goal x and goal
                heading all change sign
    left-right: turn rate, w_z, turning torque, goal y and goal heading change sign; pitch, speed, v_x and the common
                torque stay

The goal heading is the direction the car's front must face. A front-back mirror makes the image of the car's front its
back, so the heading becomes -heading, not the mirror image pi - heading of the goal arrow (the first position run used
that: half of its mirrored data turned the wrong way and the heading was never learned).

Without it the first trained pitch policy of the old cascade leaned forward at a command of 0 and the cars turned while
driving straight (``guide/05_controllers.md``). Each function returns the batch with the mirrored copies appended, four
copies: original, front-back, left-right, both (rsl_rl repeats the other rollout tensors to match). The observations
are the raw ones: the normalizer is inside the actor.
"""

from __future__ import annotations

import torch

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

# ── Layouts (``rl_control/velocity_env_cfg.py``, ``position_env_cfg.py``) ──────────────────────────────────────
VELOCITY_OBS = ["pitch", "pitch_rate", "speed", "yaw_rate", "v_cmd", "w_cmd", "last_common", "last_turn"]
VELOCITY_ACTIONS = ["common", "turn"]
POSITION_OBS = ["pitch", "pitch_rate", "speed", "yaw_rate", "goal_x", "goal_y", "goal_heading", "last_v", "last_w"]
POSITION_ACTIONS = ["v", "w"]

# per mirror: the entries that change sign
_FRONT_BACK_FLIPS = {"pitch", "pitch_rate", "speed", "yaw_rate", "v_cmd", "w_cmd", "last_common", "last_turn"}
_FRONT_BACK_FLIPS |= {"common", "turn", "goal_x", "goal_heading", "last_v", "last_w", "v", "w"}
_LEFT_RIGHT_FLIPS = {"yaw_rate", "w_cmd", "last_turn", "turn", "goal_y", "goal_heading", "last_w", "w"}


def _mirror(x: torch.Tensor, layout: list[str], front_back: bool, left_right: bool) -> torch.Tensor:
    out = x.clone()
    if left_right:
        for name in _LEFT_RIGHT_FLIPS & set(layout):
            out[:, layout.index(name)] *= -1.0
    if front_back:
        for name in _FRONT_BACK_FLIPS & set(layout):
            out[:, layout.index(name)] *= -1.0
    return out


_COPIES = ((False, False), (True, False), (False, True), (True, True))


def _augment(obs, actions, obs_layout: list[str], action_layout: list[str]):
    obs_aug = None
    if obs is not None:
        n = obs.batch_size[0]
        obs_aug = obs.repeat(len(_COPIES))
        for k, (fb, lr) in enumerate(_COPIES):
            obs_aug["policy"][k * n : (k + 1) * n] = _mirror(obs["policy"], obs_layout, fb, lr)
    actions_aug = None
    if actions is not None:
        actions_aug = torch.cat([_mirror(actions, action_layout, fb, lr) for fb, lr in _COPIES])
    return obs_aug, actions_aug


@torch.no_grad()
def velocity_symmetry(env=None, obs=None, actions=None):
    """Velocity task: four copies (original, front-back, left-right, both)."""
    return _augment(obs, actions, VELOCITY_OBS, VELOCITY_ACTIONS)


@torch.no_grad()
def position_symmetry(env=None, obs=None, actions=None):
    """Position task: four copies (original, front-back, left-right, both)."""
    return _augment(obs, actions, POSITION_OBS, POSITION_ACTIONS)


def symmetry_cfg(func) -> RslRlSymmetryCfg:
    """Data augmentation with the mirrored copies (Mittal et al., ICRA 2024); the mirror loss is only logged."""
    return RslRlSymmetryCfg(use_data_augmentation=True, data_augmentation_func=func)
