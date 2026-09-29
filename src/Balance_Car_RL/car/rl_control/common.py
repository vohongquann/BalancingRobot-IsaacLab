# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building blocks shared by the RL environments: scene, reset and push events, terminations and simulation timing.

A task file (``upright_env_cfg.py``, ``pitch_env_cfg.py``, ``velocity_env_cfg.py``) keeps only what is specific to the
task (action, observation, reward) and calls ``configure_car_sim(self)`` in its ``__post_init__``.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from Balance_Car_RL.car import mdp
from Balance_Car_RL.car.car_cfg import CAR_CFG, IMU_POS_M, IMU_QUAT_XYZW

# ── Timing ────────────────────────────────────────────────────────────────────────────────
PHYSICS_HZ = 200
"""Physics rate [Hz]."""
DECIMATION = 4
"""Physics steps per policy step: the policy runs at 50 Hz."""
EPISODE_LENGTH_S = 10.0
"""Episode length [s], 500 policy steps."""
FALL_TILT_RAD = 0.8
"""Tilt from vertical [rad] above which the episode ends as a fall."""


# ── Scene ─────────────────────────────────────────────────────────────────────────────────
@configclass
class BalanceCarSceneCfg(InteractiveSceneCfg):
    """Ground plane, light, one Balboa robot per environment and its IMU on the control board."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        offset=ImuCfg.OffsetCfg(pos=IMU_POS_M, rot=IMU_QUAT_XYZW),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# ── Events ────────────────────────────────────────────────────────────────────────────────
@configclass
class BalanceCarEventCfg:
    """Random start tilt so the policy learns to recover, and random pushes while it balances."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-math.pi, math.pi), "pitch": (-0.25, 0.25)},
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "pitch": (-0.5, 0.5)},
        },
    )

    reset_wheels = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (-0.5, 0.5)},
    )

    push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.1, 0.1)}},
    )


# ── Terminations ──────────────────────────────────────────────────────────────────────────
@configclass
class BalanceCarTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": FALL_TILT_RAD})


# ── Simulation ────────────────────────────────────────────────────────────────────────────
def configure_car_sim(cfg) -> None:
    """Set the timing and the default viewer camera of an environment config (call from ``__post_init__``)."""
    cfg.decimation = DECIMATION
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.sim.dt = 1.0 / PHYSICS_HZ
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.6, 0.6, 0.3), lookat=(0.0, 0.0, 0.05))
