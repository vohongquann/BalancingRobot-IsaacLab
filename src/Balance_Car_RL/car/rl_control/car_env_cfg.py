"""Shared part of the three RL tasks (``upright_env_cfg.py``, ``pitch_env_cfg.py``, ``velocity_env_cfg.py``).

    scene:        ground, light, one Balboa per environment and its IMU on the control board
    action:       one normalized torque per wheel (the velocity stage replaces it with mdp.FrozenPitchAction)
    observation:  IMU pitch and pitch rate first (mdp.ImuPitchAndRate); each task adds its terms in its env cfg
    reward:       alive, terminated, pitch and yaw rate, action rate; each task adds what it tracks
    events:       random start tilt and wheel speed, random pushes while it balances
    termination:  time out, or tilted more than ``FALL_TILT_RAD`` (a fall)

Physics runs at 200 Hz, the policy at 50 Hz, an episode lasts 10 s (500 policy steps).
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from Balance_Car_RL.car import mdp
from Balance_Car_RL.car.car_cfg import CAR_CFG, IMU_POS_M, IMU_QUAT_XYZW, WHEEL_STALL_TORQUE_NM

# ── Timing ────────────────────────────────────────────────────────────────────────────────
PHYSICS_HZ = 200
"""Physics rate [Hz]."""
DECIMATION = 4
"""Physics steps per policy step: the policy runs at 50 Hz."""
EPISODE_LENGTH_S = 10.0
"""Episode length [s], 500 policy steps."""
FALL_TILT_RAD = 0.8
"""Tilt from vertical [rad] above which the episode ends as a fall."""


@configclass
class CarSceneCfg(InteractiveSceneCfg):
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


@configclass
class ActionsCfg:
    """One normalized torque per wheel; scale (the motor stall torque) converts it to N m."""

    wheel_torque = isaac_mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*_wheel_joint"],
        scale=WHEEL_STALL_TORQUE_NM,
    )


@configclass
class PolicyCfg(ObsGroup):
    """Observation of the policy, one vector. Everything is measurable on the real car: a raw IMU (filtered to pitch)
    and the wheel encoders. Each task adds its terms after ``imu`` in its env cfg; the order is the contract of a frozen
    policy."""

    imu = ObsTerm(func=mdp.ImuPitchAndRate, params={"sensor_cfg": SceneEntityCfg("imu")})  # [pitch, pitch rate]

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Random start tilt so the policy learns to recover, and random pushes while it balances."""

    reset_base = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-math.pi, math.pi), "pitch": (-0.25, 0.25)},
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "pitch": (-0.5, 0.5)},
        },
    )
    reset_wheels = EventTerm(
        func=isaac_mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (-0.5, 0.5)},
    )
    push = EventTerm(
        func=isaac_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.1, 0.1)}},
    )


@configclass
class RewardsCfg:
    """What every task pays; each task adds the error of what it tracks (``mdp.<quantity>_error_exp``) or
    ``upright_exp``."""

    alive = RewTerm(
        func=isaac_mdp.is_alive,
        weight=1.0,
    )
    terminated = RewTerm(
        func=isaac_mdp.is_terminated,
        weight=-5.0,
    )
    pitch_rate = RewTerm(
        func=isaac_mdp.ang_vel_xy_l2,
        weight=-0.02,
    )
    yaw_rate = RewTerm(
        func=mdp.yaw_rate_l2,
        weight=-0.05,
    )
    output_change = RewTerm(
        func=isaac_mdp.action_rate_l2,
        weight=-0.01,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=isaac_mdp.time_out,
        time_out=True,
    )
    fallen = DoneTerm(
        func=isaac_mdp.bad_orientation,
        params={"limit_angle": FALL_TILT_RAD},
    )


@configclass
class CarEnvCfg(ManagerBasedRLEnvCfg):
    """Shared part. A task overrides ``observations`` and ``rewards``, and adds ``commands`` (and ``actions``)."""

    scene: CarSceneCfg = CarSceneCfg(
        num_envs=4096,
        env_spacing=1.0,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = DECIMATION
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = 1.0 / PHYSICS_HZ
        self.sim.render_interval = self.decimation
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.6, 0.6, 0.3), lookat=(0.0, 0.0, 0.05))
