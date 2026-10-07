"""Velocity task (``BalanceCar-Velocity-v0``), stage 1 of the RL cascade: follow a forward speed and turn rate command.

Isaac Lab's locomotion velocity task (``isaaclab_tasks/core/velocity/velocity_env_cfg.py``) for a two-wheeled car: one
policy goes from the command straight to the actuators, balancing included.

    command      (v_x, w_z): Isaac Lab's UniformVelocityCommand, v_y = 0 (the car cannot move sideways), 10 % standing
    observation  IMU pitch and rate, speed (encoders), turn rate (gyro), command, last action (8)
    action       common and turning torque, mixed into the two wheel torques (``mdp/actions/wheel_action.py``)
    reward       Isaac Lab's track_lin_vel_xy_exp and track_ang_vel_z_exp, an L1 term of each error (slope where the
                 kernels are flat), upright, Isaac Lab's flat_orientation_l2 against deep leans, torque and torque
                 change, alive / fall
    events       Isaac Lab's domain randomization at startup (wheel friction, body mass +-25 %, centre of mass), the
                 shared random start and pushes (``car_env_cfg.py``)

The turn-rate penalty of the shared rewards is off: here the turn rate is commanded. Everything else is
``car_env_cfg.py``; details: ``guide/04_training.md``.
"""

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from Balance_Car_RL.car import mdp
from Balance_Car_RL.car.car_cfg import WHEEL_STALL_TORQUE_NM
from Balance_Car_RL.car.rl_control.car_env_cfg import CarEnvCfg, EventCfg, ObservationsCfg, PolicyCfg, RewardsCfg

SPEED_MAX_M_S = 0.4
"""Largest commanded forward speed [m/s]; the wheels can reach 0.94 m/s (no-load speed 23.6 rad/s x 0.04 m)."""

YAW_RATE_MAX_RAD_S = 2.0
"""Largest commanded turn rate [rad/s]: 0.11 m/s of speed difference at each wheel (track 0.107 m)."""

TURN_SHARE = 0.2
"""Share of the wheel torque range the turning action spans: 0.04 N m of difference, plenty for a yaw this light, and
its exploration noise is five times smaller (``mdp/actions/wheel_action.py``)."""


@configclass
class VelocityCommandsCfg:
    base_velocity = isaac_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 6.0),
        rel_standing_envs=0.1,
        heading_command=False,
        debug_vis=True,
        marker_pos_offset=(0.0, 0.0, 0.2),
        ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-SPEED_MAX_M_S, SPEED_MAX_M_S),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-YAW_RATE_MAX_RAD_S, YAW_RATE_MAX_RAD_S),
        ),
    )

    def __post_init__(self):
        self.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.15, 0.05, 0.05)
        self.base_velocity.current_vel_visualizer_cfg.markers["arrow"].scale = (0.15, 0.05, 0.05)


@configclass
class VelocityPolicyCfg(PolicyCfg):
    """IMU pitch and rate, forward speed, turn rate, command (v_x, w_z), last action (common, turn): 8.

    The order is fixed: the position task rebuilds it for the frozen policy (``mdp/actions/velocity_action.py``).
    """

    speed = ObsTerm(func=mdp.wheel_speed_estimate, params={"asset_cfg": mdp.WHEELS})
    yaw_rate = ObsTerm(func=mdp.gyro_yaw_rate)
    velocity_command = ObsTerm(func=mdp.velocity_command, params={"command_name": "base_velocity"})
    last_action = ObsTerm(func=isaac_mdp.last_action)


@configclass
class VelocityActionsCfg:
    """Common and turning torque in [-1, 1]; the wheels get common -+ TURN_SHARE x turn, times the stall torque."""

    wheel_torque = mdp.WheelMixActionCfg(asset_name="robot", scale=WHEEL_STALL_TORQUE_NM, turn_share=TURN_SHARE)


@configclass
class VelocityEventCfg(EventCfg):
    """The shared resets and pushes, and Isaac Lab's startup randomization of the robot. The masses of the Balboa are
    estimates (no published weights), so the body mass gets Isaac Lab's +-25 %; its centre of mass moves a few mm (a
    real chassis is never balanced exactly over the axle)."""

    wheel_friction = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.5, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    body_mass = EventTerm(
        func=isaac_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (1 / 1.25, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    body_com = EventTerm(
        func=isaac_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.002, 0.002), "y": (-0.002, 0.002), "z": (-0.003, 0.003)},
        },
    )


@configclass
class VelocityRewardsCfg(RewardsCfg):
    track_lin_vel = RewTerm(
        func=isaac_mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_ang_vel = RewTerm(
        func=isaac_mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    speed_error = RewTerm(
        func=mdp.speed_error_l1,
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )
    yaw_rate_error = RewTerm(
        func=mdp.yaw_rate_error_l1,
        weight=-0.2,
        params={"command_name": "base_velocity"},
    )
    upright = RewTerm(
        func=mdp.upright_exp,
        weight=0.5,
        params={"std": 0.3},
    )
    lean = RewTerm(
        func=isaac_mdp.flat_orientation_l2,  # sin^2(tilt): a speed step was taken leaning 0.5 rad (falls at 0.8)
        weight=-10.0,
    )
    torque = RewTerm(
        func=isaac_mdp.action_l2,
        weight=-0.01,
    )
    output_change = RewTerm(
        func=isaac_mdp.action_rate_l2,  # five times the shared weight: the turn chattered standing still
        weight=-0.05,
    )


@configclass
class VelocityEnvCfg(CarEnvCfg):
    commands: VelocityCommandsCfg = VelocityCommandsCfg()
    actions: VelocityActionsCfg = VelocityActionsCfg()
    observations: ObservationsCfg = ObservationsCfg(policy=VelocityPolicyCfg())
    events: VelocityEventCfg = VelocityEventCfg()
    rewards: VelocityRewardsCfg = VelocityRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.yaw_rate = None  # the shared turn-rate penalty: here the turn rate is the command
