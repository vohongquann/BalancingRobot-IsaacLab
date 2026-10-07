"""Gain tasks (``BalanceCar-<Stage>-Gains-v0``): the network writes the three PID gains of its layer (kp, ki, kd;
``mdp/gains.py``, ``mdp/actions/gain_action.py``) and the PID of ``pid_control/`` computes the command with them. The
zero action is the tuned PID.

The layers are those of the classical cascade (``pid_control/cascade_pid.py``): a speed PID writes a pitch target, a
pitch PID writes the wheel torque. The pitch task is ``pitch_env_cfg.py`` with another action; the speed task is defined
here (:class:`SpeedLayerEnvCfg`: forward-speed command, observation, rewards). Trained bottom-up: pitch, then speed over
the frozen pitch gain network (``frozen/pitch_gains/``). The action history in the observation is that of the three
gains.
"""

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from Balance_Car_RL.car import mdp
from Balance_Car_RL.car.car_cfg import WHEEL_STALL_TORQUE_NM
from Balance_Car_RL.car.rl_control.car_env_cfg import CarEnvCfg, ObservationsCfg, PolicyCfg, RewardsCfg
from Balance_Car_RL.car.rl_control.pitch_env_cfg import (
    PITCH_TARGET_MAX_RAD,
    SPEED_GUARD_M_S,
    PitchEnvCfg,
    PitchRewardsCfg,
)

GAIN_CHANGE_WEIGHT = -0.05
"""Penalty of the change of the gains between two steps. The gains act through a PID, so this is the light weight of the
speed layer, not the small one of the wheel torques."""

SPEED_MAX_M_S = 0.4
"""Largest commanded speed of the speed layer [m/s]; the wheels can reach 0.94 m/s (no-load speed 23.6 rad/s x
0.04 m)."""


# ── Speed layer: forward-speed command, over the pitch layer ───────────────────────────────────────────────────


@configclass
class SpeedLayerCommandsCfg:
    target = mdp.ScalarCommandCfg(
        resampling_time_range=(3.0, 6.0),
        low=-SPEED_MAX_M_S,
        high=SPEED_MAX_M_S,
        zero_probability=0.2,
    )


@configclass
class SpeedLayerPolicyCfg(PolicyCfg):
    """IMU pitch and rate, forward speed from the encoders, speed target, last action."""

    speed = ObsTerm(
        func=mdp.wheel_speed_estimate,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )
    speed_target = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "target"})
    last_action = ObsTerm(func=isaac_mdp.last_action)


@configclass
class SpeedLayerRewardsCfg(RewardsCfg):
    speed = RewTerm(
        func=mdp.speed_error_exp,
        weight=2.0,
        params={"command_name": "target", "std": 0.2},
    )
    speed_error = RewTerm(
        func=mdp.speed_error_l1,
        weight=-1.0,
        params={"command_name": "target"},
    )
    upright = RewTerm(
        func=mdp.upright_exp,
        weight=0.5,
        params={"std": 0.3},
    )
    output_change = RewTerm(
        func=isaac_mdp.action_rate_l2,
        weight=GAIN_CHANGE_WEIGHT,
    )


@configclass
class SpeedLayerEnvCfg(CarEnvCfg):
    """Forward-speed command, observation and rewards of the speed layer; the action is set by the task."""

    commands: SpeedLayerCommandsCfg = SpeedLayerCommandsCfg()
    observations: ObservationsCfg = ObservationsCfg(policy=SpeedLayerPolicyCfg())
    rewards: SpeedLayerRewardsCfg = SpeedLayerRewardsCfg()


# ── The two gain tasks ──────────────────────────────────────────────────────────────────────────────────────


@configclass
class PitchGainsActionsCfg:
    """The pitch PID with the gains of the network."""

    pitch_gains = mdp.PitchGainActionCfg(asset_name="robot", torque_scale=WHEEL_STALL_TORQUE_NM)


@configclass
class VelocityGainsActionsCfg:
    """The speed PID with the gains of the network, over the pitch gain layer of the stage below."""

    speed_gains = mdp.SpeedGainActionCfg(
        asset_name="robot",
        torque_scale=WHEEL_STALL_TORQUE_NM,
        pitch_scale=PITCH_TARGET_MAX_RAD,
        speed_guard=SPEED_GUARD_M_S,
    )


@configclass
class PitchGainsRewardsCfg(PitchRewardsCfg):
    output_change = RewTerm(func=isaac_mdp.action_rate_l2, weight=GAIN_CHANGE_WEIGHT)


@configclass
class PitchGainsEnvCfg(PitchEnvCfg):
    """Pitch PID gains. Observation 8: IMU pitch and rate, wheel speeds, last 3 gains, pitch target."""

    actions: PitchGainsActionsCfg = PitchGainsActionsCfg()
    rewards: PitchGainsRewardsCfg = PitchGainsRewardsCfg()


@configclass
class VelocityGainsEnvCfg(SpeedLayerEnvCfg):
    """Speed PID gains over the frozen pitch gain network. Observation 7: IMU pitch and rate, speed, speed target, last
    3 gains."""

    actions: VelocityGainsActionsCfg = VelocityGainsActionsCfg()
