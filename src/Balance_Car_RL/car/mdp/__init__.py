"""MDP terms shared by every car task (actions, commands, observations, rewards).

    actions/        FrozenVelocityAction (position task -> frozen velocity policy -> wheel torques), load_frozen,
                    PitchGainAction / SpeedGainAction (a network writes PID gains, mdp/gains.py)
    commands.py     ScalarCommand: pitch target or forward speed of the gain tasks, with the speed guard
    observations.py ImuPitchAndRate (raw IMU -> pitch and pitch rate, as on the robot), gyro_yaw_rate,
                    wheel_speed_estimate
    rewards.py      upright, pitch and speed tracking, wheel speed
    locomotion.py   velocity task (Isaac Lab locomotion): (v_x, w_z) command, turn-rate error
    navigation.py   position task (Isaac Lab navigation): 2D pose command around the car, goal and heading rewards

Isaac Lab's own terms (``joint_vel_rel``, ``last_action``, ``is_alive``, ``UniformVelocityCommandCfg`` ...) are used as
``isaac_mdp``. The task files in ``rl_control/`` only assemble these terms into environments.
"""

from .actions import *  # noqa: F401, F403
from .commands import *  # noqa: F401, F403
from .gains import *  # noqa: F401, F403
from .locomotion import *  # noqa: F401, F403
from .navigation import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
