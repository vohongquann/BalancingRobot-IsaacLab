"""Action terms of the car.

    ``pitch_action.py``  — ``FrozenPitchAction``: the velocity stage outputs a pitch target, the frozen pitch policy
                           turns it into wheel torques.
    ``frozen_policy.py`` — ``load_frozen``: a finished stage loaded from ``rl_control/frozen/<stage>/policy.pt``, and
                           the contract fingerprint stored in its ``meta.json``.

The upright and pitch tasks use Isaac Lab's ``JointEffortActionCfg`` (one torque per wheel, ``car_env_cfg.py``).
"""

from .frozen_policy import *  # noqa: F401, F403
from .pitch_action import *  # noqa: F401, F403
