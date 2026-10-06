"""Car: Pololu Balboa 32U4 balancing tasks (``BalanceCar-*``).

Tasks (all in ``rl_control/``): ``BalanceCar-Upright-v0`` (baseline) and ``BalanceCar-{Pitch,Velocity}-v0`` (the frozen
cascade, one stage per task). Their MDP terms are shared in ``mdp/``. Physical constants live in ``car_cfg.py``; the
URDF is ``assets/data/balboa/balboa.urdf``. Classical baselines: ``pid_control/`` (cascaded PID), ``lqr_control/``
(LQR).
"""

from . import rl_control  # noqa: F401
from .car_cfg import *  # noqa: F401, F403
