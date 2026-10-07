"""Discrete LQR balance controller on the state ``[pitch, pitch rate, wheel speed]``.

Discretization: ``guide/03_dynamics.md``; Riccati solution: ``guide/05_controllers.md``.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from Balance_Car_RL.car.pid_control.cascade_pid import both_wheels, upright_state

from .model import PlantParams, linear_model

DEFAULT_Q = (100.0, 1.0, 0.1)
"""State weights for pitch [rad], pitch rate [rad/s] and wheel speed [rad/s]."""
DEFAULT_R = 10.0
"""Input weight on the total wheel torque [N m]."""


def design_lqr(
    params: PlantParams | None = None,
    q: tuple[float, float, float] = DEFAULT_Q,
    r: float = DEFAULT_R,
    dt: float = 0.02,
) -> np.ndarray:
    """Return the gain ``K`` (shape 3) of ``u = -K z`` for the zero-order-hold model at ``dt``."""
    a, b = linear_model(params or PlantParams.from_urdf())
    ad, bd, *_ = cont2discrete((a, b, np.eye(3), np.zeros((3, 1))), dt, method="zoh")
    qm = np.diag(q)
    rm = np.array([[r]])
    p = solve_discrete_are(ad, bd, qm, rm)
    return np.linalg.solve(rm + bd.T @ p @ bd, bd.T @ p @ ad)[0]


class LQRController:
    """Maps the policy observation to the normalized wheel action, vectorized over environments."""

    def __init__(self, gain: np.ndarray | None = None, device: str | torch.device = "cpu"):
        self.gain = torch.as_tensor(design_lqr() if gain is None else gain, dtype=torch.float32, device=device)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """``obs`` is ``[pitch, pitch_rate, wheel_vel_L, wheel_vel_R, ...]`` (N, >= 4); returns actions (N, 2)."""
        return self.step(*upright_state(obs))

    def step(self, pitch, pitch_rate, psi_dot, psi_dot_target=0.0) -> torch.Tensor:
        """Balance at the absolute wheel speed ``psi_dot_target`` [rad/s]: driving upright at a constant wheel speed is
        an equilibrium of the model too, so the regulator acts on the speed error; actions (N, 2)."""
        state = torch.stack([pitch, pitch_rate, psi_dot - psi_dot_target], dim=1)
        return both_wheels(-(state @ self.gain))
