"""PID gains as the output of an RL layer (the ``-Gains-`` tasks), after Drone_RL's ``mdp/gains.py``.

The network does not write the command of the layer below; it writes the three gains of the PID of its layer (kp, ki,
kd), and the PID of ``pid_control/`` computes the command with them:

    pitch layer   pitch target         -> wheel torque   (the pitch loop of ``CascadePID``; kd acts on the pitch rate)
    speed layer   forward speed target -> pitch target   (the speed loop of ``CascadePID``)

Action ``a`` in [-1, 1], one value per gain, order [kp, ki, kd]. The tuned PID is the zero action:

    gain of the tuned PID not 0:   gain = nominal * 3^a          (a = -1 .. 1: a third to three times the tuned gain)
    gain of the tuned PID is 0:    gain = maximum * max(a, 0)    (a <= 0: the term is off)

``maximum`` is ``KI_PER_KP * kp`` [1/s] for ki and ``KD_PER_KP * kp`` [s] for kd. Pure torch, no Isaac.
"""

from __future__ import annotations

import torch

from Balance_Car_RL.car.pid_control import PITCH_INT_LIMIT, PITCH_KD, PITCH_KI, PITCH_KP, SPEED_KV

FACTOR = 3.0
"""Range of a gain around the tuned one: from ``1 / FACTOR`` to ``FACTOR`` times."""
KI_PER_KP = 0.3
"""[1/s] Largest ki of a layer whose tuned PID has none, relative to its kp."""
KD_PER_KP = 0.05
"""[s] Largest kd of such a layer, relative to its kp (a derivative time of 50 ms)."""
SPEED_INT_LIMIT = 5.0
"""Limit of the integral of the speed error of the speed layer [rad]: the error is in wheel rad/s."""


class GainLayer:
    """The gains of one layer: the tuned values, and the map from the network output to gains."""

    action_dim = 3  # kp, ki, kd

    def __init__(self, name: str, kp: float, ki: float, kd: float, int_limit: float):
        self.name = name
        self.nominal = torch.tensor([kp, ki, kd])
        self.maximum = torch.tensor([0.0, KI_PER_KP * kp, KD_PER_KP * kp])
        self.int_limit = int_limit  # limit of the integral of the error (anti-windup)

    def gains(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Network output (N, 3) in [-1, 1] -> kp, ki, kd, each (N,)."""
        a = action.clamp(-1.0, 1.0)
        nominal, maximum = self.nominal.to(a.device), self.maximum.to(a.device)
        gains = torch.where(nominal > 0.0, nominal * FACTOR**a, maximum * a.clamp(min=0.0))
        return gains[:, 0], gains[:, 1], gains[:, 2]


GAIN_LAYERS: dict[str, GainLayer] = {
    "pitch": GainLayer("pitch", PITCH_KP, PITCH_KI, PITCH_KD, int_limit=PITCH_INT_LIMIT),
    "speed": GainLayer("speed", SPEED_KV, 0.0, 0.0, int_limit=SPEED_INT_LIMIT),
}
