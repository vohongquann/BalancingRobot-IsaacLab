"""The speed guard of the pitch layer of the gain cascade (``gain_action.py``, and its ROS copy ``gain_cascade.py``).

A held lean is a constant acceleration, and the wheels have a top speed, so the pitch layer was never trained on a lean
that speeds an already fast car up further (``mdp/commands.py``, ``speed_guard``); the speed layer above it is not
allowed to ask for one.
"""

from __future__ import annotations

import torch


def apply_speed_guard(target: torch.Tensor, speed: torch.Tensor, guard: float) -> torch.Tensor:
    """Zero the pitch ``target`` (N, 1) of the cars faster than ``guard`` [m/s] (``speed``, (N, 1)) that it would speed
    up further: the pitch layer never trained on such a target (``mdp/commands.py``, ``speed_guard``), so it is not fed
    one."""
    out_of_distribution = (speed.abs() > guard) & (torch.sign(target) == torch.sign(speed))
    return torch.where(out_of_distribution, torch.zeros_like(target), target)
