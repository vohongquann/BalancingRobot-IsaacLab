"""Path following with the position layer: a figure 8 on the floor.

The position controllers drive to a 2D pose (x, y, heading in the car's frame); to follow a path they are given, at
every step, the point of the path ``PATH_LEAD`` s ahead of the reference point, facing along the path. Used by
``record_demo.py --cascade position`` and ``compare_layers.py --layer path``. No simulator needed.

The path: stand still ``PATH_HOLD`` s, a figure 8 (lemniscate of Gerono), stand still ``PATH_END_HOLD`` s.
"""

import math

import numpy as np

PATH_HOLD = 2.0  # [s] standing still before and (PATH_END_HOLD) after the figure 8
PATH_END_HOLD = 3.0
EIGHT_SIZE, EIGHT_RATE = 0.6, 0.29  # [m] half width, phase rate [rad/s]: at most 0.25 m/s, curve radius 0.6 m at least
EIGHT_PERIOD = 2.0 * math.pi / EIGHT_RATE
EIGHT_TURN = -math.pi / 4.0  # the curve starts along (1, 1): turned so the car, facing +x, starts along it
PATH_DOT_SPACING = 0.12  # [s] of program time between two dots of the drawn path
LOOKAHEAD = 2.0  # [s] a dot of the path appears this long before the reference point reaches it
PATH_LEAD = 0.6  # [s] the goal given to the controller is the point of the path this far ahead of the reference


def path_reference(t: float):
    """Reference point and its velocity (x, y) at time ``t``, relative to the start of the car."""
    phase = EIGHT_RATE * min(max(t - PATH_HOLD, 0.0), EIGHT_PERIOD)
    point = np.array([EIGHT_SIZE * math.sin(phase), 0.5 * EIGHT_SIZE * math.sin(2.0 * phase)])
    moving = PATH_HOLD <= t < PATH_HOLD + EIGHT_PERIOD
    velocity = EIGHT_RATE * np.array([EIGHT_SIZE * math.cos(phase), EIGHT_SIZE * math.cos(2.0 * phase)]) * moving
    c, s = math.cos(EIGHT_TURN), math.sin(EIGHT_TURN)
    turn = np.array([[c, -s], [s, c]])
    return turn @ point, turn @ velocity


def path_heading(t: float) -> float:
    """Direction of the path [rad] at time ``t`` (held while the reference point stands still)."""
    t = min(max(t, PATH_HOLD + 1e-3), PATH_HOLD + EIGHT_PERIOD - 1e-3)
    velocity = path_reference(t)[1]
    return math.atan2(velocity[1], velocity[0])


def path_goal(t: float):
    """Goal for the controllers at time ``t``: the path point ``PATH_LEAD`` s ahead and the path direction there,
    ``(x, y, heading)`` relative to the start of the car."""
    point = path_reference(t + PATH_LEAD)[0]
    return point[0], point[1], path_heading(t + PATH_LEAD)
