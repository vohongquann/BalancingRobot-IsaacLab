"""
Observation math and ONNX inference for the balance car policy.

The observation layout must match ``ObservationsCfg`` in
``Balance_Car_RL/car/rl_control/upright_env_cfg.py``:
``[pitch, pitch_rate, left_wheel_vel, right_wheel_vel, last_action_left, last_action_right]``.
"""

import numpy as np

TORQUE_SCALE_NM = 0.20917
"""Action-to-torque scale [N m], equal to ``WHEEL_STALL_TORQUE_NM`` in ``Balance_Car_RL.car.car_cfg``
(``tests/test_car_cfg.py`` checks that the two stay in sync)."""


class BalancePolicy:
    """Wraps the exported ONNX actor (observation normalization is baked into the graph)."""

    def __init__(self, onnx_path: str):
        try:
            import onnxruntime as ort
        except ImportError as err:  # pragma: no cover - depends on the deployment machine
            raise RuntimeError(
                'onnxruntime is required to run the policy: pip install onnxruntime') from err
        self._session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self._input_name = self._session.get_inputs()[0].name
        self._last_action = np.zeros(2, dtype=np.float32)

    def act(
        self, pitch: float, pitch_rate: float, wheel_vel: tuple[float, float]
    ) -> np.ndarray:
        """Return the wheel torques [left, right] in N m for one control step.

        The action is clipped to [-1, 1] and the clipped value is fed back as ``last_action``, matching training
        (``clip_actions = 1.0`` in ``car_ppo_cfg.py``).
        """
        obs = np.array(
            [[pitch, pitch_rate, wheel_vel[0], wheel_vel[1],
              self._last_action[0], self._last_action[1]]],
            dtype=np.float32,
        )
        action = np.clip(self._session.run(None, {self._input_name: obs})[0][0], -1.0, 1.0)
        self._last_action = action.astype(np.float32)
        return action * TORQUE_SCALE_NM

    def reset(self) -> None:
        """Clear the recurrent action input, e.g. after the car has been picked up."""
        self._last_action[:] = 0.0
