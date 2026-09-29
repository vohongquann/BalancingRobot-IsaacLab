import pathlib

from car_bridge.policy import BalancePolicy, TORQUE_SCALE_NM
import numpy as np

MODEL = pathlib.Path(__file__).resolve().parents[4] / 'models' / 'balance_car_policy.onnx'


def test_policy_outputs_bounded_torque_and_reacts_to_lean():
    policy = BalancePolicy(str(MODEL))
    upright = policy.act(0.0, 0.0, (0.0, 0.0))
    policy.reset()
    leaning = policy.act(0.3, 1.0, (0.0, 0.0))
    assert upright.shape == (2,)
    assert np.all(np.isfinite(leaning))
    assert not np.allclose(upright, leaning)
    assert np.all(np.abs(leaning) <= TORQUE_SCALE_NM)
