"""Plain PID (proportional, integral and derivative terms, limits) and the cascaded PID balance controller: no
simulator."""

import torch

from Balance_Car_RL.car.pid_control import PID, CascadePID


def test_proportional():
    assert torch.allclose(PID(kp=2.0).update(torch.tensor([1.0, -3.0]), dt=0.1), torch.tensor([2.0, -6.0]))


def test_integral_accumulates_and_reset_clears():
    pid = PID(kp=0.0, ki=1.0)
    e = torch.tensor([1.0])
    pid.update(e, dt=0.5)
    assert torch.allclose(pid.update(e, dt=0.5), torch.tensor([1.0]))  # integral of 1 over 1 s
    pid.reset()
    assert torch.allclose(pid.update(e, dt=0.5), torch.tensor([0.5]))


def test_derivative_has_no_kick_on_first_call():
    pid = PID(kp=0.0, kd=1.0)
    assert torch.allclose(pid.update(torch.tensor([5.0]), dt=0.1), torch.tensor([0.0]))
    assert torch.allclose(pid.update(torch.tensor([5.2]), dt=0.1), torch.tensor([2.0]))  # (5.2 - 5.0) / 0.1


def test_limits():
    pid = PID(kp=10.0, ki=100.0, out_limit=1.0, int_limit=0.5)
    out = pid.update(torch.tensor([1.0]), dt=1.0)
    assert out.item() == 1.0 and pid.integral.item() == 0.5


def test_pid_recovers_from_tilt_and_kick(closed_loop):
    for z0 in ([0.25, 0, 0], [-0.25, 0, 0], [0, 0, 7.5]):
        z = closed_loop(CascadePID(), z0)
        assert abs(z[0]) < 1e-2 and abs(z[2]) < 0.5, z0


def test_pid_leans_into_the_fall():
    """A forward lean must give a forward (positive) wheel torque on both wheels."""
    action = CascadePID().act(torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert action.shape == (1, 2) and torch.all(action > 0.0) and torch.all(action <= 1.0)


def test_pid_reset_clears_the_integral():
    pid = CascadePID(ki=1.0)
    obs = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    pid.act(obs)
    assert torch.all(pid.pitch.integral > 0.0)
    pid.reset(torch.tensor([0]))
    assert pid.pitch.integral[0] == 0.0 and pid.pitch.integral[1] > 0.0
