"""Mirror symmetry of the RL cascade (``rl_control/agents/symmetry.py``): no simulator."""

import pytest
import torch
from tensordict import TensorDict

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import Balance_Car_RL.tasks  # noqa: F401
from Balance_Car_RL.car.mdp.actions.frozen_policy import STAGE_IO
from Balance_Car_RL.car.rl_control.agents.symmetry import (
    POSITION_OBS,
    VELOCITY_OBS,
    position_symmetry,
    velocity_symmetry,
)


def _obs(n, dim):
    return TensorDict({"policy": torch.randn(n, dim)}, batch_size=[n])


def test_layouts_have_the_size_of_the_stages():
    assert len(VELOCITY_OBS) == STAGE_IO["velocity"][0] and len(POSITION_OBS) == STAGE_IO["position"][0]


def test_velocity_mirrors():
    obs, actions = _obs(3, 8), torch.randn(3, 2)
    obs_aug, actions_aug = velocity_symmetry(obs=obs, actions=actions)
    o, a = obs["policy"], actions
    assert obs_aug["policy"].shape == (12, 8) and actions_aug.shape == (12, 2)
    assert torch.equal(obs_aug["policy"][:3], o) and torch.equal(actions_aug[:3], a)
    # front-back: everything changes sign (pitch, rates, speed, both commands, both torques)
    assert torch.equal(obs_aug["policy"][3:6], -o) and torch.equal(actions_aug[3:6], -a)
    # left-right: the turn rate, the w command and the turning torque change sign
    lr = o * torch.tensor([1, 1, 1, -1, 1, -1, 1, -1])
    assert torch.equal(obs_aug["policy"][6:9], lr) and torch.equal(actions_aug[6:9], a * torch.tensor([1, -1]))
    assert torch.equal(obs_aug["policy"][9:], -lr) and torch.equal(actions_aug[9:], -a * torch.tensor([1, -1]))


def test_mirrored_turn_reaches_the_mirrored_heading():
    """A car on its goal that must face 0.5 rad to the left turns left (w > 0). In every mirror the sign of the heading
    and of the turn must change together, or the mirrored data teach the wrong turn."""
    obs = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]])
    obs_aug, actions_aug = position_symmetry(
        obs=TensorDict({"policy": obs}, batch_size=[1]), actions=torch.tensor([[0.0, 0.8]])
    )
    assert torch.equal(torch.sign(obs_aug["policy"][:, 6]), torch.sign(actions_aug[:, 1]))


def test_position_mirrors_the_goal():
    goal = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.5, 0.2, 0.3, 0.0, 0.0]])
    obs_aug, _ = position_symmetry(obs=TensorDict({"policy": goal}, batch_size=[1]))
    x, y, h = obs_aug["policy"][:, 4], obs_aug["policy"][:, 5], obs_aug["policy"][:, 6]
    assert x.tolist() == pytest.approx([0.5, -0.5, 0.5, -0.5])  # front-back flips x
    assert y.tolist() == pytest.approx([0.2, 0.2, -0.2, -0.2])  # left-right flips y
    # the heading is where the car's front must face: a front-back mirror swaps front and back, so it flips like a turn
    assert h.tolist() == pytest.approx([0.3, -0.3, -0.3, 0.3])


def test_position_actions_mirror_like_a_velocity_command():
    actions = torch.tensor([[0.4, 0.7]])
    _, aug = position_symmetry(actions=actions)
    assert torch.equal(aug, torch.tensor([[0.4, 0.7], [-0.4, -0.7], [0.4, -0.7], [-0.4, 0.7]]))


@pytest.mark.parametrize("func,dim", [(velocity_symmetry, 8), (position_symmetry, 9)])
def test_mirroring_twice_is_the_identity(func, dim):
    obs, actions = _obs(4, dim), torch.randn(4, 2)
    obs_aug, actions_aug = func(obs=obs, actions=actions)
    for k in range(4):  # every mirror is its own inverse
        part = TensorDict({"policy": obs_aug["policy"][4 * k : 4 * k + 4]}, batch_size=[4])
        back, back_actions = func(obs=part, actions=actions_aug[4 * k : 4 * k + 4])
        assert torch.equal(back["policy"][4 * k : 4 * k + 4], obs["policy"])
        assert torch.allclose(back_actions[4 * k : 4 * k + 4], actions)


def test_only_the_rl_cascade_trains_with_symmetry():
    """The gain tasks write gains, which mirror differently; the upright task is the plain baseline."""
    with_symmetry = {"BalanceCar-Velocity-v0", "BalanceCar-Position-v0"}
    for task in (*with_symmetry, "BalanceCar-Pitch-Gains-v0", "BalanceCar-Velocity-Gains-v0", "BalanceCar-Upright-v0"):
        cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        assert (cfg.algorithm.symmetry_cfg is not None) == (task in with_symmetry), task
