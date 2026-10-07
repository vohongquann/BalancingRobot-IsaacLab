# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out a controller and report balance statistics (headless, no real-time pacing).

The controller is a trained RSL-RL checkpoint (default) or one of the classical baselines (``lqr``, ``pid``), all run on
the same environment, pushes and motor model. Usage (same flags as ``isaaclab play``, plus the ones below):

    python scripts/evaluate.py --task BalanceCar-Upright-v0 --num_envs 256 --checkpoint <run>/model_299.pt
    python scripts/evaluate.py --task BalanceCar-Upright-v0 --num_envs 256 --controller lqr
    python scripts/evaluate.py --task BalanceCar-Position-v0 --num_envs 256 --controller lqr   # or Velocity, pid

On the velocity and position tasks the classical controller drives the ``[common, turn]`` action of the velocity task
(``pid_control/drive.py``): LQR or PID balances at the commanded wheel speed, a PI on the gyro holds the turn rate, and
on the position task the go-to-goal PID writes that command (``tools/scripted.py`` ``ClassicalDrive``).

It reuses the private helpers of Isaac Lab's ``play_rsl_rl`` backend, so it is tied to the Isaac Lab version this
project was developed against. Exit code is 1 when a threshold is violated.
"""

import argparse
import json
import os
import pathlib
import sys

import torch

from isaaclab.app import launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.math import wrap_to_pi

from isaaclab_rl.entrypoints.backends import play_rsl_rl as backend
from isaaclab_rl.entrypoints.common import apply_env_overrides, create_isaaclab_env, startup_screen
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

from isaaclab_tasks.utils import resolve_task_config

import Balance_Car_RL.tasks  # noqa: F401  registers BalanceCar-Upright-v0
from Balance_Car_RL.car.lqr_control import LQRController, design_lqr
from Balance_Car_RL.car.mdp.rewards import body_pitch
from Balance_Car_RL.car.pid_control import CascadePID
from Balance_Car_RL.car.tools.scripted import ClassicalDrive, with_drive_action

SETTLE_S = {"pitch": 0.5, "speed": 1.5, "turn": 1.0, "distance": 5.0, "heading": 5.0}
"""Seconds after a command change before the tracking error counts as settled (reported, not used to pass): the lean is
followed within a fraction of a second, a speed step of 0.8 m/s takes about 1 s, a goal 1.4 m away about 5 s (0.4 m/s at
most, a turn toward it, and the car must slow down before it)."""

TRACK_LIMIT = {"pitch": 0.03, "speed": 0.08, "turn": 0.3, "distance": 0.10}
"""Largest median absolute tracking error that passes [rad, m/s, rad/s, m]. The median is used because resets from a
tilt of 0.25 rad, command steps and pushes give large but short errors that dominate an RMS; holding still would give a
median of 0.03 rad and 0.13 m/s. The distance to a goal is judged on the last step of each goal (the car must have
arrived), not over the drive to it."""

UNITS = {"pitch": "rad", "speed": "m/s", "turn": "rad/s", "distance": "m", "heading": "rad"}


def tracked_kinds(task: str) -> list[str]:
    """What the task's command asks for: the quantities whose error is reported (and judged, see ``TRACK_LIMIT``)."""
    if "Pitch" in task:
        return ["pitch"]
    if "Velocity-Gains" in task:
        return ["speed"]
    if "Velocity" in task:
        return ["speed", "turn"]
    if "Position" in task:
        return ["distance", "heading"]
    return []


def snapshot(env):
    """The command active while the next action is chosen, i.e. the one it is scored against after the step: read
    before ``env.step()``, which may resample it for the *following* action. Returns (what to score against, a tensor
    whose change per env marks a new command)."""
    commands = env.command_manager
    if "target" in commands.active_terms:  # pitch or speed layer of the gain cascade
        target = commands.get_command("target")[:, 0].clone()
        return target, target.unsqueeze(-1)
    if "base_velocity" in commands.active_terms:
        velocity = commands.get_command("base_velocity").clone()
        return velocity, velocity
    term = commands.get_term("pose_command")
    goal = torch.cat([term.pos_command_w[:, :2], term.heading_command_w.unsqueeze(-1)], dim=1).clone()
    return goal, goal


def measure(kind: str, env, wanted) -> tuple[torch.Tensor, torch.Tensor]:
    """``(value, command)`` of one tracked quantity after the step, both (num_envs,)."""
    robot = env.scene["robot"]
    if kind == "pitch":
        return body_pitch(env), wanted
    if kind == "speed":
        value = robot.data.root_lin_vel_b.torch[:, 0]
        return value, wanted if wanted.dim() == 1 else wanted[:, 0]
    if kind == "turn":
        return robot.data.root_ang_vel_b.torch[:, 2], wanted[:, 2]
    if kind == "distance":
        distance = (robot.data.root_pos_w.torch[:, :2] - wanted[:, :2]).norm(dim=1)
        return distance, torch.zeros_like(distance)
    heading = wrap_to_pi(robot.data.heading_w.torch - wanted[:, 2])
    return heading, torch.zeros_like(heading)


def _no_reset(_env_ids) -> None:
    """Stand-in for controllers without state to clear."""


def _split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--controller", choices=("policy", "lqr", "pid"), default="policy")
    parser.add_argument(
        "--pid_gains", type=float, nargs=4, metavar=("KP", "KI", "KD", "KV"), help="Override the PID gains."
    )
    parser.add_argument(
        "--lqr_weights",
        type=float,
        nargs=4,
        metavar=("Q_PITCH", "Q_RATE", "Q_WHEEL", "R"),
        help="Override the LQR weights (lqr_control/lqr.py DEFAULT_Q, DEFAULT_R).",
    )
    parser.add_argument("--eval_steps", type=int, default=1500, help="Policy steps to roll out (50 Hz).")
    parser.add_argument("--max_fall_rate", type=float, default=0.02, help="Allowed share of episodes ending in a fall.")
    parser.add_argument("--max_rms_pitch", type=float, default=0.10, help="Allowed RMS pitch [rad] (Upright task).")
    parser.add_argument(
        "--max_track_error",
        type=float,
        default=None,
        help="Allowed median error of the first tracked quantity (default: TRACK_LIMIT of this script).",
    )
    parser.add_argument(
        "--no_push", action="store_true", help="Disable the random pushes (tracking without disturbances)."
    )
    parser.add_argument(
        "--trace", type=int, default=0, help="Print command, speed, pitch and action of one env for N steps."
    )
    parser.add_argument("--trace_env", type=int, default=0, help="Which environment --trace prints.")
    parser.add_argument("--json", default=None, help="Also write the statistics to this JSON file.")
    return parser.parse_known_args(argv)


def main(argv: list[str]) -> int:
    own, rest = _split_args(argv)
    args_cli = backend._parse_args(rest)
    drive = own.controller != "policy" and any(k in args_cli.task for k in ("Velocity-v", "Position-v"))
    if own.controller != "policy" and not drive and "Upright" not in args_cli.task:
        sys.exit(
            f"--controller {own.controller} runs the Upright, Velocity and Position tasks; {args_cli.task} is a gain "
            "task, whose zero action already is the tuned PID."
        )
    if own.controller == "policy" and not args_cli.checkpoint:
        sys.exit(
            "--checkpoint <path to model_N.pt> is required: without it the newest run of the task is loaded, which can "
            "be one that never trained (a fresh model_0.pt falls every episode)."
        )
    installed_version = check_rsl_rl_version()
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
        with launch_simulation(env_cfg, args_cli):
            agent_cfg = backend.cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
            apply_env_overrides(args_cli, env_cfg)
            if own.no_push:
                env_cfg.events.push = None
            if drive:  # the classical drive writes the wheel torques, not the frozen velocity network
                with_drive_action(env_cfg)
            env_cfg.seed = agent_cfg.seed
            log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            checkpoint = None
            if own.controller == "policy":
                checkpoint = backend._resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root)
            env = create_isaaclab_env(
                args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg)
            )
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            device = env.unwrapped.device
            if own.controller == "policy":
                runner = create_rsl_rl_runner(env, agent_cfg)
                runner.load(checkpoint)
                policy = runner.get_inference_policy(device=device)
                reset = _no_reset
            else:
                ctrl = (
                    LQRController(
                        design_lqr(q=tuple(own.lqr_weights[:3]), r=own.lqr_weights[3]) if own.lqr_weights else None,
                        device=device,
                    )
                    if own.controller == "lqr"
                    else CascadePID(*own.pid_gains)
                    if own.pid_gains
                    else CascadePID()
                )

                if drive:
                    ctrl = ClassicalDrive(env.unwrapped, ctrl)

                def policy(obs):
                    return ctrl(obs["policy"]) if drive else ctrl.act(obs["policy"])

                reset = getattr(ctrl, "reset", _no_reset)
            screen.close()

            robot = env.unwrapped.scene["robot"]
            n = env.unwrapped.num_envs
            pitch_sq, est_sq, yaw_sq, wheel_abs, torque_abs, speed_abs, count = [0.0] * 6 + [0]
            kinds = tracked_kinds(args_cli.task)
            track = {
                k: {
                    "sq": 0.0,
                    "errors": [],
                    # the error on the last step of each command, before the next one: did the car get there in time
                    "arrivals": [],
                    "prev_error": None,
                    # per command sign: sum of command, sum of reached value, samples (a policy that only goes one way)
                    "by_sign": {"+": [0.0, 0.0, 0], "-": [0.0, 0.0, 0]},
                    "settled_sq": 0.0,
                    "settled_count": 0,
                }
                for k in kinds
            }
            since = torch.zeros(n, device=env.unwrapped.device)
            prev_key = None
            settle_steps = {k: int(SETTLE_S[k] / env.unwrapped.step_dt) for k in kinds}
            peak = torch.zeros(n, device=env.unwrapped.device)
            falls, timeouts = 0, 0
            obs = env.get_observations()
            with torch.inference_mode():
                for step in range(own.eval_steps):
                    wanted, key = snapshot(env.unwrapped) if kinds else (None, None)
                    action = policy(obs)
                    obs, _, dones, infos = env.step(action)
                    if step < own.trace and kinds:
                        e = own.trace_env
                        values = {k: measure(k, env.unwrapped, wanted) for k in kinds}
                        shown = " ".join(f"{k}={float(v[e]):+.3f}/{float(c[e]):+.3f}" for k, (v, c) in values.items())
                        acts = [round(float(a), 3) for a in action[e]]
                        pitch0 = float(obs["policy"][e, 0])
                        print(f"[TRACE] {step:4d} value/command {shown} pitch={pitch0:+.3f} action={acts}")
                    pitch = body_pitch(env.unwrapped)
                    if step >= 50:  # skip the randomized start
                        pitch_sq += float((pitch**2).mean())
                        est_sq += float(((obs["policy"][:, 0] - pitch) ** 2).mean())  # IMU estimate vs truth
                        yaw_sq += float((robot.data.root_ang_vel_b.torch[:, 2] ** 2).mean())
                        wheel_abs += float(robot.data.joint_vel.torch.abs().mean())
                        torque_abs += float(robot.data.applied_torque.torch.abs().mean())
                        speed_abs += float(robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1).mean())
                        peak = torch.maximum(peak, pitch.abs())
                        if kinds:
                            changed = (
                                torch.ones(n, dtype=torch.bool, device=key.device)
                                if prev_key is None
                                else (key != prev_key).any(dim=1)
                            )
                            since = torch.where(changed, torch.zeros_like(since), since + 1)
                            prev_key = key.clone()
                        for k in kinds:
                            t = track[k]
                            value, cmd = measure(k, env.unwrapped, wanted)
                            error = value - cmd
                            t["sq"] += float((error**2).mean())
                            t["errors"].append(error.abs())
                            if k in ("pitch", "speed", "turn"):  # a goal or a heading has no direction of its own
                                for sign, mask in (("+", cmd > 0.01), ("-", cmd < -0.01)):
                                    t["by_sign"][sign][0] += float(cmd[mask].sum())
                                    t["by_sign"][sign][1] += float(value[mask].sum())
                                    t["by_sign"][sign][2] += int(mask.sum())
                            if t["prev_error"] is not None:
                                t["arrivals"].append(t["prev_error"][changed & ~t["prev_error"].isnan()].abs())
                            # an env reset in this step is already at its new start but still scored against its old
                            # command: not an arrival
                            t["prev_error"] = torch.where(dones.bool(), torch.nan, error)
                            settled = since >= settle_steps[k]
                            t["settled_sq"] += float((error**2)[settled].sum())
                            t["settled_count"] += int(settled.sum())
                        count += 1
                    ended = dones.bool()
                    if ended.any():
                        reset(ended.nonzero().flatten())
                    to = infos.get("time_outs", torch.zeros_like(dones)).bool()
                    timeouts += int((ended & to).sum())
                    falls += int((ended & ~to).sum())
            env.close()

            episodes = max(falls + timeouts, 1)
            stats = {
                "controller": own.controller if checkpoint is None else f"policy {checkpoint}",
                "episodes finished": falls + timeouts,
                "fall rate": falls / episodes,
                "rms pitch [rad]": (pitch_sq / count) ** 0.5,
                "pitch estimate RMS error [rad]": (est_sq / count) ** 0.5,
                "RMS turn rate [rad/s]": (yaw_sq / count) ** 0.5,
                "mean |wheel speed| [rad/s]": wheel_abs / count,
                "mean |wheel torque| [N m]": torque_abs / count,
                "mean planar speed [m/s]": speed_abs / count,
                "max pitch seen [rad]": float(peak.max()),
            }
            for k in kinds:
                t, unit = track[k], UNITS[k]
                stats[f"RMS {k} tracking error [{unit}]"] = (t["sq"] / count) ** 0.5
                stats[f"settled RMS {k} error, {SETTLE_S[k]} s after a command change [{unit}]"] = (
                    t["settled_sq"] / max(t["settled_count"], 1)
                ) ** 0.5
                for sign, (cmd_sum, value_sum, m) in t["by_sign"].items() if k in ("pitch", "speed", "turn") else ():
                    stats[f"{k} command {sign}: mean command / mean reached [{unit}]"] = [
                        cmd_sum / max(m, 1),
                        value_sum / max(m, 1),
                    ]
                flat = torch.cat(t["errors"])
                last = torch.cat(t["arrivals"]) if t["arrivals"] else torch.zeros(0)
                if last.numel() == 0:  # a short run in which no command ended
                    last = torch.full((1,), float("nan"))
                stats[f"median / 90th percentile |{k} error| on the last step of a command [{unit}]"] = [
                    float(last.median()),
                    float(torch.quantile(last[:: max(1, last.numel() // 100000)], 0.9)),
                ]
                stats[f"median / 90th percentile |{k} error| [{unit}]"] = [
                    float(flat.median()),
                    float(torch.quantile(flat[:: max(1, flat.numel() // 100000)], 0.9)),
                ]
            for k, v in stats.items():
                print(f"[EVAL] {k:28s}: {v}")
            ok = stats["fall rate"] <= own.max_fall_rate
            judged = [k for k in kinds if k in TRACK_LIMIT]
            for i, k in enumerate(judged):
                limit = own.max_track_error if (own.max_track_error and i == 0) else TRACK_LIMIT[k]
                where = " on the last step of a command" if k == "distance" else ""
                ok = ok and stats[f"median / 90th percentile |{k} error|{where} [{UNITS[k]}]"][0] <= limit
            if not kinds:
                ok = ok and stats["rms pitch [rad]"] <= own.max_rms_pitch
            if own.json:
                stats["pass"] = bool(ok)
                pathlib.Path(own.json).write_text(json.dumps(stats, indent=2, default=str) + "\n")
            print(f"[EVAL] {'PASS' if ok else 'FAIL'}")
            # closing the Kit app on leaving the block terminates the process, so exit from here with the verdict
            sys.stdout.flush()
            os._exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
