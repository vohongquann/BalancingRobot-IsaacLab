# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out a controller and report balance statistics (headless, no real-time pacing).

The controller is a trained RSL-RL checkpoint (default) or one of the classical baselines (``lqr``, ``pid``), all run on
the same environment, pushes and motor model. Usage (same flags as ``isaaclab play``, plus the ones below):

    python scripts/evaluate.py --task BalanceCar-Upright-v0 --num_envs 256 --eval_steps 1500
    python scripts/evaluate.py --task BalanceCar-Upright-v0 --num_envs 256 --controller lqr

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
from Balance_Car_RL.car.lqr_control import LQRController
from Balance_Car_RL.car.pid_control import CascadePID

SETTLE_S = {"pitch": 0.5, "speed": 1.5}
"""Seconds after a command change before the tracking error counts as settled (reported, not used to pass): the lean is
followed within a fraction of a second, a speed step of 0.8 m/s takes about 1 s at the largest lean."""

TRACK_LIMIT = {"pitch": 0.03, "speed": 0.08}
"""Largest median absolute tracking error that passes [rad, m/s]. The median is used because resets from a tilt of
0.25 rad, command steps and pushes give large but short errors that dominate an RMS; holding still would give a
median of 0.03 rad and 0.13 m/s."""


def _no_reset(_env_ids) -> None:
    """Stand-in for controllers without state to clear."""


def _split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--controller", choices=("policy", "lqr", "pid"), default="policy")
    parser.add_argument(
        "--pid_gains", type=float, nargs=4, metavar=("KP", "KI", "KD", "KV"), help="Override the PID gains."
    )
    parser.add_argument("--eval_steps", type=int, default=1500, help="Policy steps to roll out (50 Hz).")
    parser.add_argument("--max_fall_rate", type=float, default=0.02, help="Allowed share of episodes ending in a fall.")
    parser.add_argument("--max_rms_pitch", type=float, default=0.10, help="Allowed RMS pitch [rad] (Upright task).")
    parser.add_argument(
        "--max_track_error",
        type=float,
        default=None,
        help="Allowed median tracking error: Pitch [rad] (default 0.03), Velocity [m/s] (default 0.08).",
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
    if own.controller != "policy" and ("Pitch" in args_cli.task or "Velocity" in args_cli.task):
        sys.exit(
            f"--controller {own.controller} expects the Upright observation (6 values) and action (2 wheels); "
            f"{args_cli.task} does not match that layout."
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
                    LQRController(device=device)
                    if own.controller == "lqr"
                    else CascadePID(*own.pid_gains)
                    if own.pid_gains
                    else CascadePID()
                )

                def policy(obs):
                    return ctrl.act(obs["policy"])

                reset = getattr(ctrl, "reset", _no_reset)
            screen.close()

            robot = env.unwrapped.scene["robot"]
            n = env.unwrapped.num_envs
            pitch_sq, est_sq, wheel_abs, torque_abs, speed_abs, track_sq, count = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
            errors: list[torch.Tensor] = []
            since = torch.zeros(n, device=env.unwrapped.device)
            prev_cmd = None
            settled_sq, settled_count = 0.0, 0
            has_cmd = "target" in env.unwrapped.command_manager.active_terms
            kind = "pitch" if "Pitch" in args_cli.task else "speed" if "Velocity" in args_cli.task else None
            settle_steps = int(SETTLE_S.get(kind, 1.0) / env.unwrapped.step_dt)
            peak = torch.zeros(n, device=env.unwrapped.device)
            falls, timeouts = 0, 0
            obs = env.get_observations()
            with torch.inference_mode():
                for step in range(own.eval_steps):
                    # the command active while `action` is chosen, i.e. the one it is scored against below: read it
                    # now, since env.step() may resample it for the *next* action before this one is scored
                    cmd_active = env.unwrapped.command_manager.get_command("target")[:, 0].clone() if has_cmd else None
                    action = policy(obs)
                    obs, _, dones, infos = env.step(action)
                    if step < own.trace and has_cmd:
                        e = own.trace_env
                        cmd0 = float(env.unwrapped.command_manager.get_command("target")[e, 0])
                        v0 = float(robot.data.root_lin_vel_b.torch[e, 0])
                        acts = [round(float(a), 3) for a in action[e]]
                        pitch0 = float(obs["policy"][e, 0])
                        print(f"[TRACE] {step:4d} cmd={cmd0:+.3f} v={v0:+.3f} pitch={pitch0:+.3f} action={acts}")
                    g = robot.data.projected_gravity_b.torch
                    pitch = torch.atan2(g[:, 0], -g[:, 2])
                    if step >= 50:  # skip the randomized start
                        pitch_sq += float((pitch**2).mean())
                        est_sq += float(((obs["policy"][:, 0] - pitch) ** 2).mean())  # IMU estimate vs truth
                        wheel_abs += float(robot.data.joint_vel.torch.abs().mean())
                        torque_abs += float(robot.data.applied_torque.torch.abs().mean())
                        speed_abs += float(robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1).mean())
                        peak = torch.maximum(peak, pitch.abs())
                        if has_cmd and kind:
                            cmd = cmd_active
                            value = pitch if kind == "pitch" else robot.data.root_lin_vel_b.torch[:, 0]
                            track_sq += float(((value - cmd) ** 2).mean())
                            errors.append((value - cmd).abs())
                            changed = torch.ones_like(cmd, dtype=torch.bool) if prev_cmd is None else cmd != prev_cmd
                            since = torch.where(changed, torch.zeros_like(since), since + 1)
                            prev_cmd = cmd.clone()
                            settled = since >= settle_steps
                            settled_sq += float(((value - cmd) ** 2)[settled].sum())
                            settled_count += int(settled.sum())
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
                "mean |wheel speed| [rad/s]": wheel_abs / count,
                "mean |wheel torque| [N m]": torque_abs / count,
                "mean planar speed [m/s]": speed_abs / count,
                "max pitch seen [rad]": float(peak.max()),
            }
            if has_cmd and kind:
                unit = "rad" if kind == "pitch" else "m/s"
                stats[f"RMS {kind} tracking error [{unit}]"] = (track_sq / count) ** 0.5
                stats[f"settled RMS {kind} error, {SETTLE_S[kind]} s after a command change [{unit}]"] = (
                    settled_sq / max(settled_count, 1)
                ) ** 0.5
                flat = torch.cat(errors)
                stats[f"median / 90th percentile |{kind} error| [{unit}]"] = [
                    float(flat.median()),
                    float(torch.quantile(flat[:: max(1, flat.numel() // 100000)], 0.9)),
                ]
            for k, v in stats.items():
                print(f"[EVAL] {k:28s}: {v}")
            if has_cmd and kind:
                limit = own.max_track_error or TRACK_LIMIT[kind]
                key = next(k for k in stats if k.startswith("median"))
                ok = stats["fall rate"] <= own.max_fall_rate and stats[key][0] <= limit
            else:
                ok = stats["fall rate"] <= own.max_fall_rate and stats["rms pitch [rad]"] <= own.max_rms_pitch
            if own.json:
                stats["pass"] = bool(ok)
                pathlib.Path(own.json).write_text(json.dumps(stats, indent=2, default=str) + "\n")
            print(f"[EVAL] {'PASS' if ok else 'FAIL'}")
            # closing the Kit app on leaving the block terminates the process, so exit from here with the verdict
            sys.stdout.flush()
            os._exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
