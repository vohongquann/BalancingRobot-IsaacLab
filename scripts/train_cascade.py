# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the cascade stage by stage, freezing each stage's weights before the next stage uses them.

    python scripts/train_cascade.py                       # pitch, then velocity
    python scripts/train_cascade.py --stages pitch        # one stage (velocity needs pitch frozen first)
    python scripts/train_cascade.py --dry-run             # print the plan only

For every stage the script does the same five things, in this order, and stops at the first failure:

1. train with ``isaaclab train`` while recording video clips of the training itself;
2. play the last checkpoint with ``isaaclab play``: records the demo video and exports ``policy.pt`` / ``policy.onnx``;
3. evaluate it headless (``scripts/evaluate.py``) against the stage's pass criteria;
4. freeze it: copy the exported policy to ``src/Balance_Car_RL/car/rl_control/frozen/<stage>/`` with a ``meta.json``;
5. copy the videos (mp4 and gif) and the training curve to ``docs/media/``.

A stage that fails its evaluation is not frozen, so a later stage never builds on a bad policy. ``isaaclab train``
returns exit code 0 even when it crashes; success means ``Training time`` in its output.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys

from Balance_Car_RL.car.mdp.actions import contract_fingerprint  # editable install, like scripts/evaluate.py

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT / "src/Balance_Car_RL/car/rl_control/frozen"
MEDIA = ROOT / "docs/media"
LOGS = ROOT / "logs/rsl_rl"
PLOT = ROOT / "src/Balance_Car_RL/car/tools/plot_training.py"


@dataclasses.dataclass(frozen=True)
class Stage:
    name: str
    task: str
    experiment: str
    needs: tuple[str, ...] = ()
    """Stages whose frozen policy this stage loads."""


STAGES = {
    "pitch": Stage("pitch", "BalanceCar-Pitch-v0", "balance_car_pitch"),
    "velocity": Stage("velocity", "BalanceCar-Velocity-v0", "balance_car_velocity", needs=("pitch",)),
}
DEFAULT_ORDER = ("pitch", "velocity")


def run(cmd: list[str], log: pathlib.Path | None = None, timeout: float | None = None) -> str:
    """Run a command in the project root, echo it, and return its output (also written to ``log``).

    With ``timeout`` the command is stopped after that many seconds; the output so far is returned (used for ``isaaclab
    play`` without video, which plays until interrupted once the export is done).
    """
    print("+", " ".join(str(c) for c in cmd), flush=True)
    with subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
    ) as proc:
        try:
            output, _ = proc.communicate(timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # the whole group: Kit starts helper processes that keep the pipe open
            output, _ = proc.communicate()
            code = 0
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(output)
    return output if code == 0 else output + f"\n[exit code {code}]"


def retry(step, done, attempts: int = 3) -> None:
    """Run ``step()`` until ``done()`` is true. Kit sometimes crashes while starting (exit code 139), a rerun works."""
    for attempt in range(1, attempts + 1):
        step()
        if done():
            return
        print(
            f"  attempt {attempt}/{attempts} did not produce its result, retrying"
            if attempt < attempts
            else "  giving up"
        )


def isaaclab() -> str:
    exe = shutil.which("isaaclab")
    if exe is None:
        sys.exit("isaaclab not found: activate the conda environment (conda activate env_isaaclab).")
    return exe


def newest_run(experiment: str, after: float) -> pathlib.Path:
    runs = [p for p in (LOGS / experiment).glob("*/") if p.stat().st_mtime >= after]
    if not runs:
        sys.exit(f"No new run in {LOGS / experiment}: training did not start.")
    return max(runs, key=lambda p: p.stat().st_mtime)


def last_checkpoint(run_dir: pathlib.Path) -> pathlib.Path:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        sys.exit(f"No checkpoint in {run_dir}.")
    return ckpts[-1]


def to_gif(mp4: pathlib.Path, gif: pathlib.Path, seconds: float = 6.0, fps: int = 10, width: int = 360) -> None:
    """Small gif of the first seconds of a clip: the pre-commit hook of this repo rejects files above 2000 kB."""
    from moviepy.editor import VideoFileClip

    clip = VideoFileClip(str(mp4))
    clip.subclip(0, min(seconds, clip.duration)).resize(width=width).write_gif(str(gif), fps=fps, logger=None)
    clip.close()


def onnx_single_file(src: pathlib.Path, dst: pathlib.Path) -> None:
    import onnx

    onnx.save_model(onnx.load(str(src), load_external_data=True), str(dst), save_as_external_data=False)


def do_stage(stage: Stage, args: argparse.Namespace) -> None:
    for needed in stage.needs:
        if not (FROZEN / needed / "policy.pt").is_file():
            sys.exit(f"Stage '{stage.name}' needs the frozen '{needed}' policy: run --stages {needed} first.")
        needed_meta = FROZEN / needed / "meta.json"
        expected = json.loads(needed_meta.read_text()).get("contract_sha256") if needed_meta.is_file() else None
        if expected and expected != contract_fingerprint(needed):
            print(
                f"warning: frozen '{needed}' policy looks stale (its observation/action contract changed since it "
                f"was frozen); '{stage.name}' would train on top of it anyway. Consider `--stages {needed} "
                f"{stage.name}` to refreeze '{needed}' first."
            )
    print(f"\n===== stage '{stage.name}' ({stage.task}) =====", flush=True)
    logdir = ROOT / "logs/cascade"
    video = ["--video", "--video_length", str(args.video_length), "--video_interval", str(args.video_interval)]

    # 1. train (with video), or reuse the newest run
    if args.reuse_latest:
        run_dir = newest_run(stage.experiment, after=0.0)
        ckpt = last_checkpoint(run_dir)
    else:
        started = datetime.datetime.now().timestamp() - 1.0
        train_cmd = [
            isaaclab(),
            "train",
            "--rl_library",
            "rsl_rl",
            "--task",
            stage.task,
            "--num_envs",
            str(args.num_envs),
        ]
        if args.iterations:
            train_cmd += ["--max_iterations", str(args.iterations)]
        result: dict[str, str] = {}
        retry(
            lambda: result.update(
                out=run(train_cmd + ([] if args.no_video else video), logdir / f"{stage.name}_train.log")
            ),
            lambda: "Training time" in result.get("out", ""),
        )
        if "Training time" not in result.get("out", ""):
            sys.exit(f"Training of '{stage.name}' failed, see {logdir / f'{stage.name}_train.log'}")
        run_dir = newest_run(stage.experiment, started)
        ckpt = last_checkpoint(run_dir)
    print(f"trained: {run_dir.name}, checkpoint {ckpt.name}")

    # 2. play: demo video and export
    play_cmd = [isaaclab(), "play", "--rl_library", "rsl_rl", "--task", stage.task, "--num_envs", "4"]
    play_cmd += ["--checkpoint", str(ckpt)]
    # with video the playback ends with the clip; without video it never ends, so it is stopped after the export
    if not args.no_video:
        play_cmd += ["--video", "--video_length", "500"]
    exported = run_dir / "exported"
    # a video clip ends the process on its own well before this; --no-video plays until interrupted, so it needs the
    # short timeout. Both get a bound: a genuine Kit hang must not block the cascade forever (L6).
    play_timeout = 150 if args.no_video else 900
    retry(
        lambda: run(play_cmd, logdir / f"{stage.name}_play.log", timeout=play_timeout),
        lambda: (exported / "policy.pt").is_file(),
    )
    if not (exported / "policy.pt").is_file():
        sys.exit(f"Export of '{stage.name}' failed, see {logdir / f'{stage.name}_play.log'}")

    # 3. evaluate against the pass criteria
    eval_json = run_dir / "eval.json"
    eval_json.unlink(missing_ok=True)  # a stale file (e.g. from a previous --reuse-latest) must not look like a pass
    eval_cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--task",
        stage.task,
        "--num_envs",
        "256",
        "--checkpoint",
        str(ckpt),
        "--json",
        str(eval_json),
    ]
    retry(lambda: run(eval_cmd, logdir / f"{stage.name}_eval.log"), eval_json.is_file)
    if not eval_json.is_file():
        sys.exit(f"Evaluation of '{stage.name}' did not finish, see {logdir / f'{stage.name}_eval.log'}")
    stats = json.loads(eval_json.read_text())
    stats["controller"] = f"policy {ckpt.relative_to(ROOT)}"  # no absolute paths in meta.json
    print(json.dumps({k: v for k, v in stats.items() if k != "controller"}, indent=2))
    if not stats["pass"]:
        sys.exit(f"Stage '{stage.name}' failed its evaluation: NOT frozen, later stages are not run.")

    # 4. freeze
    target = FROZEN / stage.name
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported / "policy.pt", target / "policy.pt")
    onnx_single_file(exported / "policy.onnx", target / "policy.onnx")
    meta = {
        "stage": stage.name,
        "task": stage.task,
        "run": str(run_dir.relative_to(ROOT)),
        "checkpoint": ckpt.name,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "needs": list(stage.needs),
        "evaluation": stats,
    }
    fingerprint = contract_fingerprint(stage.name)
    if fingerprint:  # only stages with a known contract (mdp/actions.py::_CONTRACT_FILES) get one
        meta["contract_sha256"] = fingerprint
    (target / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"frozen: {target.relative_to(ROOT)}")

    # 5. media
    MEDIA.mkdir(parents=True, exist_ok=True)
    run(
        [sys.executable, str(PLOT), str(run_dir), "--out", str(MEDIA / f"{stage.name}_training_curve.png")],
        logdir / f"{stage.name}_plot.log",
    )
    if not args.no_video:
        for kind, sub in (("train", "train"), ("demo", "play")):
            clips = sorted((run_dir / "videos" / sub).glob("*.mp4"))
            if not clips:
                print(f"warning: no {kind} video found in {run_dir / 'videos' / sub}")
                continue
            clip = clips[-1]
            shutil.copy2(clip, MEDIA / f"{stage.name}_{kind}.mp4")
            to_gif(clip, MEDIA / f"{stage.name}_{kind}.gif")
        print(f"media: {MEDIA.relative_to(ROOT)}/{stage.name}_*")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(DEFAULT_ORDER))
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=None, help="Override the PPO iterations of every stage.")
    parser.add_argument("--video_length", type=int, default=300, help="Env steps per training clip (50 steps = 1 s).")
    parser.add_argument("--video_interval", type=int, default=2400, help="Env steps between training clips.")
    parser.add_argument("--no-video", action="store_true", help="Skip video recording (faster, for smoke tests).")
    parser.add_argument(
        "--reuse-latest", action="store_true", help="Skip training: evaluate and freeze the newest run."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    args = parser.parse_args()

    order = [s for s in DEFAULT_ORDER if s in args.stages]
    print("plan:", " -> ".join(order))
    for name in order:
        stage = STAGES[name]
        print(
            f"  {name}: task {stage.task}, experiment {stage.experiment}, loads frozen {list(stage.needs) or 'nothing'}"
        )
    if args.dry_run:
        return
    for name in order:
        do_stage(STAGES[name], args)
    print("\ncascade done:", " -> ".join(order))


if __name__ == "__main__":
    main()
