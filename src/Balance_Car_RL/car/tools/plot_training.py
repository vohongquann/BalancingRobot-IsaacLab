"""Plot reward and episode length of a training run from its TensorBoard events.

Usage: python src/Balance_Car_RL/car/tools/plot_training.py RUN_DIR [--out guide/media/training_curve.png]
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = pathlib.Path(__file__).resolve().parents[4]
TAGS = {"Train/mean_reward": "Mean episode return", "Train/mean_episode_length": "Mean episode length (steps)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=pathlib.Path, help="Run folder with the TensorBoard events (logs/rsl_rl/<experiment>/<run>)."
    )
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "guide/media/training_curve.png")
    parser.add_argument("--title", default=None, help="Figure title (default: the run's experiment name).")
    args = parser.parse_args()
    acc = EventAccumulator(str(args.run))
    acc.Reload()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (tag, title) in zip(axes, TAGS.items(), strict=True):
        pts = acc.Scalars(tag)
        ax.plot([p.step for p in pts], [p.value for p in pts])
        ax.set(title=title, xlabel="iteration")
        ax.grid(alpha=0.3)
    fig.suptitle(args.title or args.run.parent.name)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"{args.run.name} -> {args.out}")


if __name__ == "__main__":
    main()
