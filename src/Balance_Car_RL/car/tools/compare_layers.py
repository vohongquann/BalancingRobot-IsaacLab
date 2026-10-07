"""Step responses of the controllers of one layer on the same test signal, plotted on the same axes.

    upright    LQR | PID (``pid_control/cascade_pid.py``) | RL (``BalanceCar-Upright-v0``, newest exported run)
    pitch      PID (``BalanceCar-Pitch-Gains-v0``, action 0) | RL gains (frozen/pitch_gains)
    velocity   LQR | PID (``pid_control/drive.py``: the balance controller at the commanded wheel speed, PI on the turn
               rate) | RL gains (frozen/velocity_gains over frozen/pitch_gains) | RL (frozen/velocity)
    turn       LQR | PID | RL (frozen/velocity): turn-rate steps
    position   LQR | PID (the go-to-goal PID of ``pid_control/go_to_goal.py`` over the classical drive) | PID + RL
               velocity (``BalanceCar-Position-Gains-v0`` with action 0) | RL gains (frozen/position_gains: the network
               writes the six gains of that PID) | RL (frozen/position); the last three drive frozen/velocity
    path       the same five following the figure 8 of the demo: top view and distance to the reference point

Test signals (fixed, the same for every controller): upright: start tilted 0.2 rad, pushes of +0.3 and -0.3 m/s; pitch
steps of ±0.05 and ±0.08 rad; speed steps of ±0.3 m/s; turn-rate steps of ±1 and ±2 rad/s standing; goals (x, y,
heading) 1 m ahead, then to the left facing left, then back behind the start facing back. 8 cars per controller start
upright and still, no random pushes, nominal robot (no mass or friction randomization), IMU noise on; the curves are the
mean of the 8.

    python src/Balance_Car_RL/car/tools/compare_layers.py                       # every layer
    python src/Balance_Car_RL/car/tools/compare_layers.py --layer position

Writes ``guide/media/compare_<layer>.png`` and ``logs/compare/<layer>.json`` (per controller: mean |error| and, per
step, overshoot, 10-90 % rise time and settling time; per goal, final distance and heading error). A controller whose
frozen policy is missing is skipped.
"""

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[4]
LAYERS = ["upright", "pitch", "velocity", "turn", "position", "path"]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--layer", default="all", choices=[*LAYERS, "all"])
parser.add_argument("--num_envs", type=int, default=8, help="cars per controller")
parser.add_argument("--methods", nargs="+", default=None, help="only these controllers (lqr pid pid_rl gains rl)")
parser.add_argument("--figures", default=str(REPO / "guide" / "media"), help="folder of compare_<layer>.png")
parser.add_argument("--metrics", default=str(REPO / "logs" / "compare"), help="folder of <layer>.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.envs import mdp as isaac_mdp  # noqa: E402

from Balance_Car_RL.car import mdp  # noqa: E402
from Balance_Car_RL.car.lqr_control import LQRController  # noqa: E402
from Balance_Car_RL.car.mdp.actions.frozen_policy import FROZEN_DIR  # noqa: E402
from Balance_Car_RL.car.pid_control import CascadePID  # noqa: E402
from Balance_Car_RL.car.rl_control.gains_env_cfg import PitchGainsEnvCfg, VelocityGainsEnvCfg  # noqa: E402
from Balance_Car_RL.car.rl_control.position_env_cfg import PositionEnvCfg  # noqa: E402
from Balance_Car_RL.car.rl_control.position_gains_env_cfg import PositionGainsEnvCfg  # noqa: E402
from Balance_Car_RL.car.rl_control.upright_env_cfg import UprightEnvCfg  # noqa: E402
from Balance_Car_RL.car.rl_control.velocity_env_cfg import VelocityEnvCfg  # noqa: E402
from Balance_Car_RL.car.tools.path_following import (  # noqa: E402
    EIGHT_PERIOD,
    PATH_END_HOLD,
    PATH_HOLD,
    path_goal,
    path_reference,
)
from Balance_Car_RL.car.tools.scripted import (  # noqa: E402
    ClassicalDrive,
    Command,
    program_value,
    scripted_cfg,
    with_drive_action,
)

torch.set_grad_enabled(False)

NAMES = {"lqr": "LQR", "pid": "PID", "pid_rl": "PID + RL velocity", "gains": "RL gains", "rl": "RL"}
COLORS = {"lqr": "#9467bd", "pid": "#4c72b0", "pid_rl": "#17becf", "gains": "#2ca02c", "rl": "#dd5555"}


def drive_cfg(cfg_class):
    """An env cfg of ``cfg_class`` whose action is the [common, turn] torque of the velocity task, for the classical
    drive controllers (``pid_control/drive.py``)."""

    return lambda: with_drive_action(cfg_class())


# layer -> controllers {name: (env cfg, frozen stage or None)}, test signal [(start s, value)], length s, measured
# quantity, unit, error band of the settling time. A controller without a stage is the zero action of a gain task (the
# tuned PID), or for "lqr" / "pid" outside the pitch layer the classical controller of that name.
SPEC = {
    "upright": {
        "methods": {"lqr": (UprightEnvCfg, None), "pid": (UprightEnvCfg, None), "rl": (UprightEnvCfg, "upright")},
        "program": [(0.0, 0.0)],
        "start_pitch": 0.2,
        "pushes": [(3.0, 0.3), (6.0, -0.3)],
        "length": 9.5,
        "label": "pitch",
        "unit": "rad",
        "band": 0.02,
    },
    "pitch": {
        "methods": {"pid": (PitchGainsEnvCfg, None), "gains": (PitchGainsEnvCfg, "pitch_gains")},
        "program": [(0.0, 0.0), (0.5, 0.05), (1.5, -0.05), (2.5, 0.0), (3.5, 0.08), (4.0, -0.08), (4.5, 0.0)],
        "length": 5.5,
        "label": "pitch",
        "unit": "rad",
        "band": 0.01,
    },
    "velocity": {
        "methods": {
            "lqr": (VelocityEnvCfg, None),
            "pid": (VelocityEnvCfg, None),
            "gains": (VelocityGainsEnvCfg, "velocity_gains"),
            "rl": (VelocityEnvCfg, "velocity"),
        },
        "program": [(0.0, 0.0), (0.5, 0.3), (3.5, 0.0), (5.5, -0.3), (8.5, 0.0)],
        "length": 10.5,
        "label": "forward speed",
        "unit": "m/s",
        "band": 0.03,
    },
    "turn": {
        "methods": {"lqr": (VelocityEnvCfg, None), "pid": (VelocityEnvCfg, None), "rl": (VelocityEnvCfg, "velocity")},
        "program": [(0.0, 0.0), (0.5, 1.0), (2.5, -1.0), (4.5, 2.0), (6.5, -2.0), (8.5, 0.0)],
        "length": 10.0,
        "label": "turn rate",
        "unit": "rad/s",
        "band": 0.1,
    },
    "position": {
        "methods": {
            "lqr": (drive_cfg(PositionEnvCfg), None),
            "pid": (drive_cfg(PositionEnvCfg), None),
            "pid_rl": (PositionGainsEnvCfg, None),
            "gains": (PositionGainsEnvCfg, "position_gains"),
            "rl": (PositionEnvCfg, "position"),
        },
        # goals (x, y, heading) relative to the start of the car, facing +x
        "program": [
            (0.0, (0.0, 0.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0)),
            (7.0, (1.0, 0.6, math.pi / 2)),
            (13.5, (0.2, -0.4, math.pi)),
            (21.0, (0.0, 0.0, 0.0)),
        ],
        "length": 28.0,
        "label": "distance to the goal",
        "unit": "m",
    },
}
SPEC["path"] = {
    **SPEC["position"],
    "program": [],
    "length": PATH_HOLD + EIGHT_PERIOD + PATH_END_HOLD,
    "label": "distance to the reference point",
}
BELOW = {"velocity_gains": ["pitch_gains"], "position": ["velocity"], "position_gains": ["velocity"]}
"""Frozen policies a network needs besides its own: the layer below it (``pid_rl`` drives frozen/velocity too)."""


def classical(layer, method):
    return method == "lqr" or (method == "pid" and layer != "pitch")


def policy_path(stage):
    """TorchScript policy of a stage: the frozen one, or for ``upright`` (nothing is trained on it, so it is never
    frozen; ``models/`` holds its ONNX) the one exported with the newest run."""
    if stage == "upright":
        runs = sorted((REPO / "logs" / "rsl_rl" / "balance_car_upright").glob("*/exported/policy.pt"))
        return runs[-1] if runs else REPO / "logs" / "rsl_rl" / "balance_car_upright" / "policy.pt"
    return FROZEN_DIR / stage / "policy.pt"


def missing(method, stage):
    """Policies a controller needs that do not exist (its own and those below it)."""
    needed = ([stage] if stage else []) + BELOW.get(stage, []) + (["velocity"] if method == "pid_rl" else [])
    return [s for s in needed if not policy_path(s).is_file()]


def make_env(layer, cfg_class):
    cfg = scripted_cfg(cfg_class(), SPEC[layer]["length"], SPEC[layer].get("start_pitch", 0.0))
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device  # --device cpu runs beside a training on the GPU
    return ManagerBasedRLEnv(cfg)


def step_metrics(t, y, program, band):
    """Overshoot [% of the step], 10-90 % rise time [s] and settling time [s] (last time outside ``band`` of the
    command) of every step of the program, on the mean response ``y``."""
    out = []
    starts = [s for s, _ in program] + [t[-1] + 1e-9]
    for i in range(1, len(program)):
        t0, value = program[i]
        before = program[i - 1][1]
        step = value - before
        if step == 0.0:
            continue
        seg = (t >= t0) & (t < starts[i + 1])
        ts, ys = t[seg] - t0, y[seg]
        sign = math.copysign(1.0, step)
        overshoot = max(0.0, float(np.max(sign * (ys - value)))) / abs(step) * 100.0
        progress = sign * (ys - before) / abs(step)
        rise = None
        if (progress >= 0.1).any() and (progress >= 0.9).any():
            rise = float(ts[np.argmax(progress >= 0.9)] - ts[np.argmax(progress >= 0.1)])
        outside = np.abs(ys - value) > max(band, 0.05 * abs(step))
        settle = float(ts[np.nonzero(outside)[0][-1]] + (ts[1] - ts[0])) if outside.any() else 0.0
        if outside[-1]:
            settle = None  # still outside the band when the next step comes
        out.append(
            {
                "t": t0,
                "from": before,
                "to": value,
                "overshoot_pct": round(overshoot, 1),
                "rise_s": None if rise is None else round(rise, 3),
                "settle_s": None if settle is None else round(settle, 3),
            }
        )
    return out


def goal_metrics(t, distance, heading, program):
    """Per goal of the position program: distance and heading error when the next goal comes, and the time it took to
    come within 5 cm."""
    out = []
    starts = [s for s, _ in program] + [t[-1] + 1e-9]
    for i in range(1, len(program)):
        seg = (t >= program[i][0]) & (t < starts[i + 1])
        inside = np.nonzero(distance[seg] < 0.05)[0]
        out.append(
            {
                "t": program[i][0],
                "goal": program[i][1],
                "final_distance_m": round(float(distance[seg][-1]), 4),
                "final_heading_error_rad": round(float(heading[seg][-1]), 3),
                "time_to_5cm_s": round(float(t[seg][inside[0]] - program[i][0]), 2) if len(inside) else None,
            }
        )
    return out


class Controller:
    """``controller(obs) -> action``; ``command`` is the (v_x, w_z) it asks of the velocity layer (2D layers)."""

    def __init__(self, layer, method, stage, env):
        self.env, self.layer = env, layer
        self.policy = torch.jit.load(str(policy_path(stage)), map_location=env.device) if stage else None
        self.classical = None
        if classical(layer, method):
            balance = LQRController(device=env.device) if method == "lqr" else CascadePID(dt=env.step_dt)
            self.classical = balance if layer == "upright" else ClassicalDrive(env, balance)

    def __call__(self, obs):
        if self.policy is not None:
            return self.policy(obs).clamp(-1.0, 1.0)
        if self.classical is None:  # the zero action of a gain task: the tuned PID
            return torch.zeros(self.env.num_envs, self.env.action_manager.total_action_dim, device=self.env.device)
        if self.layer == "upright":
            return self.classical.act(obs)
        return self.classical(obs)

    @property
    def command(self):
        if isinstance(self.classical, ClassicalDrive) and self.classical.command is not None:
            return self.classical.command
        return self.env.action_manager.get_term(self.env.action_manager.active_terms[0]).processed_actions[:, :2]


def event_metrics(t, pitch, x, spec):
    """Upright layer, per event (the start tilt, each push): largest |pitch| of the cars, time until every car stays
    within the band, how far the cars roll on average before they stop."""
    events = [(0.0, f"start tilt {spec['start_pitch']} rad")] + [(t0, f"push {v:+} m/s") for t0, v in spec["pushes"]]
    starts = [s for s, _ in events] + [t[-1] + 1e-9]
    out = []
    for i, (t0, what) in enumerate(events):
        seg = (t >= t0) & (t < starts[i + 1])
        worst = np.abs(pitch[seg]).max(axis=1)
        outside = np.nonzero(worst > spec["band"])[0]
        travel = np.abs(x[seg] - x[seg][0]).max(axis=0).mean()
        out.append(
            {
                "event": what,
                "peak_tilt_rad": round(float(worst.max()), 3),
                "settle_s": round(float(t[seg][outside[-1]] - t0 + (t[1] - t[0])), 2) if len(outside) else 0.0,
                "travel_m": round(float(travel), 3),
            }
        )
    return out


def run(layer, method, cfg_class, stage):
    """Roll out one controller on the test signal of ``layer`` (``path``: the figure 8)."""
    spec = SPEC[layer]
    env = make_env(layer, cfg_class)
    torch.manual_seed(0)
    obs = env.reset()[0]["policy"]
    robot = env.scene["robot"]
    wheels = robot.find_joints(["left_wheel_joint", "right_wheel_joint"], preserve_order=True, as_proxy=True)[0].torch
    command = Command(env) if layer != "upright" else None
    act = Controller(layer, method, stage, env)
    two_d = layer in ("position", "path")
    start = robot.data.root_pos_w.torch[:, :2].cpu().numpy()
    pushes = {int(round(t0 / env.step_dt)): v for t0, v in spec.get("pushes", [])}
    everyone = torch.arange(env.num_envs, device=env.device)
    log = {k: [] for k in ("t", "cmd", "meas", "heading", "pitch", "speed", "yaw", "torque", "xy", "inner", "ref")}
    for k in range(int(round(spec["length"] / env.step_dt))):
        t = k * env.step_dt
        if command is not None:
            value = path_goal(t) if layer == "path" else program_value(spec["program"], t)
            if command.kind == "base_velocity":  # (v_x, w_z); the gain task's command is the speed alone
                value = (0.0, value) if layer == "turn" else (value, 0.0)
            command.set(obs, value)
        if k in pushes:  # Isaac Lab's push event, the same velocity change for every car
            isaac_mdp.push_by_setting_velocity(env, everyone, {"x": (pushes[k], pushes[k])})
        obs = env.step(act(obs))[0]["policy"]
        pitch = mdp.body_pitch(env)
        speed = robot.data.root_lin_vel_b.torch[:, 0]
        yaw = robot.data.root_ang_vel_b.torch[:, 2]
        xy = robot.data.root_pos_w.torch[:, :2].cpu().numpy() - start
        if layer == "path":
            reference = path_reference(t)[0]
            log["ref"].append(reference)
            meas = np.linalg.norm(xy - reference, axis=1)
        elif layer == "position":
            meas = command.term.pos_command_b[:, :2].norm(dim=1).cpu().numpy()
        else:
            meas = {"upright": pitch, "pitch": pitch, "velocity": speed, "turn": yaw}[layer].cpu().numpy()
        log["t"].append(t)
        log["cmd"].append(0.0 if two_d else program_value(spec["program"], t))
        log["meas"].append(meas.copy())
        log["heading"].append(command.term.heading_command_b.abs().cpu().numpy() if two_d else np.zeros(env.num_envs))
        log["pitch"].append(pitch.cpu().numpy().copy())
        log["speed"].append(speed.cpu().numpy().copy())
        log["yaw"].append(yaw.cpu().numpy().copy())
        log["torque"].append(robot.data.applied_torque.torch[:, wheels].mean(dim=1).cpu().numpy().copy())
        log["xy"].append(xy)
        log["inner"].append(act.command.cpu().numpy().copy() if two_d else np.zeros((env.num_envs, 2)))
    env.close()
    r = {k: np.array(v) for k, v in log.items()}  # (time,) or (time, cars) or (time, cars, 2)
    r["error"] = float(np.abs(r["meas"] - r["cmd"][:, None]).mean())
    if layer == "path":
        r["extra"] = {
            "median_m": round(float(np.median(r["meas"])), 4),
            "worst_m": round(float(r["meas"].max()), 4),
            "last_m": round(float(r["meas"][-1].mean()), 4),
        }
    elif layer == "position":
        r["extra"] = {"goals": goal_metrics(r["t"], r["meas"].mean(1), r["heading"].mean(1), spec["program"])}
    elif layer == "upright":
        r["extra"] = {"events": event_metrics(r["t"], r["pitch"], r["xy"][:, :, 0], spec)}
    else:
        r["extra"] = {"steps": step_metrics(r["t"], r["meas"].mean(1), spec["program"], spec["band"])}
    r["max_tilt"] = float(np.abs(r["pitch"]).max())
    r["torque_change"] = float(np.abs(np.diff(r["torque"], axis=0)).mean())
    return r


def top_view(a, results, path=None, goals=()):
    if path is not None:
        a.plot(path[:, 0], path[:, 1], "k--", lw=1.2, label="path")
    for m, r in results.items():
        for car in range(r["xy"].shape[1]):
            a.plot(
                r["xy"][:, car, 0],
                r["xy"][:, car, 1],
                color=COLORS[m],
                lw=1.0,
                alpha=0.9 if car == 0 else 0.25,
                label=NAMES[m] if car == 0 else None,
            )
    for x, y, heading in goals:
        tip = (x + 0.12 * math.cos(heading), y + 0.12 * math.sin(heading))
        a.annotate("", xy=tip, xytext=(x, y), arrowprops={"arrowstyle": "->", "color": "#ff8c1a", "lw": 2})
        a.plot(x, y, "o", color="#ff8c1a", ms=6)
    a.set_aspect("equal")
    a.set_xlabel("x [m]")
    a.set_ylabel("y [m]")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)


def plot_two_d(layer, results):
    """Position and path layers: distance, speed and turn command, pitch, torque over time; top view on the right."""
    fig = plt.figure(figsize=(14, 11))
    grid = fig.add_gridspec(5, 3)
    first = next(iter(results.values()))
    t = first["t"]
    a = fig.add_subplot(grid[0, :2])
    for m, r in results.items():
        a.plot(t, r["meas"].mean(1), color=COLORS[m], lw=1.4, label=f"{NAMES[m]} (mean {r['error']:.3f} m)")
    a.set_ylabel(f"{SPEC[layer]['label']} [m]")
    for row, (key, inner, label) in enumerate((("speed", 0, "speed [m/s]"), ("yaw", 1, "turn rate [rad/s]")), 1):
        a = fig.add_subplot(grid[row, :2])
        for m, r in results.items():
            a.plot(t, r[key].mean(1), color=COLORS[m], lw=1.2, label=NAMES[m])
            a.plot(t, r["inner"][:, :, inner].mean(1), color=COLORS[m], lw=0.8, ls=":", label=f"{NAMES[m]}: command")
        a.set_ylabel(label)
    a = fig.add_subplot(grid[3, :2])
    for m, r in results.items():
        a.plot(t, r["pitch"].mean(1), color=COLORS[m], lw=1.2, label=NAMES[m])
    a.set_ylabel("pitch [rad]")
    a = fig.add_subplot(grid[4, :2])
    for m, r in results.items():
        a.plot(t, r["torque"][:, 0], color=COLORS[m], lw=0.8, label=f"{NAMES[m]} (car 1)")
    a.set_ylabel("wheel torque [N m]")
    a.set_xlabel("t [s]")
    for a in fig.axes:
        a.legend(fontsize=8, loc="upper right")
        a.grid(alpha=0.3)
    a = fig.add_subplot(grid[:, 2])
    if layer == "path":
        top_view(a, results, path=first["ref"])
        a.set_title("top view: path and tracks")
        title = "figure 8 path: the goal is the path point 0.6 s ahead, facing along the path"
    else:
        top_view(a, results, goals=[g for _, g in SPEC[layer]["program"]])
        a.set_title("top view of the tracks (goals: orange)")
        title = "position layer: goals (x, y, heading)"
    fig.suptitle(f"{title} (mean of {args.num_envs} cars, IMU noise, no pushes)")
    fig.tight_layout()
    save(fig, layer)


def plot(layer, results):
    spec = SPEC[layer]
    unit = spec["unit"]
    rows = {"pitch": ["main", "speed", "torque"], "upright": ["main", "speed", "x", "torque"]}.get(
        layer, ["main", "speed", "pitch", "torque"]
    )
    fig = plt.figure(figsize=(11, 2.3 * len(rows) + 0.4))
    grid = fig.add_gridspec(len(rows), 1)
    first = next(iter(results.values()))
    t = first["t"]
    for i, row in enumerate(rows):
        a = fig.add_subplot(grid[i, 0])
        if row == "main":
            a.plot(t, first["cmd"], "k--", lw=1.2, label="wanted")
            for m, r in results.items():
                label = f"{NAMES[m]} (mean |error| {r['error']:.3g} {unit})"
                a.plot(t, r["meas"].mean(1), color=COLORS[m], lw=1.4, label=label)
            a.set_ylabel(f"{spec['label']} [{unit}]")
        else:
            key, ylabel = {"speed": ("speed", "speed [m/s]"), "pitch": ("pitch", "pitch [rad]")}.get(row, (None, None))
            for m, r in results.items():
                if row == "x":
                    a.plot(t, r["xy"][:, :, 0].mean(1), color=COLORS[m], lw=1.2, label=NAMES[m])
                    ylabel = "x, distance rolled [m]"
                elif key:
                    a.plot(t, r[key].mean(1), color=COLORS[m], lw=1.2, label=NAMES[m])
                else:
                    a.plot(t, r["torque"][:, 0], color=COLORS[m], lw=0.8, label=f"{NAMES[m]} (car 1)")
            a.set_ylabel(ylabel or "wheel torque [N m]")
        for t0, _ in spec.get("pushes", []):
            a.axvline(t0, color="0.5", lw=0.8, ls=":")
        a.legend(fontsize=8, loc="upper right")
        a.grid(alpha=0.3)
    a.set_xlabel("t [s]")
    skipped = [NAMES[m] for m in spec["methods"] if m not in results]
    if layer == "upright":
        pushes = " and ".join(f"{v:+} m/s at {t0:g} s" for t0, v in spec["pushes"])
        what = f"upright: start tilted {spec['start_pitch']} rad, pushes of {pushes}"
        what += f" (mean of {args.num_envs} cars, IMU noise)"
    else:
        what = f"{layer} layer: step response (mean of {args.num_envs} cars, IMU noise, no pushes)"
    fig.suptitle(what + (f"; not trained yet: {', '.join(skipped)}" if skipped else ""))
    fig.tight_layout()
    save(fig, layer)


def save(fig, layer):
    path = Path(args.figures) / f"compare_{layer}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print("COMPARE saved", path, flush=True)


for layer in LAYERS if args.layer == "all" else [args.layer]:
    results, metrics = {}, {"layer": layer, "unit": SPEC[layer]["unit"], "methods": {}}
    for method, (cfg_class, stage) in SPEC[layer]["methods"].items():
        if args.methods and method not in args.methods:
            continue
        lacking = missing(method, stage)
        if lacking:
            print(f"COMPARE {layer} {method}: skipped, no frozen {lacking}", flush=True)
            metrics["methods"][method] = {"trained": False, "missing": lacking}
            continue
        r = results[method] = run(layer, method, cfg_class, stage)
        metrics["methods"][method] = {
            "trained": True,
            "mean_abs_error": round(r["error"], 4),
            "max_tilt_rad": round(r["max_tilt"], 3),
            "torque_change_per_step_nm": round(r["torque_change"], 5),
            **r["extra"],
        }
        print(f"COMPARE {layer} {method}: {json.dumps(metrics['methods'][method])}", flush=True)
    out = Path(args.metrics)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{layer}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if results:
        plot_two_d(layer, results) if layer in ("position", "path") else plot(layer, results)
app.close()
