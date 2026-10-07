"""Record a demo clip of a cascade for the README.

    rl        the velocity network (``frozen/velocity``): stands, drives forward, turns while driving, spins in place,
              drives backward, stops and takes a push; the command is (v_x, w_z), straight to the wheel torques
    gains     the gain cascade (``frozen/velocity_gains`` over ``frozen/pitch_gains``): the same program without the
              turns (the PID cascade has no turn command)
    position  the position network over the velocity network (``frozen/position``, ``frozen/velocity``) follows a
              figure 8 on the floor, like the position demo of Drone_RL: the path is drawn a little ahead of the car
              (blue dots), a reference point (orange ball) runs along it, and the network is given the path point
              0.6 s ahead of it as its goal (``tools/path_following.py``)
    position_gains  the same figure 8 with the go-to-goal PID whose gains a network writes (``frozen/position_gains``)
    lqr       no network: LQR balances at the commanded wheel speed and a PI on the gyro holds the turn rate
              (``pid_control/drive.py``), on the program of ``rl``
    lqr_position    the figure 8 with the go-to-goal PID over that LQR drive

The command is a fixed program instead of the random one of training. The camera is scripted: it follows the car,
circles it slowly and always looks at it. The scene is drawn for the picture: an HDR sky, a sun that casts shadows, ray
tracing (RTX real-time path tracing; ``--path_tracing`` for the slow, offline-quality one) and DLAA. Two
arrows above the car show the commanded speed (blue) and the speed (green), the cyan dots on the ground are the last
seconds of the track, and the position clips draw the path ahead (blue dots; ``--reference_ball`` also draws the
reference point running along it).

    python src/Balance_Car_RL/car/tools/record_demo.py              # RL velocity network
    python src/Balance_Car_RL/car/tools/record_demo.py --cascade gains
    python src/Balance_Car_RL/car/tools/record_demo.py --cascade position
    python src/Balance_Car_RL/car/tools/record_demo.py --cascade lqr_position

With ``--ros`` the controller is the ROS 2 node of ``ros/src/car_bridge`` instead of the frozen TorchScript: the
simulated car publishes its raw IMU and wheel speeds and applies the node's wheel torques (``tools/ros_link.py``), as
the real car would. Run it with ROS sourced:

    source /opt/ros/jazzy/setup.bash
    python src/Balance_Car_RL/car/tools/record_demo.py --cascade position --ros --name position_ros_demo

The clip is written to ``videos/demo/<name>.mp4`` (1920x1080, 50 fps, real time) and, with ``--gif``, a small gif for
the README (under the 2000 kB of the pre-commit hook). Try ``--seconds 4 --width 960 --height 540`` first.
"""

import argparse
import math
import subprocess
import time
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parents[1]
FROZEN = HERE / "rl_control" / "frozen"
CASCADES = {
    "rl": "velocity",
    "gains": "velocity_gains",
    "position": "position",
    "position_gains": "position_gains",
    "lqr": None,
    "lqr_position": None,
}
"""Cascade -> frozen stage of its top network (None: the classical LQR drive, no network)."""

# ── Command line ──────────────────────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--cascade",
    default="rl",
    choices=list(CASCADES),
    help="rl: the velocity network; gains: networks write the PID gains; position: the position network over the "
    "velocity network; position_gains: the go-to-goal PID with network gains over the velocity network; lqr, "
    "lqr_position: the same programs as rl and position with LQR and PID instead of networks",
)
parser.add_argument("--policy", default=None, help="exported policy.pt of the top layer (default: the frozen one)")
parser.add_argument("--ros", action="store_true", help="the ROS 2 node runs the cascade (its frozen ONNX networks)")
parser.add_argument("--out", default="videos/demo", help="folder of the clip")
parser.add_argument("--name", default=None, help="file name: <name>.mp4 (default: <cascade>_demo)")
parser.add_argument("--seconds", type=float, default=None, help="length of the clip (default: the whole program)")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--fps", type=int, default=50, help="50 is real time (the speed layer runs at 50 Hz)")
parser.add_argument("--light", type=float, default=1.0, help="scales the sun and the sky light: 0.5 darker, 2 brighter")
parser.add_argument(
    "--sky",
    default="day",
    choices=["day", "color"],
    help="day: sky texture of the Isaac Sim assets (downloaded the first time); color: plain sky",
)
parser.add_argument(
    "--load_wait",
    type=float,
    default=30.0,
    help="seconds to render before the clip starts, so the sky texture is loaded (it streams in)",
)
parser.add_argument(
    "--path_tracing",
    action="store_true",
    help="full path tracing with the denoiser: best picture, several times slower",
)
parser.add_argument(
    "--gif", default=None, help="also write a small gif here, at twice the speed (e.g. guide/media/rl_demo.gif)"
)
parser.add_argument("--gif_speedup", type=float, default=2.0, help="how much faster the gif plays than real time")
parser.add_argument("--gif_fps", type=int, default=10, help="frames per second of the gif (fewer: smaller file)")
parser.add_argument(
    "--reference_ball",
    action="store_true",
    help="position clips: also draw the reference point running along the path (orange ball)",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.video = True  # the viewport must render, also headless
app = AppLauncher(args).app

import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.envs import mdp as isaac_mdp  # noqa: E402
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, SPHERE_MARKER_CFG  # noqa: E402
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR  # noqa: E402

from Balance_Car_RL.car import mdp  # noqa: E402
from Balance_Car_RL.car.lqr_control import LQRController  # noqa: E402
from Balance_Car_RL.car.rl_control.gains_env_cfg import VelocityGainsEnvCfg  # noqa: E402
from Balance_Car_RL.car.rl_control.position_env_cfg import PositionEnvCfg  # noqa: E402
from Balance_Car_RL.car.rl_control.position_gains_env_cfg import PositionGainsEnvCfg  # noqa: E402
from Balance_Car_RL.car.rl_control.velocity_env_cfg import VelocityEnvCfg  # noqa: E402
from Balance_Car_RL.car.tools.path_following import (  # noqa: E402
    EIGHT_PERIOD,
    LOOKAHEAD,
    PATH_DOT_SPACING,
    PATH_END_HOLD,
    PATH_HOLD,
    path_goal,
    path_reference,
)
from Balance_Car_RL.car.tools.ros_link import RosLink, wheel_torque_actions_cfg  # noqa: E402
from Balance_Car_RL.car.tools.scripted import (  # noqa: E402
    ClassicalDrive,
    Command,
    program_value,
    scripted_cfg,
    with_drive_action,
)

torch.set_grad_enabled(False)

# ── Look ──────────────────────────────────────────────────────────────────────────────────────────────────────
BACKGROUND = (0.55, 0.7, 0.9)  # sky colour with --sky color
SKY = (
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    300.0,
    3000.0,
    (1.0, 0.95, 0.88),
)  # texture, sky light intensity, sun intensity, sun colour
TRAIL_COLOR, TRAIL_RADIUS = (0.15, 0.95, 1.0), 0.004  # [m] radius of a dot
GOAL_COLOR, GOAL_RADIUS = (1.0, 0.45, 0.05), 0.02  # [m] the reference point of the position clip
PATH_COLOR, PATH_RADIUS = (0.25, 0.55, 1.0), 0.005  # [m] the dots of the path ahead
TRAIL_POINTS, TRAIL_EVERY = 75, 2  # 75 dots, one every 2 steps: the last 3 s
ARROW_HEIGHT = 0.20  # [m] arrows above the ground
ARROW_LENGTH = 4.0  # arrow marker scale along its axis per m/s (0.3 m/s: about 12 cm)
ARROW_WIDTH = 0.03  # arrow marker scale across its axis
FLOOR = (
    f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl"
)
"""Marble tiles of the Isaac Lab locomotion scenes: the default grid has bright green marks every few metres."""

# ── The program: (start time [s], (speed [m/s], turn rate [rad/s])); a push at PUSH_TIME ──────────────────────────
STEP_DT = 0.02  # [s] one policy step
PROGRAM = [(0.0, (0.0, 0.0)), (2.0, (0.3, 0.0)), (4.5, (0.3, 1.0)), (7.0, (0.0, 0.0)), (8.0, (0.0, -2.0))]
PROGRAM += [(10.0, (-0.3, 0.0)), (13.0, (0.0, 0.0))]
PUSH_TIME, PUSH_SPEED = 14.5, 0.35  # [s], [m/s] forward speed added to the body
PROGRAM_END = 18.0
POSITION = args.cascade in ("position", "position_gains", "lqr_position")
VELOCITY = args.cascade in ("rl", "lqr")  # (v_x, w_z) program
if args.cascade == "gains":  # no turn command in the PID cascade
    PROGRAM = [(0.0, (0.0, 0.0)), (2.0, (0.3, 0.0)), (6.0, (0.0, 0.0)), (8.5, (-0.3, 0.0)), (12.5, (0.0, 0.0))]


# ── The path of the position clip (tools/path_following.py) ───────────────────────────────────────────────────────
if POSITION:
    PROGRAM_END = PATH_HOLD + EIGHT_PERIOD + PATH_END_HOLD


# ── Camera: keyframes (share of the program, azimuth [deg], distance [m], height [m]) around the car ──────────
CAMERA_KEYS = [(0.0, -60.0, 0.9, 0.35), (0.3, -95.0, 0.75, 0.22), (0.6, -130.0, 0.8, 0.3), (1.0, -200.0, 0.9, 0.35)]
if POSITION:  # higher and farther: the figure 8 is 1.2 m wide
    CAMERA_KEYS = [(0.0, -70.0, 1.4, 0.85), (0.5, -160.0, 1.25, 0.7), (1.0, -250.0, 1.4, 0.85)]


def smooth_keys(share: float, keys):
    """Interpolate the keyframes with a smoothstep between two of them (no jerk at the keys)."""
    if share <= keys[0][0]:
        return np.array(keys[0][1:], dtype=float)
    for (s0, *a), (s1, *b) in zip(keys[:-1], keys[1:], strict=True):
        if share < s1:
            u = (share - s0) / (s1 - s0)
            u = u * u * (3.0 - 2.0 * u)
            return (1.0 - u) * np.array(a) + u * np.array(b)
    return np.array(keys[-1][1:], dtype=float)


def camera_pose(t: float, car: np.ndarray):
    """Camera position and the point it looks at: the scripted orbit around the car, looking at its body."""
    azimuth, distance, height = smooth_keys(t / PROGRAM_END, CAMERA_KEYS)
    a = math.radians(azimuth)
    look = car + np.array([0.0, 0.0, 0.07])
    return look + np.array([distance * math.cos(a), distance * math.sin(a), height - 0.07]), look


# ── Scene ─────────────────────────────────────────────────────────────────────────────────────────────────────
def make_ground() -> TerrainImporterCfg:
    """A flat floor of marble tiles (one 20 m tile and a border), same friction as the training ground plane."""
    generator = TerrainGeneratorCfg(
        size=(20.0, 20.0),
        border_width=20.0,
        num_rows=1,
        num_cols=1,
        use_cache=False,
        sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
    )
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        visual_material=sim_utils.MdlFileCfg(mdl_path=FLOOR, project_uvw=True, texture_scale=(0.25, 0.25)),
    )


@configclass
class RosActionsCfg:
    wheel_torque = wheel_torque_actions_cfg()


def make_cfg(steps: int):
    cfg_class = {
        "rl": VelocityEnvCfg,
        "gains": VelocityGainsEnvCfg,
        "position": PositionEnvCfg,
        "position_gains": PositionGainsEnvCfg,
        "lqr": VelocityEnvCfg,
        "lqr_position": PositionEnvCfg,
    }[args.cascade]
    cfg = scripted_cfg(cfg_class(), steps * STEP_DT)  # one scripted push; the demo draws its own arrows, ball, path
    if CASCADES[args.cascade] is None:
        with_drive_action(cfg)
    cfg.scene.num_envs = 1
    if args.ros:  # the node writes the wheel torques
        cfg.actions = RosActionsCfg()
    cfg.scene.ground = make_ground()
    if args.sky != "color":
        texture, sky_light, sun_light, sun_color = SKY
        sky = sim_utils.DomeLightCfg(
            intensity=sky_light * args.light, texture_file=texture, visible_in_primary_ray=True
        )
    else:
        sun_light, sun_color = SKY[2], SKY[3]
        sky = sim_utils.DomeLightCfg(intensity=400.0 * args.light, color=BACKGROUND, visible_in_primary_ray=True)
    cfg.scene.dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sky)
    cfg.scene.sun = AssetBaseCfg(  # the only light that casts a shadow of the car
        prim_path="/World/sun",
        spawn=sim_utils.DistantLightCfg(intensity=sun_light * args.light, angle=0.6, color=sun_color),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3827, 0.0, 0.0, 0.9239)),  # 45 degrees about x
    )
    cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(
            eye=(0.6, -0.6, 0.3),
            lookat=(0.0, 0.0, 0.07),
            window_width=args.width,
            window_height=args.height,
            background_color=BACKGROUND if args.sky == "color" else None,
        )
    ]  # a colour here would hide the sky texture
    # The recorder of Isaac Lab keeps every frame in memory until the end: here it only grabs frames (step_offset keeps
    # it from recording by itself), and each frame goes straight to ffmpeg.
    cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=args.out, step_offset=10**9)]
    return cfg


def set_render_quality() -> None:
    """Maximum picture quality: ray tracing, DLAA, shadows, ambient occlusion, global illumination."""
    try:
        import carb.settings

        quality = {
            "/rtx/post/aa/op": 4,
            "/rtx/post/histogram/enabled": True,  # DLAA; fixed exposure
            "/rtx/shadows/enabled": True,
            "/rtx/ambientOcclusion/enabled": True,
            "/rtx/indirectDiffuse/enabled": True,
            "/rtx/directLighting/sampledLighting/enabled": True,
            "/rtx/raytracing/subpixel/mode": 2,
            "/rtx/rtpt/maxBounces": 8,
            "/rtx/rtpt/adaptiveSampling/disocclusion/spp": 8,
        }
        if args.path_tracing:
            quality.update(
                {
                    "/rtx/rendermode": "PathTracing",
                    "/rtx/pathtracing/spp": 16,
                    "/rtx/pathtracing/totalSpp": 16,
                    "/rtx/pathtracing/maxBounces": 8,
                    "/rtx/pathtracing/optixDenoiser/enabled": True,
                }
            )
        else:
            quality["/rtx/rendermode"] = "RealTimePathTracing"
        settings = carb.settings.get_settings()
        for key, value in quality.items():
            settings.set(key, value)
    except Exception as error:  # noqa: BLE001
        print("DEMO render settings skipped:", error)


class Arrow:
    """An arrow along the car's heading above the car whose length is a speed; hidden near zero."""

    def __init__(self, prim_path: str, cfg, device):
        cfg = cfg.copy()
        cfg.prim_path = prim_path
        self.markers = VisualizationMarkers(cfg)
        self.markers.set_visibility(False)
        self.device = device

    def show(self, position: np.ndarray, speed: float, heading: float = 0.0) -> None:
        self.markers.set_visibility(abs(speed) > 0.05)
        if abs(speed) <= 0.05:
            return
        yaw = heading if speed > 0 else heading + math.pi  # quaternion (x, y, z, w) about z
        quat = torch.tensor([[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]], device=self.device)
        scale = torch.tensor([[abs(speed) * ARROW_LENGTH, ARROW_WIDTH, ARROW_WIDTH]], device=self.device)
        self.markers.visualize(
            translations=torch.tensor([position], dtype=torch.float32, device=self.device),
            orientations=quat,
            scales=scale,
        )


class Hidden:
    """Stands in for a marker that is not drawn."""

    def show(self, *args, **kwargs) -> None:
        pass


class Dots:
    """A group of small glowing spheres (the track, or the goal), hidden while it has no point."""

    def __init__(self, prim_path: str, device, color=TRAIL_COLOR, radius=TRAIL_RADIUS):
        marker = SPHERE_MARKER_CFG.copy()
        marker.prim_path = prim_path
        marker.markers["sphere"].radius = radius
        marker.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=color, emissive_color=tuple(0.5 * c for c in color)
        )
        self.markers = VisualizationMarkers(marker)
        self.markers.set_visibility(False)
        self.device = device

    def show(self, points: np.ndarray) -> None:
        self.markers.set_visibility(len(points) > 0)
        if len(points) > 0:
            self.markers.visualize(translations=torch.tensor(points, dtype=torch.float32, device=self.device))


def write_gif(clip: Path, gif: Path, width: int = 360, fps: int = 10, speedup: float = 2.0) -> None:
    """A small gif of the whole clip, played ``speedup`` times faster, with its own palette and no dithering. The
    pre-commit hook rejects files above 2000 kB, and the camera moves, so every frame changes: in real time the gif of
    the 18 s program is about 3 MB even at 360 pixels."""
    gif.parent.mkdir(parents=True, exist_ok=True)
    scale = f"setpts=PTS/{speedup},fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(clip),
            "-filter_complex",
            f"[0:v]{scale},split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=none:diff_mode=rectangle",
            "-loop",
            "0",
            str(gif),
        ],
        check=True,
    )
    print(f"DEMO gif written to {gif} ({gif.stat().st_size // 1024} kB)")


# ── Run ───────────────────────────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    seconds = args.seconds or PROGRAM_END
    steps, warm_steps = int(round(seconds / STEP_DT)), 10
    env = ManagerBasedRLEnv(make_cfg(steps))
    set_render_quality()
    device = env.device
    robot = env.scene["robot"]

    position = POSITION
    trail = Dots("/Visuals/Demo/trail", device)
    target_arrow = Arrow("/Visuals/Demo/target", BLUE_ARROW_X_MARKER_CFG, device)
    speed_arrow = Arrow("/Visuals/Demo/speed", GREEN_ARROW_X_MARKER_CFG, device)
    goal_ball = Dots("/Visuals/Demo/goal", device, GOAL_COLOR, GOAL_RADIUS) if args.reference_ball else Hidden()
    path_dots = Dots("/Visuals/Demo/path", device, PATH_COLOR, PATH_RADIUS) if position else None
    obs, _ = env.reset()
    if args.ros:
        policy = None
    elif CASCADES[args.cascade] is None:
        policy = ClassicalDrive(env, LQRController(device=device))
    else:
        policy = torch.jit.load(args.policy or str(FROZEN / CASCADES[args.cascade] / "policy.pt"), map_location=device)
    if args.sky != "color":  # without the wait the first frames have no sky texture
        deadline = time.time() + args.load_wait
        while time.time() < deadline:
            app.update()
    link = RosLink(env, args.cascade) if args.ros else None
    command = Command(env)  # the command is the program, not a random one
    action_term = env.action_manager.get_term(env.action_manager.active_terms[0])
    start_xy = command.start_xy[0].cpu().numpy()  # the car starts here, facing +x
    dot_times = np.arange(PATH_HOLD, PATH_HOLD + EIGHT_PERIOD, PATH_DOT_SPACING)
    dot_points = np.array([[*(start_xy + path_reference(tp)[0]), PATH_RADIUS] for tp in dot_times])

    Path(args.out).mkdir(parents=True, exist_ok=True)
    clip = Path(args.out) / f"{args.name or args.cascade + ('_ros' if args.ros else '') + '_demo'}.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(clip),
        (args.width, args.height),
        fps=args.fps,
        codec="libx264",
        quality=None,
        macro_block_size=1,
        output_params=["-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    writer.send(None)

    trail_points: list = []
    car_smooth, eye_smooth = None, None
    max_tilt, errors = 0.0, []
    for k in range(steps + warm_steps):
        t = max(0.0, (k - warm_steps) * STEP_DT)
        target_speed, target_turn = (0.0, 0.0) if position else program_value(PROGRAM, t)
        if k == warm_steps + int(round(PUSH_TIME / STEP_DT)) and not position:
            push = {"x": (PUSH_SPEED, PUSH_SPEED)}
            isaac_mdp.push_by_setting_velocity(env, torch.tensor([0], device=device), push)

        car = isaac_mdp.root_pos_w(env)[0].cpu().numpy()
        speed = float(robot.data.root_lin_vel_b.torch[0, 0])
        car_smooth = car.copy() if car_smooth is None else car_smooth + 0.1 * (car - car_smooth)
        eye, look = camera_pose(t, car_smooth)
        eye_smooth = eye if eye_smooth is None else eye_smooth + 0.15 * (eye - eye_smooth)
        env.sim.set_camera_view(tuple(float(v) for v in eye_smooth), tuple(float(v) for v in look))

        if k % TRAIL_EVERY == 0:
            trail_points = (trail_points + [[car[0], car[1], 0.002]])[-TRAIL_POINTS:]
            trail.show(np.array(trail_points))
        above = car + np.array([0.0, 0.0, ARROW_HEIGHT - car[2]])
        heading = float(robot.data.heading_w.torch[0])
        if position:  # the goal: the path point a little ahead, facing along the path
            reference = start_xy + path_reference(t)[0]
            command.set(obs["policy"], path_goal(t))
            goal_ball.show(np.array([[reference[0], reference[1], GOAL_RADIUS]]))
            path_dots.show(dot_points[dot_times <= t + LOOKAHEAD])
            # blue: the speed command the position layer chose last step (the processed action is (v_x, w_z))
            if link:
                target_speed = link.command[0]
            elif isinstance(policy, ClassicalDrive):
                target_speed = float(policy.command[0, 0]) if policy.command is not None else 0.0
            else:
                target_speed = float(action_term.processed_actions[0, 0])
        elif args.cascade == "gains":
            command.set(obs["policy"], target_speed)  # also read by the speed layer's guard
        else:
            command.set(obs["policy"], (target_speed, target_turn))
        target_arrow.show(above, target_speed, heading)
        speed_arrow.show(above + np.array([0.0, 0.0, 0.03]), speed, heading)
        if link is not None:
            wanted = (target_speed, target_turn) if VELOCITY else target_speed
            action = link.step(path_goal(t) if position else wanted)
        else:
            action = policy(obs["policy"]).clamp(-1.0, 1.0)
        obs, _, _, _, _ = env.step(action)
        if k >= warm_steps:
            max_tilt = max(max_tilt, abs(float(mdp.body_pitch(env)[0])))
            if position:
                errors.append(float(np.linalg.norm(isaac_mdp.root_pos_w(env)[0, :2].cpu().numpy() - reference)))
            else:
                errors.append(abs(speed - target_speed))
            frame = env.video_recorders[0]._get_frame()
            if frame is not None:
                writer.send(np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8))
    writer.close()
    what = "distance to the reference point| {:.3f} m, last {:.3f} m" if position else "speed error| {:.3f} m/s"
    print(
        f"DEMO written to {clip}: largest tilt {max_tilt:.3f} rad, median |"
        + what.format(np.median(errors), errors[-1] if errors else float("nan"))
    )
    if args.gif:
        write_gif(clip, Path(args.gif), fps=args.gif_fps, speedup=args.gif_speedup)
    if link is not None:
        link.close()
    env.close()


main()
app.close()
