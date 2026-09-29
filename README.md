<p align="center">
  <img src="docs/media/banner.png" alt="Pololu Balboa balancing in Isaac Sim" width="720">
</p>

# Balance_Car_RL

Reinforcement learning for a **two-wheeled self-balancing robot** (Pololu Balboa 32U4) in **Isaac Lab 3.0**. The controller works from
a raw IMU and wheel encoders, as the real robot would, and is trained as a **frozen cascade**: a first policy learns to hold a lean,
its weights are frozen, and a second policy learns to drive at a commanded speed on top of it. PID and LQR baselines and a ROS 2 node
are included. Every training run is recorded on video.

<p align="center">
  <img src="docs/media/pitch_demo.gif" alt="Stage 1: holding a commanded lean" width="360">
  <img src="docs/media/velocity_demo.gif" alt="Stage 2: tracking a speed" width="360">
</p>

<p align="center"><a href="https://www.youtube.com/watch?v=YOUR_VIDEO_ID">Demo video</a></p>

| | |
|---|---|
| Robot | Pololu Balboa 32U4: 80 mm wheels, 50:1 HPCB 6 V gearmotors behind a 49:17 gearbox, LSM6DS33 IMU |
| Simulator | Isaac Sim 6.1 + Isaac Lab 3.0, PhysX |
| Sensing | raw accelerometer and gyro with the datasheet noise, complementary filter to pitch |
| RL | rsl_rl 5.5.1 (PPO); stage 1 `BalanceCar-Pitch-v0`, stage 2 `BalanceCar-Velocity-v0`; baseline `BalanceCar-Upright-v0` |
| Baselines | cascaded PID, LQR |
| Extras | CAD to URDF tool, one-command cascade training with video, evaluation script, ROS 2 bridge, tests that need no simulator |

## 1. Requirements

- Linux x86_64 (Ubuntu), NVIDIA GPU with a recent driver (developed on an RTX 3060, 12 GB)
- Python 3.12, [Miniconda](https://docs.conda.io/en/latest/miniconda.html), [Git](https://git-scm.com/)
- MoviePy for the training videos (installed below)
- Optional, for deployment: [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html) (Ubuntu 24.04)

## 2. Install

```bash
# 1. Python environment
conda create -n env_isaaclab python=3.12 -y
conda activate env_isaaclab

# 2. Isaac Sim 6.1 from the NVIDIA index
pip install "isaacsim[all,extscache]==6.1.0" --extra-index-url https://pypi.nvidia.com

# 3. Isaac Lab (sibling folder)
cd ~/Documents/GitHub
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh -i     # installs Isaac Lab, rsl_rl and the `isaaclab` command

# 4. This project
cd ~/Documents/GitHub
git clone https://github.com/vohongquann/Balance_Car_RL.git
cd Balance_Car_RL
pip install -e . --no-deps          # registers the BalanceCar-* tasks
pip install pytest scipy "moviepy<2" onnx
```

`isaaclab` does not exist before step 3: `./isaaclab.sh -i` pip-installs the Isaac Lab packages, one of which declares a console
script named `isaaclab` in the `bin/` folder of the environment. `isaaclab.sh` is deprecated in Isaac Lab 3.x in favour of
`uv run isaaclab`; this project was developed with the conda route above. The first Isaac Sim launch asks you to accept the NVIDIA EULA
and downloads extensions and assets, so it takes a few minutes. If a step fails, the
[Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html) is the reference for steps 2 and 3.

Check it (every command below runs in a terminal with `env_isaaclab` active, from the `Balance_Car_RL` folder):

```bash
which isaaclab && isaaclab --help
isaaclab list_envs --keyword BalanceCar                 # three tasks
PYTHONPATH=ros/src/car_bridge python -m pytest tests ros/src/car_bridge/test/test_policy.py ros/src/car_bridge/test/test_estimator.py -q
```

## 3. Run

Everything in one command: train stage 1, freeze it, train stage 2 on top of it, evaluate each, record video.

```bash
python scripts/train_cascade.py          # about 6 and 10 minutes with video, 3 to 5 without (--no-video)
```

For each stage it trains (recording clips of the training), plays the result and exports it, evaluates it against pass criteria,
freezes it into `src/Balance_Car_RL/car/rl_control/frozen/<stage>/` only if it passed, and writes the videos, gifs and the training curve
to `docs/media/`. Details: [docs/04-training.md](docs/04-training.md).

Other things you will want:

```bash
python scripts/train_cascade.py --stages pitch                  # one stage
python scripts/evaluate.py --task BalanceCar-Velocity-v0        # numbers for the newest run of a task
python scripts/evaluate.py --task BalanceCar-Upright-v0 --controller lqr     # or pid
isaaclab play --rl_library rsl_rl --task BalanceCar-Velocity-v0 --num_envs 4 --viz kit    # watch it
tensorboard --logdir logs/rsl_rl
```

Headless is the default in Isaac Lab 3.0; `--viz kit` opens the Isaac Sim viewer. Logs and checkpoints go to
`./logs/rsl_rl/<experiment>/<run>/`. `isaaclab train` returns exit code 0 even when it crashes: success means `Training time` in its output.
Without `--checkpoint`, `evaluate.py` loads the **newest** run, which may be an untrained one.

## 4. Results

Final policies, 768 evaluation episodes each (random start tilt, pushes, IMU noise). Details, and what changed on the way: [docs/04-training.md](docs/04-training.md) §4.7 and
[docs/05-controllers.md](docs/05-controllers.md) §5.5.

| Cascade stage | Falls | Median error | RMS error |
|---|---|---|---|
| 1. `BalanceCar-Pitch-v0`: hold a commanded lean of ±0.08 rad | 0.13 % | 0.017 rad | 0.053 rad |
| 2. `BalanceCar-Velocity-v0`: track a commanded speed of ±0.4 m/s, on the frozen stage 1 | 0 % | 0.038 m/s | 0.164 m/s |

| Upright baseline | Falls | RMS pitch | Mean wheel torque |
|---|---|---|---|
| LQR | 0 | 0.033 rad | 0.00094 N m |
| Cascaded PID | 0.4 % | 0.050 rad | 0.00205 N m |
| RL policy | 0 | 0.043 rad | 0.00538 N m |

The RMS speed error is dominated by short transients (command steps, recovery from the start tilt): the 90th percentile is 0.29 m/s while
the median is 0.038 m/s. Sim-to-real: not tested. Curves: [pitch](docs/media/pitch_training_curve.png), [velocity](docs/media/velocity_training_curve.png).

## 5. Documentation

Read [docs/](docs/README.md) in order:

1. [00-overview.md](docs/00-overview.md): the flow, the layout, which command for which question
2. [01-robot.md](docs/01-robot.md): hardware, CAD to URDF, every number with its source, how to measure the real robot
3. [02-sensing.md](docs/02-sensing.md): raw IMU to gravity direction to pitch
4. [03-dynamics.md](docs/03-dynamics.md): pendulum model, motor, discretization
5. [04-training.md](docs/04-training.md): tasks, rewards, the frozen cascade, videos, PPO
6. [05-controllers.md](docs/05-controllers.md): PID, LQR and RL compared, tuning
7. [06-deployment.md](docs/06-deployment.md): ROS 2 node, checklist for the real robot

## 6. Limitations

Masses (0.27 kg body, 0.02 kg per wheel), the motor rotor inertia, gearbox efficiency, the IMU offsets left after calibration and the IMU
mounting are assumptions ([docs/01-robot.md](docs/01-robot.md) §1.5). The simulation has IMU noise, but no encoder noise, latency, or
mass and friction randomization, and nothing has run on the real robot: treat sim-to-real as untested
([docs/06-deployment.md](docs/06-deployment.md)).

## 7. Development

```bash
PYTHONPATH=ros/src/car_bridge python -m pytest tests ros/src/car_bridge/test/test_policy.py ros/src/car_bridge/test/test_estimator.py -q   # no simulator needed
python tools/build_balboa_urdf.py            # rebuild the URDF after a CAD or mass change (needs cad/, see tools/fetch_pololu_cad.sh)
```

After changing a constant in `src/Balance_Car_RL/car/car_cfg.py`: retrain the stages in order (`train_cascade.py`), update
`TORQUE_SCALE_NM` in `ros/src/car_bridge/car_bridge/policy.py` if the stall torque changed (a test checks it), and rerun `design_lqr()`.

## Acknowledgements

Built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab). Geometry, motor and IMU numbers come from the
[Pololu Balboa 32U4 kit](https://www.pololu.com/product/3575) CAD, user's guide and gearmotor datasheet (Pololu Corporation) and the
ST LSM6DS33 datasheet.

## License

See [LICENSE](LICENSE).
