r"""RL tasks of the Balboa: one file per task, built from the shared terms of ``car/mdp/``.

The RL cascade copies Isaac Lab's legged locomotion: a velocity policy that goes from a (v_x, w_z) command straight to
the actuators (``isaaclab_tasks/core/velocity``), and a navigation policy over it, frozen, that writes that command to
reach a 2D pose (``isaaclab_tasks/contrib/navigation``):

    BalanceCar-Velocity-v0   velocity_env_cfg.py   IMU, speed, turn rate, (v_x, w_z)   -> wheel torques  (stage 1)
    BalanceCar-Position-v0   position_env_cfg.py   IMU, speed, turn rate, goal x y h   -> (v_x, w_z)     (stage 2)
    BalanceCar-Upright-v0    upright_env_cfg.py    IMU, wheel speeds                   -> wheel torques  (ROS baseline)

Gain tasks: the layers of the classical cascade (speed PID -> pitch target -> pitch PID -> torque), where a network
writes the three PID gains (kp, ki, kd) of its layer and the PID of ``pid_control/`` computes the output (the zero
action is the tuned PID; ``gains_env_cfg.py``, ``mdp/gains.py``). Trained bottom-up, the speed layer over the frozen
pitch one:

    BalanceCar-Pitch-Gains-v0      pitch target -> pitch PID gains -> wheel torques      (frozen/pitch_gains/)
    BalanceCar-Velocity-Gains-v0   speed target -> speed PID gains -> pitch target -> frozen pitch gain layer
    BalanceCar-Position-Gains-v0   goal (x, y, heading) -> go-to-goal PID gains -> (v_x, w_z) -> frozen velocity network
                                   (the PID layer of the position task, ``position_gains_env_cfg.py``)

    car_env_cfg.py           scene, action, observation, events, rewards, terminations shared by every task
    pitch_env_cfg.py         the pitch layer of the gain cascade (command, observation, rewards)
    frozen/                  <stage>/policy.pt (+ policy.onnx, meta.json) of a finished stage, loaded by the stage above

PPO settings: ``agents/<task>_ppo_cfg.py``. Workflow: guide/04_training.md.

Commands, from the repo root. Freezing = play and export the newest run, evaluate it, and only if it passes copy it to
``frozen/<stage>/`` with its ``meta.json`` (``scripts/train_cascade.py``). ``--viz kit`` opens the Isaac Sim window
(slower, so fewer envs); without it: headless. ``isaaclab train`` exits with code 0 even when it crashes: look for
``Training time``.

    # everything in one command: train, play, evaluate, freeze, video, for velocity and then position
    python scripts/train_cascade.py                      # [--no-video] [--stages velocity] [--dry-run]

    # the same step by step
    # 1. velocity (nothing frozen below it)
    isaaclab train --rl_library rsl_rl --task BalanceCar-Velocity-v0 --num_envs 16384               # [--viz kit]
    python scripts/train_cascade.py --stages velocity --reuse-latest    # export, evaluate, freeze the newest run

    # 2. position (needs frozen/velocity/)
    isaaclab train --rl_library rsl_rl --task BalanceCar-Position-v0 --num_envs 16384
    python scripts/train_cascade.py --stages position --reuse-latest

    # 3. upright, for the ROS node (nothing is frozen: its ONNX goes to models/)
    isaaclab train --rl_library rsl_rl --task BalanceCar-Upright-v0 --num_envs 4096
    RUN=$(ls -d logs/rsl_rl/balance_car_upright/*/ | tail -1)
    python scripts/evaluate.py --task BalanceCar-Upright-v0 --num_envs 256 --checkpoint ${RUN}model_299.pt
    timeout 150 isaaclab play --rl_library rsl_rl --task BalanceCar-Upright-v0 --num_envs 4 \
        --checkpoint ${RUN}model_299.pt
    # one file, without the stack traces (absolute paths of this machine) the exporter stores per node
    mkdir -p models && python -c "import runpy; runpy.run_path('scripts/train_cascade.py')['onnx_single_file'](\
        '${RUN}exported/policy.onnx', 'models/balance_car_policy.onnx')"

    # 4. the gain tasks: the network writes PID gains. Velocity needs frozen/pitch_gains/, position frozen/velocity/
    python scripts/train_cascade.py --stages pitch_gains velocity_gains position_gains    # [--no-video] [--dry-run]

    # look at a policy, and the baselines for comparison
    isaaclab play --rl_library rsl_rl --task BalanceCar-Velocity-v0 --num_envs 4 --viz kit
    python scripts/evaluate.py --task BalanceCar-Upright-v0 --num_envs 256 --controller lqr     # or pid

Quick check that an env builds (a few seconds): add ``--num_envs 64 --max_iterations 2`` to a train command.
"""

import gymnasium as gym

_TASKS = {
    "BalanceCar-Upright-v0": ("upright_env_cfg:UprightEnvCfg", "upright_ppo_cfg:UprightPPORunnerCfg"),
    "BalanceCar-Velocity-v0": ("velocity_env_cfg:VelocityEnvCfg", "velocity_ppo_cfg:VelocityPPORunnerCfg"),
    "BalanceCar-Position-v0": ("position_env_cfg:PositionEnvCfg", "position_ppo_cfg:PositionPPORunnerCfg"),
    "BalanceCar-Pitch-Gains-v0": ("gains_env_cfg:PitchGainsEnvCfg", "gains_ppo_cfg:PitchGainsPPORunnerCfg"),
    "BalanceCar-Velocity-Gains-v0": ("gains_env_cfg:VelocityGainsEnvCfg", "gains_ppo_cfg:VelocityGainsPPORunnerCfg"),
    "BalanceCar-Position-Gains-v0": (
        "position_gains_env_cfg:PositionGainsEnvCfg",
        "position_gains_ppo_cfg:PositionGainsPPORunnerCfg",
    ),
}

for _task, (_env, _agent) in _TASKS.items():
    gym.register(
        id=_task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_env}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.{_agent}",
            "default_agent": "rsl_rl",
        },
    )
