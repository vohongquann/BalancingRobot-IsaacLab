r"""RL tasks of the Balboa: one file per task, built from the shared terms of ``car/mdp/``.

    BalanceCar-Upright-v0    upright_env_cfg.py    IMU, wheel speeds         -> wheel torques  (baseline)
    BalanceCar-Pitch-v0      pitch_env_cfg.py      + pitch target            -> wheel torques  (cascade stage 1)
    BalanceCar-Velocity-v0   velocity_env_cfg.py   IMU, speed, speed target  -> pitch target   (stage 2, frozen pitch)

    car_env_cfg.py           scene, action, observation, events, rewards, terminations shared by the three
    frozen/                  <stage>/policy.pt (+ policy.onnx, meta.json) of a finished stage, loaded by the stage above

PPO settings: ``agents/<task>_ppo_cfg.py``. Workflow: guide/04_training.md.

Commands, from the repo root, in the order to run them. The cascade is trained bottom-up: pitch, then velocity on the
frozen pitch. Freezing = play and export the newest run, evaluate it, and only if it passes copy it to
``frozen/<stage>/`` with its ``meta.json`` (``scripts/train_cascade.py``). ``--viz kit`` opens the Isaac Sim window
(slower, so fewer envs); without it: headless. ``isaaclab train`` exits with code 0 even when it crashes: look for
``Training time``.

    # everything in one command: train, play, evaluate, freeze, video, for pitch and then velocity
    python scripts/train_cascade.py                      # [--no-video] [--stages pitch] [--dry-run]

    # the same step by step
    # 1. pitch (nothing frozen below it)
    isaaclab train --rl_library rsl_rl --task BalanceCar-Pitch-v0 --num_envs 4096 \
        --video --video_length 300 --video_interval 2400                           # [--viz kit]
    python scripts/train_cascade.py --stages pitch --reuse-latest       # export, evaluate, freeze the newest run

    # 2. velocity (needs frozen/pitch/)
    isaaclab train --rl_library rsl_rl --task BalanceCar-Velocity-v0 --num_envs 4096 \
        --video --video_length 300 --video_interval 2400
    python scripts/train_cascade.py --stages velocity --reuse-latest

    # 3. upright, for the ROS node (nothing is frozen: its ONNX goes to models/)
    isaaclab train --rl_library rsl_rl --task BalanceCar-Upright-v0 --num_envs 4096
    RUN=$(ls -d logs/rsl_rl/balance_car_upright/*/ | tail -1)
    python scripts/evaluate.py --task BalanceCar-Upright-v0 --num_envs 256 --checkpoint ${RUN}model_299.pt
    timeout 150 isaaclab play --rl_library rsl_rl --task BalanceCar-Upright-v0 --num_envs 4 \
        --checkpoint ${RUN}model_299.pt
    mkdir -p models && python -c "import onnx; onnx.save_model(onnx.load('${RUN}exported/policy.onnx', \
        load_external_data=True), 'models/balance_car_policy.onnx', save_as_external_data=False)"

    # look at a policy, and the baselines for comparison
    isaaclab play --rl_library rsl_rl --task BalanceCar-Velocity-v0 --num_envs 4 --viz kit
    python scripts/evaluate.py --task BalanceCar-Upright-v0 --num_envs 256 --controller lqr     # or pid

Quick check that an env builds (a few seconds): add ``--num_envs 64 --max_iterations 2`` to a train command.
"""

import gymnasium as gym

_TASKS = {
    "BalanceCar-Upright-v0": ("upright_env_cfg:UprightEnvCfg", "upright_ppo_cfg:UprightPPORunnerCfg"),
    "BalanceCar-Pitch-v0": ("pitch_env_cfg:PitchEnvCfg", "pitch_ppo_cfg:PitchPPORunnerCfg"),
    "BalanceCar-Velocity-v0": ("velocity_env_cfg:VelocityEnvCfg", "velocity_ppo_cfg:VelocityPPORunnerCfg"),
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
