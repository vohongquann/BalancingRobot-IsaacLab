"""Observation terms: what the real robot can measure, a raw IMU (``ImuPitchAndRate``, ``gyro_yaw_rate``) and the wheel
encoders (``wheel_speed_estimate``). Isaac Lab's ``joint_vel_rel``, ``last_action`` and ``generated_commands`` complete
the vectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from Balance_Car_RL.car import car_cfg
from Balance_Car_RL.car.estimation import imu_to_pitch, pitch_from_gravity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


class ImuPitchAndRate(ManagerTermBase):
    """Pitch [rad] and pitch rate [rad/s] as the real robot gets them: from a raw IMU, not from the simulator state.

    The IMU sensor gives the accelerometer (specific force) and gyroscope in the IMU frame (the control board axes).
    Noise and a per-episode offset are added with the LSM6DS33 numbers of ``car_cfg.py``; a complementary filter turns
    the samples into the gravity direction (``estimation.py``, ``guide/02_sensing.md``), which is rotated
    to the car frame with the known mounting. Output shape (num_envs, 2): ``[pitch, pitch_rate]``.

    The accelerometer is first corrected for the acceleration of the axle, which the encoders give (``estimation.py``,
    ``compensate_acceleration``): without it, driving would be taken for a tilt.

    The yaw rate of the same gyro sample (car z axis, with its noise and offset) is kept in ``yaw_rate`` for
    :func:`gyro_yaw_rate`.

    The filter runs once per policy step, and keeps running across a reset: ``reset()`` seeds the estimate from the
    simulator's true gravity direction (a reset can start mid-fall, unlike the real robot's power-up at rest, so
    restarting from one noisy accelerometer sample would throw that known state away). Until the IMU sensor produces
    its first real sample (its first tick after a reset reads zero), the estimate is held at that seeded value; the
    axle-acceleration low-pass is held at zero for the same steps, so it is not driven by a false speed jump computed
    from a zero gyro reading. The real robot's own filter (``ros/.../estimator.py``) starts cold, from rest, instead:
    it has no simulator truth to seed from.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        env.car_imu = self  # the action term of the outer stage reads the same estimate
        self._imu = env.scene[cfg.params["sensor_cfg"].name]
        self._robot = env.scene["robot"]
        rot = torch.tensor(car_cfg.IMU_R_CAR_FROM_IMU, dtype=torch.float32, device=env.device)
        self._r_car_from_imu = rot
        self._alpha = car_cfg.GRAVITY_FILTER_TAU_S / (car_cfg.GRAVITY_FILTER_TAU_S + env.step_dt)
        n = env.num_envs
        self._g = torch.zeros(n, 3, device=env.device)
        self._g[:, 2] = -1.0
        self._accel_bias = torch.zeros(n, 3, device=env.device)
        self._gyro_bias = torch.zeros(n, 3, device=env.device)
        self._cache = torch.zeros(n, 2, device=env.device)
        self._yaw_rate = torch.zeros(n, 1, device=env.device)
        self._speed_prev = torch.zeros(n, device=env.device)
        self._accel_lp = torch.zeros(n, device=env.device)
        self._accel_alpha = car_cfg.ACCEL_COMP_TAU_S / (car_cfg.ACCEL_COMP_TAU_S + env.step_dt)
        self._last_step = -1

    @property
    def cache(self) -> torch.Tensor:
        """The latest ``[pitch, pitch rate]``, shape (num_envs, 2), computed by the last observation."""
        return self._cache

    @property
    def yaw_rate(self) -> torch.Tensor:
        """The gyro's turn rate [rad/s] about the car's z axis of the last observation, shape (num_envs, 1)."""
        return self._yaw_rate

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        n = self._g[ids].shape[0]
        uniform = lambda limit: (2.0 * torch.rand(n, 3, device=self.device) - 1.0) * limit  # noqa: E731
        self._accel_bias[ids] = uniform(car_cfg.IMU_ACCEL_BIAS_MAX)
        self._gyro_bias[ids] = uniform(car_cfg.IMU_GYRO_BIAS_MAX)
        # the sensor buffers are cleared by the reset, so the estimate is seeded from the true gravity until the
        # IMU produces its first real sample (see __call__, has_sample)
        g_car = self._robot.data.projected_gravity_b.torch[ids]
        self._g[ids] = g_car @ self._r_car_from_imu
        self._cache[ids] = self._output(self._g[ids], torch.zeros(n, 3, device=self.device))
        self._yaw_rate[ids] = self._robot.data.root_ang_vel_b.torch[ids, 2:3]  # seeded like the pitch
        # the axle speed the next acceleration is measured from (encoders and the true pitch rate at the reset)
        wheel = self._robot.data.joint_vel.torch[ids].mean(dim=1)
        self._speed_prev[ids] = car_cfg.WHEEL_RADIUS_M * (wheel + self._robot.data.root_ang_vel_b.torch[ids, 1])
        self._accel_lp[ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        if env.common_step_counter == self._last_step:  # observations may be computed more than once per step
            return self._cache
        self._last_step = env.common_step_counter
        n = env.num_envs
        accel = self._imu.data.lin_acc_b.torch + self._accel_bias
        gyro = self._imu.data.ang_vel_b.torch + self._gyro_bias
        accel = accel + torch.randn(n, 3, device=env.device) * car_cfg.IMU_ACCEL_NOISE_STD
        gyro = gyro + torch.randn(n, 3, device=env.device) * car_cfg.IMU_GYRO_NOISE_STD
        has_sample = self._imu.data.lin_acc_b.torch.norm(dim=-1) > 1e-3
        # axle acceleration from the encoders and the gyro, smoothed, and removed from the accelerometer; held at its
        # reset value (zero speed change, zero low-pass) until there is a real sample, so a zero gyro reading before
        # the first sample is never mistaken for a sudden change of the true, already-known axle speed
        speed = car_cfg.WHEEL_RADIUS_M * (
            self._robot.data.joint_vel.torch.mean(dim=1) + (gyro @ self._r_car_from_imu.T)[:, 1]
        )
        raw = (speed - self._speed_prev) / env.step_dt
        accel_lp = self._accel_alpha * self._accel_lp + (1.0 - self._accel_alpha) * raw
        self._accel_lp = torch.where(has_sample, accel_lp, self._accel_lp)
        self._speed_prev = torch.where(has_sample, speed, self._speed_prev)
        g_new, pitch, pitch_rate = imu_to_pitch(
            self._g, gyro, accel, self._accel_lp, self._r_car_from_imu, env.step_dt, self._alpha
        )
        self._g = torch.where(has_sample.unsqueeze(-1), g_new, self._g)
        self._cache = torch.where(has_sample.unsqueeze(-1), torch.stack([pitch, pitch_rate], dim=1), self._cache)
        yaw_rate = (gyro @ self._r_car_from_imu.T)[:, 2:3]
        self._yaw_rate = torch.where(has_sample.unsqueeze(-1), yaw_rate, self._yaw_rate)
        return self._cache

    def _output(self, g_imu: torch.Tensor, gyro_imu: torch.Tensor) -> torch.Tensor:
        pitch = pitch_from_gravity(g_imu @ self._r_car_from_imu.T)
        pitch_rate = (gyro_imu @ self._r_car_from_imu.T)[:, 1]
        return torch.stack([pitch, pitch_rate], dim=1)


def axle_speed(env: ManagerBasedRLEnv, joint_ids) -> torch.Tensor:
    """Forward speed of the axle [m/s] from the encoders and the IMU pitch rate, shape (num_envs,).

    ``v = r (mean wheel speed + pitch rate)``: the encoders measure the wheels relative to the body, the pitch rate
    (``ImuPitchAndRate``, computed before this term in the same group) turns it into the absolute wheel speed. What the
    robot knows of its speed: the simulator's own velocity is not available on the real car.
    """
    car_imu = getattr(env, "car_imu", None)
    if car_imu is None:
        raise RuntimeError("no ImuPitchAndRate term in the observations: the speed estimate needs its pitch rate")
    wheel = env.scene["robot"].data.joint_vel.torch[:, joint_ids].mean(dim=1)
    return car_cfg.WHEEL_RADIUS_M * (wheel + car_imu.cache[:, 1])


def gyro_yaw_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Turn rate [rad/s] from the gyro (``ImuPitchAndRate``, computed before this term in the same group), shape
    (num_envs, 1). Not from the encoders: the wheels scrub while turning, and the encoder estimate
    ``r (w_right - w_left) / track`` read about 11 % more than the true turn rate."""
    car_imu = getattr(env, "car_imu", None)
    if car_imu is None:
        raise RuntimeError("no ImuPitchAndRate term in the observations: the turn rate is its gyro's")
    return car_imu.yaw_rate


def wheel_speed_estimate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:  # noqa: B008
    """``axle_speed`` of the wheels of ``asset_cfg``, shape (num_envs, 1)."""
    return axle_speed(env, asset_cfg.joint_ids).unsqueeze(-1)
