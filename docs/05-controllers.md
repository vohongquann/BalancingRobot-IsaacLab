# 5. Controllers

Four controllers read the same IMU-based observation and write the same normalized wheel torque, so they run on one environment, one
set of pushes and one motor model and can be compared directly.

| | PID (cascaded) | LQR | RL, upright | RL, cascade |
|---|---|---|---|---|
| Package | [pid_control/](../src/Balance_Car_RL/car/pid_control/) | [lqr_control/](../src/Balance_Car_RL/car/lqr_control/) | [rl_control/](../src/Balance_Car_RL/car/rl_control/) | [rl_control/](../src/Balance_Car_RL/car/rl_control/) |
| Design | gains searched in the simulator | Riccati solution on the linear plant | PPO, one network | PPO, two stages, the first frozen ([04-training.md](04-training.md)) |
| Needs a model | only to start the search | yes ([03-dynamics.md](03-dynamics.md)) | no | no |
| Task | stay upright | stay upright | stay upright | hold a pitch, then track a speed |

The baselines rebuild the absolute wheel speed from the relative joint speeds, $\dot\psi=\tfrac12(\dot q_L+\dot q_R)+\dot\theta$
([02-sensing.md](02-sensing.md)), and use the state $z=[\theta,\dot\theta,\dot\psi]$; both wheels get the same command.

## 5.1 LQR (`lqr_control/lqr.py`)

Minimize $\sum_k z_k^\top Qz_k+Ru_k^2$ on the discrete plant of [03-dynamics.md](03-dynamics.md) §3.4, with $Q=\mathrm{diag}(100,\,1,\,0.1)$,
$R=10$. The discrete Riccati solution and gain:

$$P=A_d^\top PA_d-A_d^\top PB_d(R+B_d^\top PB_d)^{-1}B_d^\top PA_d+Q,\qquad K=(R+B_d^\top PB_d)^{-1}B_d^\top PA_d$$

$$u=-Kz=0.446\,\theta+0.0457\,\dot\theta+0.0094\,\dot\psi\quad[\text{N m}]$$

Closed-loop poles $|\lambda|=0.011,\ 0.819,\ 0.957$, inside the unit circle. The positive gain on $\dot\psi$ makes the robot lean forward when it
rolls forward fast, which brings it back. The action is $a=\operatorname{clip}\big(u/(2\tau_{stall}),-1,1\big)$.

## 5.2 Cascaded PID (`pid_control/cascade_pid.py`)

An outer loop turns the wheel speed into a pitch target (a robot rolling forward is asked to lean backward); an inner PID holds that pitch,
with the measured pitch rate as the D term:

$$\theta_{ref}=-k_v\dot\psi,\qquad e=\theta-\theta_{ref},\qquad u=k_pe+k_i\!\int_0^t e\,ds+k_d\dot\theta$$

Gains: $k_p=1.6$, $k_i=0.05$, $k_d=0.05$, $k_v=0.015$. Without the integral this is the state feedback $u=[k_p,\ k_d,\ k_pk_v]\,z$ with
poles $|\lambda|=0.73,\ 0.73,\ 0.98$ on the linear model. A first guess from a search on the *linear model* ($k_p=2.1$, $k_d=0.07$, $k_v=0.0094$) was
too stiff for the noisy IMU estimate: measured with the IMU observation it chattered at a mean wheel torque of 0.13 N m, against 0.002 N m for the gains above.
Those come from a search in the simulator with the IMU observation. They are a starting point, not an optimum.

## 5.3 RL

The upright policy is a plain PPO network ([04-training.md](04-training.md) §4.4). The cascade is the same idea as the PID: the speed policy
plays the outer loop and outputs a pitch, the frozen pitch policy plays the inner loop. Its result is in §5.5 and [04-training.md](04-training.md) §4.7.

## 5.4 Run and compare

```bash
python scripts/evaluate.py --task BalanceCar-Upright-v0 --num_envs 256 --controller lqr
python scripts/evaluate.py --task BalanceCar-Upright-v0 --num_envs 256 --controller pid [--pid_gains KP KI KD KV]
python scripts/evaluate.py --task BalanceCar-Upright-v0 --num_envs 256 --checkpoint logs/rsl_rl/balance_car_upright/<run>/model_299.pt
```

Add `--viz kit` to watch, `--json out.json` to save the numbers. Without `--checkpoint` the newest run in `logs/` is used, which may be an
untrained one. For each controller the same 768 episodes are run (random start tilt up to 0.25 rad, a push every 2 to 4 s, IMU noise and offsets)
and these are printed: fall rate, RMS pitch (true pitch), RMS error of the pitch estimate, mean wheel speed and torque, mean planar speed, largest
pitch, and for the tracking tasks the RMS, settled and median tracking error and the 90th percentile. Pass: fall rate at most 2 %, and RMS pitch at most 0.10 rad (upright) or the median tracking error
limits of [04-training.md](04-training.md) §4.6.

## 5.5 Comparison

**Upright task**: the same 768 episodes for each controller (random start tilt up to 0.25 rad, a push every 2 to 4 s, IMU noise and offsets, the
encoder-corrected pitch estimate).

| Controller | Falls | RMS pitch [rad] | Pitch estimate error [rad] | Mean wheel torque [N m] | Mean planar speed [m/s] |
|---|---|---|---|---|---|
| LQR | 0 | 0.033 | 0.030 | 0.00094 | 0.041 |
| Cascaded PID | 0.4 % | 0.050 | 0.064 | 0.00205 | 0.092 |
| RL, upright (300 iterations) | 0 | 0.043 | 0.053 | 0.00538 | 0.081 |

All three keep the robot up. LQR is the calmest and the most accurate: smallest pitch error, one fifth of the RL torque, and it drifts least. The PID
falls in 3 of 768 episodes. The RL policy uses the most torque. The pitch estimate is worst for the controller that moves the robot most, as
expected: motion disturbs the estimate ([02-sensing.md](02-sensing.md) §2.5). One seed, estimated masses: this says nothing yet about which
controller transfers best to the real robot.

**Tracking tasks** (RL only; the baselines have no speed command): pitch hold 0.017 rad and speed 0.038 m/s median error, see [04-training.md](04-training.md) §4.7.

## 5.6 Tuning

| Goal | LQR | PID | RL |
|---|---|---|---|
| Stiffer against tilt | raise the pitch weight in `Q` | raise `kp` (too high amplifies IMU noise) | raise the tracking or upright weight |
| Less drift | raise the wheel-speed weight in `Q` | raise `kv` | raise the wheel-speed penalty |
| Less torque, smoother | raise `R` | lower `kp` | raise the action-rate penalty |

After changing the URDF or a constant in `car_cfg.py`: rerun `design_lqr()` (nothing to retrain), retune the PID, and retrain the RL stages in order.

## 5.7 Limits

- The linear model holds within about ±0.3 rad; the RL policies are not limited this way.
- Both wheels get the same command, so there is no heading control; a push in y makes the robot yaw.
- The baselines regulate wheel speed to zero and have no speed command; only the RL cascade tracks a speed.
- One seed, estimated masses: the comparison says nothing yet about which controller transfers best to the real robot.
