# Review toàn bộ code Balance_Car_RL, từ dễ đến khó

File này là lộ trình để tự review toàn bộ hệ thống, cùng cách viết với `Drone_RL/docs/code_review.md`. Đi theo thứ tự:
mỗi cấp chỉ dùng những gì đã kiểm tra ở cấp trước, nên khi tới cấp khó (một bước mô phỏng của stage velocity, có policy
pitch đóng băng chạy bên trong action) bạn đã chắc chắn mọi mảnh bên dưới đúng.

- Mỗi mục có: **file / hàm / class**, **đọc để hiểu gì**, **kiểm tra gì** (checkbox), **bẫy dễ sai**, và khi có thể
  một **lệnh kiểm tra nhanh** (vài giây, không chạy simulator, không train).
- Số dòng ghi theo code ngày 2026-10-05; nếu lệch vài dòng thì tìm theo tên hàm.
- Tài liệu song song: `guide/00..06` (local, ẩn khỏi git). Cột "Guide" ở mỗi cấp nói nên mở trang nào bên cạnh.
- Các phát hiện nằm ở **Phụ lục B** (có cột trạng thái: đã sửa / còn mở); mỗi cấp đánh dấu chúng là **[B<số>]**.

Tất cả lệnh Python dưới đây chạy bằng:

```bash
PY=~/miniconda3/envs/env_isaaclab/bin/python
cd ~/Documents/GitHub/Balance_Car_RL
```

---

## Mục lục

| Cấp | Nội dung | Độ khó | Cần Isaac? | Guide |
|---|---|---|---|---|
| 0 | Chuẩn bị: quy ước, bản đồ phụ thuộc, trạng thái hiện tại | ★ | không | 00 |
| 1 | Hằng số vật lý, URDF, gắn IMU | ★ | không | 01 |
| 2 | Ước lượng pitch từ IMU thô (torch và numpy) | ★★ | không | 02 |
| 3 | Bộ điều khiển cổ điển: PID, mô hình tuyến tính, LQR | ★★ | không | 03, 05 |
| 4 | Đóng băng một stage: `frozen_policy.py` | ★★ | không | 04 |
| 5 | Các term MDP chạy trong Isaac Lab: observation, command, action, reward | ★★★★ | có | 02, 04 |
| 6 | Cấu hình task, PPO, đăng ký | ★★★ | có (chỉ import) | 04 |
| 7 | Script: `train_cascade.py`, `evaluate.py`, công cụ | ★★★ | có | 04, 05 |
| 8 | Triển khai ROS 2 | ★★★ | không | 06 |
| 9 | Toàn hệ thống theo thời gian: một bước của stage velocity | ★★★★★ | có | 04 |
| 10 | Test: cái gì đã được bảo vệ, cái gì chưa | ★★ | không | — |
| Phụ lục | File ngoài luồng, vấn đề mở, bảng theo dõi | — | — | — |

---

## Cấp 0 — Chuẩn bị

### 0.1 Chạy test trước khi đọc

```bash
PYTHONPATH=ros/src/car_bridge $PY -m pytest tests ros/src/car_bridge/test/test_policy.py \
    ros/src/car_bridge/test/test_estimator.py -q          # kỳ vọng: 60 passed, 4 skipped
```

4 test bị skip là đúng: chúng chờ policy đã train (`frozen/pitch`, `frozen/velocity`) và `models/balance_car_policy.onnx`, đã bị xoá có chủ ý (0.4).
Nếu có test fail, dừng lại sửa trước: mọi thứ bên dưới giả định các test này xanh.

### 0.2 Quy ước dùng khắp nơi (nhớ trước khi đọc bất kỳ file nào)

| Quy ước | Giá trị | Kiểm ở đâu |
|---|---|---|
| Khung xe (car) | x trước, y trái, z lên, gốc trên trục bánh | docstring `car_cfg.py`, `build_balboa_urdf.py` |
| Pitch θ | **dương = nghiêng về trước** (+x) | `estimation.pitch_from_gravity`, `rewards.body_pitch` |
| Trọng lực trong khung thân | `g_b = (sin θ, 0, −cos θ)` khi nghiêng θ quanh y | `projected_gravity_b`, `atan2(g_x, −g_z)` |
| Quaternion | **(x, y, z, w)** (Isaac Lab 3.0) | `IMU_QUAT_XYZW`, `test_imu_quaternion_matches_the_rotation_matrix` |
| Ma trận gắn IMU | `v_car = R @ v_imu`, `R = IMU_R_CAR_FROM_IMU` | `imu_mount.json`, `test_mount_is_a_rotation` |
| Tốc độ khớp bánh | **tương đối với thân**: `q̇ = ψ̇ − θ̇`, nên `ψ̇ = q̇ + θ̇` | `lqr_control/model.py`, `cascade_pid.act`, `wheel_speed_estimate` |
| Tốc độ trục | `v = r · ψ̇`, r = 0.04 m | `wheel_speed_estimate`, ROS `estimator.update` |
| Dữ liệu robot | `robot.data.<x>` là `ProxyArray`, phải `.torch` | khắp `mdp/` |
| Action bánh | [−1, 1] × stall torque 0.209 N m | `ActionsCfg` (`car_env_cfg.py:63`), ROS `TORQUE_SCALE_NM` |
| Physics / policy | 200 Hz / 50 Hz (decimation 4), episode 10 s = 500 bước | `car_env_cfg.py:33-39` |
| Trọng số reward | Isaac Lab nhân với `step_dt` = 0.02, nên return tối đa của upright là (1 + 2)·500·0.02 = 30 | `guide/04` §4.3 |

### 0.3 Bản đồ phụ thuộc (ai import ai)

```text
car_cfg.py  (hằng số, CAR_CFG; đọc imu_mount.json)  ◄── tools/build_balboa_urdf.py (đọc BASE/WHEEL_MASS, ghi URDF + imu_mount.json)
   │
   ├──► estimation.py (gravity_step, compensate_acceleration, imu_to_pitch)      ≡ ros/.../estimator.py (numpy, bản sao)
   │         │
   │         ▼
   ├──► mdp/observations.py (ImuPitchAndRate → env.car_imu; axle_speed, wheel_speed_estimate)
   ├──► mdp/commands.py (ScalarCommand, speed guard)
   ├──► mdp/rewards.py (upright_exp, wheel_vel_l2, yaw_rate_l2, pitch/speed_error_*)
   ├──► mdp/actions/frozen_policy.py (FROZEN_DIR, STAGE_IO, _normalized, contract_fingerprint, file_sha256, load_frozen)
   │         ▼
   │    mdp/actions/pitch_action.py (FrozenPitchAction: đọc env.car_imu và observations.axle_speed)
   │
   ├──► pid_control/pid.py (PID, chép từ Drone) ──► pid_control/cascade_pid.py (CascadePID)
   ├──► lqr_control/model.py (PlantParams.from_urdf, linear_model) ──► lqr_control/lqr.py (design_lqr, LQRController)
   ▼
rl_control/car_env_cfg.py (scene, ActionsCfg, PolicyCfg, EventCfg, RewardsCfg, TerminationsCfg, CarEnvCfg)
   ◄── upright_env_cfg.py, pitch_env_cfg.py ◄── velocity_env_cfg.py (lấy PITCH_TARGET_MAX_RAD, SPEED_GUARD_M_S)
rl_control/agents/car_ppo_cfg.py ◄── upright_/pitch_/velocity_ppo_cfg.py
rl_control/__init__.py (_TASKS, gym.register 3 task) ◄── car/__init__.py ◄── tasks/__init__.py (Isaac Lab tìm task ở đây)

scripts/train_cascade.py (gọi isaaclab train/play, evaluate.py, ghi frozen/<stage>/) ── scripts/evaluate.py (+ PID, LQR)
ros/src/car_bridge: bridge_node.py ── estimator.py, policy.py (ONNX của task Upright)
```

Đọc theo chiều mũi tên từ trên xuống = đúng thứ tự của file này.

### 0.4 Trạng thái hiện tại (quan trọng khi review)

- **Không còn policy nào.** Ngày 2026-10-05 đã xoá có chủ ý `frozen/pitch`, `frozen/velocity`, `models/balance_car_policy.onnx` và các run
  `logs/rsl_rl/balance_car_*`, vì chúng học với action không kẹp, trên URDF 3 hộp collision, không có phạt quay yaw, và guard tốc độ đọc vận tốc thật.
  Các test chờ policy bị skip (4 test). Train lại: `python scripts/train_cascade.py` (pitch, rồi velocity) và task Upright (cho ROS).
- Mọi vấn đề của lần review trước (B1–B10) đã được xử lý hoặc có kết luận đo được, xem Phụ lục B.
- Những gì đã đổi trong ngày 2026-10-05 (review lại các chỗ này trước cũng được):

| # | Đổi | File |
|---|---|---|
| 1 | Bố cục theo Drone: đăng ký task bằng `_TASKS` trong `rl_control/__init__.py`, `car/__init__.py` chỉ import | `car/__init__.py`, `rl_control/__init__.py` |
| 2 | `common.py` → `car_env_cfg.py` với lớp cơ sở; 3 task kế thừa (`UprightEnvCfg`, `PitchEnvCfg`, `VelocityEnvCfg`) | `rl_control/*_env_cfg.py` |
| 3 | `rsl_rl_ppo_cfg.py` → `car_ppo_cfg.py` + `<task>_ppo_cfg.py`; thêm `BoundedGaussianCfg` (std trong 0.02–0.5), `clip_actions = 1.0`, `init_std` 0.5 | `rl_control/agents/` |
| 4 | `mdp/actions.py` → `mdp/actions/{frozen_policy,pitch_action}.py`; `load_frozen` kiểm kích thước theo `STAGE_IO` | `mdp/actions/` |
| 5 | `mdp/__init__` bỏ `lazy_export`/`.pyi`; term của Isaac Lab dùng qua `isaac_mdp` | `mdp/__init__.py`, các env cfg |
| 6 | Reward viết lại theo Drone, đổi tên hàm và term; giá trị không đổi (so trên dữ liệu ngẫu nhiên: lệch 0.0) | `mdp/rewards.py` |
| 7 | `PID` chép từ Drone; `CascadePID` dùng hai `PID`; kết quả trùng bản cũ (lệch 0.0 qua 300 bước có reset) | `pid_control/` |
| 8 | Policy pitch đóng băng kẹp output về [−1, 1]; ROS đưa giá trị đã kẹp vào `last_action` | `pitch_action.py:89`, `ros/.../policy.py` |
| 9 | Chép `summarize_training_run.py` từ Drone | `car/tools/` |
| 10 | URDF: **một** collision hull lồi thay 3 hộp chồng nhau (B7) | `balboa.urdf`, `meshes/body_collision.stl`, `build_balboa_urdf.py:177-185` |
| 11 | Tài liệu: `.md` sang `guide/` (ẩn bằng `.git/info/exclude`), `docs/` chỉ còn PDF nguồn | `guide/`, `docs/` |
| 12 | **B1/B2:** fingerprint băm AST (bỏ chú thích) của 12 file, gồm URDF, `imu_mount.json`, collision mesh, rewards, PPO; `load_frozen` cảnh báo cả khi thiếu fingerprint; `train_cascade` từ chối train trên policy cũ (`--allow-stale`); velocity ghi `needs_sha256` | `frozen_policy.py`, `train_cascade.py` |
| 13 | **B3:** guard tốc độ của `FrozenPitchAction` dùng ước lượng `axle_speed` (encoder + IMU), không dùng vận tốc simulator | `observations.py:116`, `pitch_action.py:81` |
| 14 | **B4:** phạt quay yaw `yaw_rate_l2` (−0.05) trong `RewardsCfg`; `evaluate.py` in `RMS turn rate` | `rewards.py:33`, `car_env_cfg.py:133` |
| 15 | **B9:** `axle_speed` báo lỗi rõ khi thiếu term `ImuPitchAndRate` | `observations.py:116` |
| 16 | **B10:** `evaluate.py` bắt buộc `--checkpoint` cho policy | `scripts/evaluate.py` |
| 17 | **B8:** `tests/test_mdp_terms.py` (guard của `ScalarCommand`, `FrozenPitchAction` với stub, `axle_speed`, `yaw_rate_l2`) | `tests/` |
| 18 | Code thừa: `gravity_from_accel` (chỉ test dùng) chuyển vào test; bỏ ignore `.pyi`; bỏ USD cũ trong `assets/data/balboa/balboa/`; `FrozenPitchActionCfg` bỏ giá trị mặc định chép tay (nay bắt buộc); `STAGES` của `frozen_policy` đổi thành `STAGE_IO` (trùng tên với `train_cascade.STAGES`) | nhiều file |

---

## Cấp 1 — Hằng số vật lý, URDF, gắn IMU (★)

Mục tiêu: mọi con số và mọi dấu ở tầng thấp nhất đúng. Sai ở đây làm hỏng mọi tầng trên mà không báo lỗi.

### 1.1 `src/Balance_Car_RL/car/car_cfg.py` (144 dòng)

Đọc cùng `guide/01_robot.md` (nguồn từng số, mục 1.2–1.5) và các PDF trong `docs/hardware/balboa/`.

| Dòng | Tên | Kiểm tra |
|---|---|---|
| 25 | `WHEEL_RADIUS_M` | 0.040 m (bánh 80 mm, bản vẽ kit) |
| 30, 34 | `BASE_MASS_KG`, `WHEEL_MASS_KG` | 0.27 và 0.020 kg: **ước lượng**, chưa cân |
| 40 | `GEARMOTOR_RATIO` | 3344/65 = 51.45 (gear ratio chart) |
| 43 | `EXTERNAL_RATIO` | 49/17 = 2.88 |
| 46 | `TOTAL_RATIO` | 148.29 |
| 51 | `WHEEL_STALL_TORQUE_NM` | 0.74 kg cm × 0.0980665 × 49/17 = **0.2092 N m** (hiệu suất hộp số 100 %) |
| 54 | `WHEEL_NO_LOAD_SPEED_RAD_S` | 650 rpm × 2π/60 × 17/49 = **23.62 rad/s** → 0.94 m/s ở vành |
| 57, 60 | `MOTOR_ROTOR_INERTIA`, `WHEEL_ARMATURE` | 7e-9 (ước lượng) × 148.3² = **1.54e-4 kg m²**, gấp ~10 lần quán tính bánh **[B6]** |
| 67-76 | `IMU_POS_M`, `IMU_QUAT_XYZW`, `IMU_R_CAR_FROM_IMU` | đọc từ `imu_mount.json` lúc import |
| 79-95 | nhiễu, offset IMU | mật độ nhiễu × √52 Hz; offset = 10 % số datasheet (giả định hiệu chuẩn) |
| 98, 102 | `GRAVITY_FILTER_TAU_S`, `ACCEL_COMP_TAU_S` | 1.0 s và 0.05 s; ROS có bản sao (test kiểm) |
| 107-143 | `CAR_CFG` | URDF spawn cao r + 1 mm; `DCMotorCfg` stiffness = damping = 0 (điều khiển mô-men thuần) |

Checklist:
- [ ] Mọi số khớp `guide/01_robot.md` mục 1.2 và PDF tương ứng (`docs/readme.md` liệt kê file nào cho số nào).
- [ ] Hiểu `DCMotorCfg`: mô-men bị cắt theo đường thẳng mô-men–tốc độ, stall ở 0 rad/s, bằng 0 ở 23.6 rad/s. Tốc độ dùng
  là tốc độ **khớp** (tương đối với thân), đúng với động cơ gắn trên thân.
- [ ] `WHEEL_ARMATURE` là quán tính rotor quy về bánh, cộng vào khớp: nó chi phối động học bánh nhiều hơn quán tính bánh.
  Số rotor 7e-9 kg m² không có nguồn.
- [ ] `_KGCM_TO_NM = 0.0980665` đúng (1 kgf·cm = 0.0980665 N m).

Lệnh kiểm tra nhanh:

```bash
$PY -c "
from Balance_Car_RL.car import car_cfg as C
print('stall', C.WHEEL_STALL_TORQUE_NM)            # 0.20917
print('no-load', C.WHEEL_NO_LOAD_SPEED_RAD_S)      # 23.615 rad/s
print('armature', C.WHEEL_ARMATURE)                # 1.539e-4
print('accel noise', C.IMU_ACCEL_NOISE_STD, 'gyro noise', C.IMU_GYRO_NOISE_STD)   # 6.36e-3, 8.81e-4
print('IMU pos', C.IMU_POS_M)                      # (-0.0102, 0.0007, 0.0258)
"
```

Bẫy:
- `IMU_*` đọc `imu_mount.json` lúc chạy, nên sửa JSON đổi hành vi mà không đổi `car_cfg.py`: `imu_mount.json` nằm trong danh sách fingerprint **[B2: đã sửa]**.
- `FrozenPitchActionCfg.torque_scale` và `.pitch_scale` không còn giá trị mặc định chép tay: env cfg phải truyền
  (`test_stage_2_uses_the_range_and_scale_of_stage_1` kiểm chúng bằng `car_cfg.WHEEL_STALL_TORQUE_NM` và `PITCH_TARGET_MAX_RAD`).

### 1.2 `tools/build_balboa_urdf.py` (257 dòng) và `assets/data/balboa/`

Cần `cad/` và `open3d`; chạy lại được (`python tools/build_balboa_urdf.py`, vài giây) và cho kết quả **trùng từng byte** với URDF và mesh đang commit,
trừ phần collision. Đọc logic:

| Bước | Dòng | Làm gì | Kiểm tra |
|---|---|---|---|
| 1 | 100-115 | đưa mọi mesh về khung CAD, bỏ offset −19.86 mm bị đếm 2 lần của board/buzzer (`EXPORT_OFFSET_BUG_LINK`) | trục = trung điểm 2 khớp bánh |
| 2 | 117-134 | đổi trục CAD → xe (`swap`), chia khối lượng thân theo **thể tích**, quay quanh trục để COM nằm thẳng trên trục (`alpha`) | COM x ≈ 0 |
| 3 | 136-142 | quán tính thân = tổng các bao lồi (mật độ đều) + định lý trục song song | |
| 4 | 177-185 | collision thân: **một hull lồi** của chassis + battery cover + control board (`COLLISION_PARTS`), giảm còn 200 tam giác, ghi `meshes/body_collision.stl` | xem checklist |
| 5 | 186-223 | bánh: hình trụ r = 40 mm, rộng 11.75 mm; khớp `continuous`, trục +y cho cả hai bên | `test_wheel_joints_drive_the_same_direction` |
| 6 | 227-236 | `imu_mount.json`: vị trí = tâm bbox board, hướng = trục board (đã nghiêng theo thân) | `test_mount_is_a_rotation` |

Checklist:
- [ ] COM thân 19.9 mm trên trục (`balboa.urdf:6`); nó phụ thuộc giả định "khối lượng tỉ lệ thể tích" **[B11]**.
- [ ] `base_link` có đúng 1 `<collision>` (mesh), mỗi bánh 1 hình trụ: tổng 3. Isaac Lab nhập mesh với `collision_type="Convex Hull"` (mặc định):
  prim `.../base_link/body_collision/body_collision` có `physics:approximation = convexHull`, 102 điểm.
- [ ] Hull phủ mesh: 1.43 % đỉnh của 3 chi tiết nằm ngoài hull, tối đa 0.63 mm (hộp cũ: 15.9 %, 8.7 mm). Điểm thấp nhất của hull = điểm thấp nhất
  của mesh = −29.2 mm, nên khoảng hở gầm 10.8 mm giữ nguyên.
- [ ] Hull rộng 107 mm theo y, vào khoảng mặt trong của bánh (bánh ở |y| = 53.5 ± 5.9 mm): chồng lấn ~6 mm với bánh, vô hại vì `enabled_self_collisions=False`
  (hộp cũ 104 mm cũng chồng ~4 mm).
- [ ] Khối lượng 0.27 + 2 × 0.02 kg là ước lượng: datasheet gearmotor, user's guide Balboa, trang sản phẩm Pololu đều **không công bố** khối lượng nào (đã kiểm).
  Cân robot thật rồi sửa `BASE_MASS_KG`, `WHEEL_MASS_KG` và chạy lại script.

Lệnh:

```bash
grep -c "<collision>" src/Balance_Car_RL/assets/data/balboa/balboa.urdf      # 3
$PY -m pytest tests/test_car_cfg.py -q                                       # 8 passed
$PY -c "
import trimesh
h = trimesh.load('src/Balance_Car_RL/assets/data/balboa/meshes/body_collision.stl')
print(len(h.vertices), h.bounds[0, 2] * 1000)                                # 102, -29.2
"
```

---

## Cấp 2 — Ước lượng pitch từ IMU thô (★★)

**Câu hỏi của cấp này:** policy không thấy pitch thật của simulator; nó thấy pitch ước lượng từ gyro + gia tốc kế
có nhiễu, giống robot thật. Ước lượng đó đúng tới đâu, và bản torch (sim) có trùng bản numpy (ROS) không.

### 2.1 `src/Balance_Car_RL/car/estimation.py` (87 dòng)

| Hàm | Dòng | Làm gì |
|---|---|---|
| `gravity_step` | 10 | dự đoán bằng gyro `g − (ω × g)·dt`, trộn với `−f/|f|` theo `alpha`; không có mẫu gia tốc thì chỉ dùng gyro; chuẩn hoá |
| `compensate_acceleration` | 30 | trừ gia tốc trục `a·R[0]` (trục x của xe, viết trong khung IMU) khỏi gia tốc kế |
| `pitch_from_gravity` | 50 | `atan2(g_x, −g_z)` trong khung xe |
| `imu_to_pitch` | 55 | bù gia tốc → lọc → pitch, pitch rate = `(R ω)_y` |

Checklist:
- [ ] Dấu `ġ = −ω × g`: một vector cố định trong thế giới quay ngược với thân, nhìn từ thân.
- [ ] Gia tốc kế đo lực riêng `f = a − g_vec`: nằm yên đọc +9.81 hướng lên, nên `g_meas = −f/|f|`.
- [ ] Tăng tốc về trước cộng `+a` vào `f_x` → nếu không bù sẽ đọc thành nghiêng **về sau** (0.5 m/s² ≈ 0.05 rad).
- [ ] `alpha = τ/(τ + dt)` = 0.98 ở 50 Hz: đúng với mọi tần số, nên sim và robot thống nhất.
- [ ] **Không** bù gia tốc tiếp tuyến do IMU ở 25.8 mm trên trục (`θ̈·h`): 50 rad/s² cho 1.3 m/s², tức 0.13 rad sai số
  tức thời trước khi lọc (`guide/02` §2.5). Đã đo: sai số ước lượng chỉ 0.010 rad dưới LQR, 0.011 dưới PID, 0.013 khi cộng thêm nhiễu mô-men ngẫu nhiên
  (gấp 4 lần mô-men trung bình của RL), nên chưa cần bù **[B5]**.

Lệnh (xe nằm yên nghiêng +10°, kỳ vọng 0.1745 rad):

```bash
$PY -c "
import math, torch
from Balance_Car_RL.car import car_cfg as C
from Balance_Car_RL.car.estimation import imu_to_pitch
R = torch.tensor(C.IMU_R_CAR_FROM_IMU, dtype=torch.float32); th = math.radians(10)
f_imu = torch.tensor([[-math.sin(th)*9.81, 0.0, math.cos(th)*9.81]]) @ R
_, pitch, _ = imu_to_pitch(-f_imu/f_imu.norm(), torch.zeros(1,3), f_imu, torch.zeros(1), R, 0.02, 0.98)
print(float(pitch))
"
```

### 2.2 `ros/src/car_bridge/car_bridge/estimator.py` — bản numpy

- `R_CAR_FROM_IMU`, `TAU_S`, `ACCEL_COMP_TAU_S`, `WHEEL_RADIUS_M` là **bản chép tay** của `car_cfg.py`
  (`test_ros_and_sim_share_the_constants` kiểm).
- `GravityEstimator.update` (dòng 48 của file) làm cùng việc với `imu_to_pitch` + phần bù gia tốc của `ImuPitchAndRate`.

Checklist:
- [ ] `test_torch_and_numpy_filters_agree` và `test_acceleration_compensation_matches_between_torch_and_numpy` xanh.
- [ ] Khác biệt **có chủ ý**: sim khởi tạo ước lượng từ trọng lực **thật** khi reset (một reset có thể bắt đầu đang ngã);
  ROS khởi tạo từ mẫu gia tốc kế đầu tiên, nên phải bật node khi xe đứng yên.
- [ ] Bước đầu ROS: `raw = 0` (chưa có tốc độ trước), giống sim giữ low-pass ở 0 tới mẫu đầu.

Lệnh: `PYTHONPATH=ros/src/car_bridge $PY -m pytest tests/test_estimation.py ros/src/car_bridge/test/test_estimator.py -q`.

---

## Cấp 3 — Bộ điều khiển cổ điển: PID, mô hình tuyến tính, LQR (★★)

Đây là baseline chạy trên cùng observation 6 số của task Upright: `[pitch, pitch rate, q̇_L, q̇_R, last_L, last_R]`.

### 3.1 `pid_control/pid.py` (45 dòng) — class `PID`, chép nguyên từ Drone

- `update(error, dt)`: I tích luỹ rồi kẹp (`int_limit`), D = (e − e_prev)/dt, lần gọi đầu không có cú đá D, kẹp đầu ra.
- `reset(None)` xoá hẳn trạng thái; `reset(env_ids)` đặt 0 các hàng đó.

Checklist:
- [ ] `int_limit` có đơn vị của ∫e dt, không phải của `ki·∫e dt`.
- [ ] Sau `reset(env_ids)`, `prev_error` các hàng = 0, nên có cú đá D nhỏ. Ở xe **không ảnh hưởng**: cả hai `PID` của
  `CascadePID` có `kd = 0` (D lấy từ pitch rate đo được).

Lệnh: `$PY -m pytest tests/test_pid.py -q`.

### 3.2 `pid_control/cascade_pid.py` (51 dòng) — `CascadePID`

```text
ψ̇ = ½(q̇_L + q̇_R) + θ̇ ──► speed: PID(kp = kv) trên (0 − ψ̇) ──► θ* ──► pitch: PID(kp, ki, int_limit) trên (θ − θ*)
                                                                          + kd·θ̇  ──► τ tổng ──► /(2·stall), kẹp ±1, cho 2 bánh
```

Checklist:
- [ ] Sai số pitch là `θ − θ*` (đo − đặt), ngược quy ước "setpoint − measurement" của `PID`: xe nghiêng trước phải chạy
  bánh về trước, nên mọi gain dương.
- [ ] Đang lăn về trước (ψ̇ > 0) cho θ* < 0: xe được yêu cầu ngả về sau để hãm.
- [ ] `dt = 0.02` cố định, phải bằng `step_dt` của env (50 Hz).
- [ ] Gain mặc định (1.6, 0.05, 0.05, 0.015) tìm bằng search trong sim (`guide/05` §5.2).

Lệnh: `$PY -m pytest tests/test_pid_control.py -q`.

### 3.3 `lqr_control/model.py` (88 dòng) — mô hình tuyến tính

- `PlantParams.from_urdf` (dòng 41): khối lượng, COM, `iyy` thân, `izz` bánh (trục bánh là z cục bộ của link bánh),
  `2·WHEEL_ARMATURE`.
- `linear_model` (dòng 64): toạ độ `q = [ψ, θ]`, ma trận khối lượng có số hạng rotor `[J_r, −J_r; −J_r, J_r]` vì rotor
  quay theo **khớp tương đối** `ψ − θ`; lực tổng quát của mô-men khớp là `B = [1, −1]`; trọng lực `m g L θ`.

Checklist:
- [ ] Năng lượng rotor `½ J_r (ψ̇ − θ̇)²` cho đúng các số hạng chéo `−J_r`.
- [ ] Trạng thái `z = [θ, θ̇, ψ̇]` (ψ là toạ độ tuần hoàn, nên điều khiển tốc độ bánh, không điều khiển vị trí).
- [ ] Cực hở: một cực không ổn định ≈ +9.2 rad/s.

### 3.4 `lqr_control/lqr.py` (51 dòng)

- `design_lqr` rời rạc hoá ZOH ở 0.02 s, giải Riccati rời rạc, `Q = diag(100, 1, 0.1)`, `R = 10`.
- `LQRController.act`: `u = −K z`, chia đôi cho 2 bánh, chuẩn hoá theo stall, kẹp ±1.

Lệnh kiểm tra nhanh:

```bash
$PY -c "
import numpy as np
from scipy.signal import cont2discrete
from Balance_Car_RL.car.lqr_control import PlantParams, linear_model, design_lqr
a, b = linear_model(PlantParams.from_urdf()); print('open-loop', np.round(np.linalg.eigvals(a), 3))   # 0, ±9.207
K = design_lqr(); print('K', np.round(K, 4))                                                          # [-0.446 -0.0457 -0.0094]
ad, bd, *_ = cont2discrete((a, b, np.eye(3), np.zeros((3, 1))), 0.02, method='zoh')
print('closed |eig|', np.round(np.abs(np.linalg.eigvals(ad - bd @ K[None])), 3))                      # 0.011 0.819 0.957
"
```

Bẫy:
- K **âm** là đúng: `u = −K z` nên nghiêng trước (θ > 0) cho mô-men dương (bánh về trước).
- Mô hình bỏ qua giới hạn mô-men theo tốc độ (DCMotor) và trễ; `guide/03` §3.5 liệt kê.

---

## Cấp 4 — Đóng băng một stage: `mdp/actions/frozen_policy.py` (★★)

**Câu hỏi của cấp này:** stage pitch đã train xong được "đóng băng" thành file nào, stage velocity đọc nó ra sao, và làm sao biết nó đã cũ.

Quy trình: `train_cascade.py` train stage → `isaaclab play` xuất `exported/policy.pt` (TorchScript, normaliser bên
trong) → chỉ khi đạt tiêu chí đánh giá mới chép vào `rl_control/frozen/<stage>/` kèm `policy.onnx` và `meta.json`.
Stage trên **không bao giờ đọc `logs/`**. Thư mục `frozen/` hiện **không tồn tại** (0.4); `train_cascade.py` tự tạo.

| Tên | Dòng | Làm gì |
|---|---|---|
| `FROZEN_DIR` | 20 | `car/rl_control/frozen/` |
| `STAGE_IO` | 24 | `{"pitch": (7, 2), "velocity": (5, 1)}`: kích thước obs/action |
| `_CONTRACT_FILES` | 27 | 12 file tạo nên "hợp đồng" của stage pitch: robot (`car_cfg.py`, `balboa.urdf`, `imu_mount.json`, `body_collision.stl`), cái policy thấy (`estimation.py`, `observations.py`, `commands.py`), cái nó được train (`rewards.py`, `car_env_cfg.py`, `pitch_env_cfg.py`, `car_ppo_cfg.py`, `pitch_ppo_cfg.py`) |
| `_normalized` | 50 | file `.py` → cây cú pháp bỏ chuỗi docstring (kể cả docstring của hằng số) và chú thích; file khác → nguyên byte |
| `frozen_policy_path` | 64 | `<frozen_dir>/<stage>/policy.pt` |
| `file_sha256` | 69 | băm một file (dùng cho `needs_sha256` của velocity) |
| `contract_fingerprint` | 73 | SHA-256 trên (tên file + nội dung chuẩn hoá); `None` cho stage không có entry (velocity) |
| `load_frozen` | 90 | load, `eval()`, tắt gradient, kiểm kích thước theo `STAGE_IO`, **cảnh báo khi fingerprint khác hoặc thiếu** |

Checklist:
- [ ] Kích thước lấy từ `mlp.<i>.weight` đầu và cuối: khoá của TorchScript do rsl_rl 5.5.1 xuất (`mlp.0`, `mlp.2`, `mlp.4`);
  nâng cấp rsl_rl thì kiểm lại tên khoá. `test_load_frozen_*` dựng actor giả đúng bố cục này nên chạy được không cần policy thật.
- [ ] Sửa chú thích, docstring hay format trong 12 file **không** làm policy bị coi là cũ; sửa một hằng số, một trọng số reward, `clip_actions`, URDF thì có
  (`test_fingerprint_ignores_comments_and_docstrings_but_not_code`, `test_fingerprint_covers_what_the_policy_depends_on`). **[B1, B2: đã sửa]**
- [ ] Hai tầng chặn dùng policy cũ: `load_frozen` in `[WARN]` (kể cả khi `meta.json` không có fingerprint, trường hợp của policy đóng băng trước đây);
  `train_cascade.py` **dừng** thay vì cảnh báo, trừ khi `--allow-stale`.
- [ ] Stage velocity ghi `needs_sha256 = {pitch: sha256(policy.pt)}`; `test_frozen_velocity_was_trained_on_the_frozen_pitch` so với file pitch hiện tại: train lại pitch
  mà không train lại velocity sẽ làm test fail, và `train_cascade.py` in cảnh báo ngay sau khi đóng băng pitch (`stale_dependents`) **[B12: đã sửa]**.
- [ ] Giới hạn còn lại: fingerprint chỉ thấy file trong danh sách; một thay đổi ngoài danh sách (ví dụ `rl_control/__init__.py`, phiên bản Isaac Lab hay rsl_rl) không bị phát hiện.

Lệnh:

```bash
$PY -m pytest tests/test_cascade.py -q
$PY -c "
from Balance_Car_RL.car.mdp.actions.frozen_policy import contract_fingerprint
print(contract_fingerprint('pitch'))      # 64 ký tự hex; đổi một trọng số reward thì đổi, đổi chú thích thì không
"
```

---

## Cấp 5 — Các term MDP chạy trong Isaac Lab (★★★★)

**Câu hỏi của cấp này:** Isaac Lab gọi những hàm nào của mình, theo thứ tự nào, trong một bước.

| Manager | Term của ta | File |
|---|---|---|
| Observation | `ImuPitchAndRate`, `wheel_speed_estimate` | `mdp/observations.py` |
| Command | `ScalarCommand` | `mdp/commands.py` |
| Action | `FrozenPitchAction` (stage velocity); upright/pitch dùng `JointEffortAction` của Isaac Lab | `mdp/actions/pitch_action.py` |
| Reward | `upright_exp`, `wheel_vel_l2`, `pitch_error_exp`, `speed_error_exp`, `speed_error_l1` | `mdp/rewards.py` |
| Termination | (của Isaac Lab) `time_out`, `bad_orientation` 0.8 rad | `car_env_cfg.py:144` |

### 5.1 `mdp/observations.py` — `ImuPitchAndRate` (dòng 23) và `axle_speed` (dòng 116)

Term có trạng thái (class, không phải hàm). Mỗi bước:

1. Đọc IMU sim (`lin_acc_b` có sẵn trọng lực, `ang_vel_b`), cộng offset theo episode và nhiễu Gauss (dòng 88-91).
2. `has_sample` (dòng 92): IMU đọc 0 ở tick đầu sau reset; khi đó giữ nguyên ước lượng đã khởi tạo.
3. Tốc độ trục từ encoder + pitch rate gyro, đạo hàm, low-pass `ACCEL_COMP_TAU_S` (dòng 96-102).
4. `imu_to_pitch` (cấp 2), ghi `_cache = [pitch, pitch rate]`.

`reset(env_ids)` (dòng 67): rút offset mới, khởi tạo `g` từ trọng lực **thật**, tốc độ trục từ encoder + pitch rate thật.

Checklist:
- [ ] Chỉ tính một lần mỗi bước (`_last_step`, dòng 84): Isaac Lab có thể gọi observation nhiều lần (ví dụ lúc dựng
  manager để suy kích thước, `observation_manager.py:591`).
- [ ] `env.car_imu = self` (dòng 45): `FrozenPitchAction` và `axle_speed` đọc **cùng** ước lượng. Thiếu term này thì `axle_speed` báo `RuntimeError` rõ ràng
  (test `test_axle_speed_needs_the_imu_term`); một term `ImuPitchAndRate` thứ hai (ví dụ cho critic) sẽ ghi đè `env.car_imu`, nên đừng thêm **[B9: giảm nhẹ]**.
- [ ] `axle_speed` (`v = r(mean q̇ + θ̇)`) là **một** hàm cho obs `wheel_speed_estimate` (dòng 130) và guard của `FrozenPitchAction`; nó dùng `env.car_imu.cache[:, 1]`, nên `imu` phải đứng **trước** trong nhóm
  (`PolicyCfg` của `car_env_cfg.py` đặt `imu` đầu tiên; `test_every_task_starts_its_observation_with_the_imu`).
- [ ] Sai số ước lượng (đo 2026-10-05, 256 env × 1500 bước): 0.010 rad dưới LQR, 0.011 dưới PID, 0.013 dưới LQR + nhiễu mô-men σ = 0.1 (mô-men trung bình 0.023 N m,
  gấp 4 lần RL). Con số 0.05–0.06 rad trong `meta.json` cũ có từ bộ lọc cũ khởi tạo từ **một** mẫu gia tốc kế nhiễu mỗi lần reset. Đo lại với policy mới **[B5]**.

### 5.2 `mdp/commands.py` — `ScalarCommand` (dòng 18)

- `_resample_command` (dòng 41): đều trong `[low, high]`, xác suất `zero_probability` cho đúng 0, giữ trong
  `resampling_time_range`.
- `_update_command` (dòng 50), **speed guard**: khi |v| > guard và lệnh cùng dấu với v, **đổi dấu** lệnh đã rút.

Checklist:
- [ ] Lý do của guard: giữ một góc nghiêng = gia tốc không đổi, sẽ chạm tốc độ tối đa 0.94 m/s và hết mô-men. Lệnh
  của stage velocity không bao giờ đòi điều đó, nên lúc train stage pitch cũng không đòi.
- [ ] Guard đọc vận tốc **thật** `root_lin_vel_b`: chấp nhận được vì đây là bộ sinh tín hiệu train, không phải bộ điều khiển (docstring ghi rõ);
  guard của `FrozenPitchAction` (5.3) dùng ước lượng.
- [ ] `_drawn` giữ lệnh gốc, `_command` là lệnh sau guard: guard có thể bật/tắt nhiều lần trong một lần giữ lệnh.
- [ ] Không log metric nào (`_update_metrics` rỗng): TensorBoard không có `Metrics/target/*`.

### 5.3 `mdp/actions/pitch_action.py` — `FrozenPitchAction` (dòng 31)

Action của stage velocity: 1 số trong [−1, 1] → pitch target → policy pitch đóng băng → mô-men 2 bánh.

| Phương thức | Khi nào chạy | Làm gì |
|---|---|---|
| `__init__` (45) | tạo env | `load_frozen(stage)`, tìm khớp bánh (giữ thứ tự khớp của articulation, như `JointEffortAction` của stage 1) |
| `process_actions` (74) | **1 lần mỗi bước env** | lưu action thô; θ* = clamp(a)·0.08; guard tốc độ (**đặt 0**, không đổi dấu); dựng obs 7 số; chạy policy; **kẹp ±1**; nhân stall |
| `apply_actions` (93) | **mỗi tick physics** (4 lần) | ghi cùng một mô-men |
| `reset` (67) | reset env | 0 cho action, θ*, last action của policy trong, mô-men |

`FrozenPitchActionCfg.pitch_scale` và `.torque_scale` là `MISSING` (bắt buộc): trước đây có giá trị mặc định chép tay trùng với hằng số thật.

Checklist:
- [ ] `_stage1_observation` (dòng 22) ghép đúng thứ tự `[imu(2), wheel_vel(2), last_action(2), pitch_target(1)]`
  (`test_frozen_action_rebuilds_the_exact_training_observation_order`, và test với stub trong `test_mdp_terms.py`).
- [ ] `env.car_imu.cache` là ước lượng của observation tính ở **cuối bước trước**, đúng cái mà stage 1 thấy khi chọn action.
- [ ] `_policy_last_action` lưu output **đã kẹp**, khớp `clip_actions = 1.0` (`test_pitch_action_feeds_back_the_clipped_action_like_training`).
- [ ] Guard ở đây **đặt 0**, còn ở `ScalarCommand` **đổi dấu**: cả hai cho giá trị nằm trong phân bố đã train.
- [ ] Guard dùng `axle_speed` = ước lượng của robot (encoder + pitch rate gyro), **không** dùng vận tốc simulator; `test_pitch_action_guard_uses_the_estimated_speed_not_the_simulator_speed`
  đặt vận tốc simulator = 2 m/s và encoder = 0.4 m/s để chứng minh. **[B3: đã sửa]** Phần còn thiếu để triển khai cascade lên robot là node ROS (cấp 8).

### 5.4 `mdp/rewards.py` (65 dòng)

| Hàm | Công thức | Dùng ở |
|---|---|---|
| `upright_exp` (21) | `exp(−(g_x² + g_y²)/std²)` = exp(−sin²(tilt)/std²), cả roll và pitch | upright (2.0), velocity (0.5) |
| `wheel_vel_l2` (28) | Σ q̇² | upright (−0.001), pitch (−0.002) |
| `yaw_rate_l2` (33) | `ω_z²` trong khung thân (rad²/s²) | mọi task, −0.05 |
| `body_pitch` (42) | `atan2(g_x, −g_z)`, pitch **thật** | trong `pitch_error_exp` |
| `pitch_error_exp` (48) | `exp(−(θ − θ*)²/std²)` | pitch (2.0, std 0.05) |
| `speed_error_exp` (54) | `exp(−(v_x − v*)²/std²)`, v trong khung thân | velocity (2.0, std 0.2) |
| `speed_error_l1` (60) | `|v_x − v*|` | velocity (−1.0) |

Checklist:
- [ ] Reward dùng trạng thái thật (privileged), observation dùng ước lượng: đúng thiết kế.
- [ ] `speed_error_l1` tồn tại vì kernel exp phẳng khi xa lệnh (`guide/04` §4.8).
- [ ] Chung cho mọi task (`RewardsCfg`, `car_env_cfg.py:117`): `alive` +1, `terminated` −5, `pitch_rate`
  (`ang_vel_xy_l2`, roll + pitch) −0.02, **`yaw_rate`** (`yaw_rate_l2`) −0.05, `output_change` (`action_rate_l2`) −0.01 (velocity −0.05).
- [ ] `yaw_rate` đóng **[B4]**: hai mô-men bánh độc lập có thể làm xe quay, và trước đây không reward nào phạt điều đó (policy pitch cũ xuất mô-men lệch nhau
  [−0.039, 0.141] ngay cả khi mọi input bằng 0). Sau khi train lại: `RMS turn rate` của `evaluate.py` (LQR 0.016, PID 0.029 rad/s) là mốc so sánh.

---

## Cấp 6 — Cấu hình task, PPO, đăng ký (★★★)

### 6.1 `rl_control/car_env_cfg.py` — phần chung của 3 task

| Lớp | Dòng | Nội dung |
|---|---|---|
| `CarSceneCfg` | 44 | mặt đất 100 m, `CAR_CFG`, IMU tại `.../Robot/Geometry/base_link` với offset từ `imu_mount.json`, đèn |
| `ActionsCfg` | 63 | `JointEffortActionCfg`, `scale = stall` |
| `PolicyCfg` | 74 | `imu` đứng đầu; `concatenate_terms`, không corruption (nhiễu nằm trong `ImuPitchAndRate`) |
| `EventCfg` | 92 | reset: pitch ±0.25 rad, yaw đều, vận tốc góc ±0.5 rad/s **quanh y thế giới**; bánh ±0.5 rad/s; đẩy ±0.3 m/s mỗi 2–4 s |
| `RewardsCfg` | 117 | xem 5.4 (có `yaw_rate`) |
| `TerminationsCfg` | 144 | hết giờ, nghiêng > 0.8 rad |
| `CarEnvCfg` | 156 | 200 Hz, decimation 4, 10 s, render mỗi bước policy |

Checklist:
- [ ] Vì yaw được rút ngẫu nhiên, "±0.5 rad/s quanh y thế giới" lúc reset là cú đá lẫn roll/pitch, không phải pitch
  thuần (`guide/04` dòng 62 có ghi).
- [ ] Với configclass, trường của lớp cha đứng trước: thứ tự obs = `imu` rồi các term của task. Đây là **hợp đồng**
  của policy đóng băng.

### 6.2 Ba file task

| Task | File | Obs (thứ tự) | Action | Reward thêm |
|---|---|---|---|---|
| `BalanceCar-Upright-v0` | `upright_env_cfg.py` | imu, wheel_vel, last_action (6) | 2 mô-men | upright 2.0, wheel_vel −0.001 |
| `BalanceCar-Pitch-v0` | `pitch_env_cfg.py` | imu, wheel_vel, last_action, pitch_target (7) | 2 mô-men | pitch 2.0, wheel_vel −0.002 |
| `BalanceCar-Velocity-v0` | `velocity_env_cfg.py` | imu, speed, speed_target, last_action (5) | θ* qua `FrozenPitchAction` | speed 2.0, speed_error −1.0, upright 0.5, output_change −0.05 |

Checklist:
- [ ] `PITCH_TARGET_MAX_RAD = 0.08` (≈ 0.8 m/s² gia tốc) là **cùng** con số cho lệnh stage 1 và `pitch_scale` stage 2.
- [ ] `SPEED_GUARD_M_S = 0.5` > `SPEED_MAX_M_S = 0.4`: guard ít khi bật ở stage 2, chủ yếu khi bị đẩy.
- [ ] Lệnh pitch đổi mỗi 0.5–1.5 s, lệnh tốc độ mỗi 3–6 s; 20 % là 0.

Lệnh (thứ tự obs của cả 3 task, không cần simulator):

```bash
$PY -c "
import Balance_Car_RL.tasks
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
for t in ['BalanceCar-Upright-v0', 'BalanceCar-Pitch-v0', 'BalanceCar-Velocity-v0']:
    p = load_cfg_from_registry(t, 'env_cfg_entry_point').observations.policy
    print(t, [k for k, v in p.__dict__.items() if hasattr(v, 'func')])
" 2>&1 | grep BalanceCar-
```

### 6.3 `rl_control/agents/*_ppo_cfg.py` — PPO (rsl_rl)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| mạng | actor 64×64, critic 64×64, ELU, chuẩn hoá obs trong actor | actor nhỏ, chạy được trên vi điều khiển |
| `num_steps_per_env` | 24 | |
| `max_iterations` | 300 (upright, pitch), 500 (velocity) | |
| `clip_actions` | **1.0** | action và `last_action` trong [−1, 1] (khớp việc kẹp trong `FrozenPitchAction` và ROS) |
| phân phối | `BoundedGaussianCfg`, `init_std` 0.5, std trong (0.02, 0.5) | chép từ Drone |
| PPO | clip 0.2, entropy 0.005, lr 1e-3 adaptive, KL 0.01, γ 0.99, λ 0.95 | Drone dùng entropy 0 và critic 128×128 |
| `experiment_name` | `balance_car_<task>` | `train_cascade.STAGES` dựa vào tên này |

Checklist:
- [ ] `test_stage_table_matches_the_registry` kiểm tên experiment khớp `train_cascade.STAGES`.

### 6.4 Đăng ký

`tasks/__init__.py` → `import Balance_Car_RL.car` → `car/__init__.py` → `from . import rl_control` → vòng `_TASKS`
gọi `gym.register`. Thiếu dòng import trong `tasks/__init__.py` thì `isaaclab train` không tìm thấy task.

Lệnh: `$PY -m pytest tests/test_registration.py tests/test_cascade.py -q`.

---

## Cấp 7 — Script (★★★)

### 7.1 `scripts/train_cascade.py` (309 dòng) và `stale_dependents` (dòng 135)

Cho mỗi stage theo thứ tự `pitch → velocity`:

| Bước | Dòng | Làm gì | Bẫy |
|---|---|---|---|
| 0 | 150-160 | stage cần stage dưới đã đóng băng và **không cũ** (fingerprint khớp, và có trong `meta.json`); nếu không thì dừng, trừ khi `--allow-stale` | **[B1: đã sửa]** |
| 1 | 165-195 | `isaaclab train` có video, thử lại 3 lần (Kit đôi khi crash 139); thành công = có dòng `Training time` | `isaaclab train` trả mã 0 cả khi lỗi |
| 2 | 196-212 | `isaaclab play` xuất policy; không video thì bị giết sau 150 s (play không tự dừng) | |
| 3 | 213-236 | `evaluate.py --json`; **không đạt thì không đóng băng** và dừng | xoá `eval.json` cũ trước (dòng 215) |
| 4 | 237-260 | chép `policy.pt`, ONNX một file, `meta.json` (+ `contract_sha256` nếu stage có trong `_CONTRACT_FILES`, + `needs_sha256` nếu stage train trên stage khác); rồi in cảnh báo cho stage trên đã cũ (`stale_dependents`) | train lại pitch mà không train lại velocity thì velocity bị báo **[B12: đã sửa]** |
| 5 | 261-276 | đường cong train, mp4 + gif vào `guide/media/` | gif < 2000 kB (hook pre-commit) |

Checklist:
- [ ] `newest_run` chọn theo mtime **sau** lúc bắt đầu, nên không lấy nhầm run cũ.
- [ ] `run()` giết **cả process group** khi timeout (Kit sinh process con giữ pipe).
- [ ] Không bao giờ dùng `pkill -f isaaclab` (giết cả các phiên Drone_RL khác).
- [ ] `--allow-stale` chỉ để thử nhanh; `test_retraining_a_stage_marks_the_stages_trained_on_it_as_stale` kiểm `stale_dependents`.

Lệnh: `$PY scripts/train_cascade.py --dry-run`.

### 7.2 `scripts/evaluate.py` (235 dòng)

- Chạy policy (checkpoint) hoặc `--controller lqr|pid` (chỉ task Upright, vì cần obs 6 số).
- Bỏ 50 bước đầu; đo RMS pitch, sai số ước lượng (obs[:, 0] so với pitch thật), **tốc độ quay yaw**, mô-men, tốc độ; với task có lệnh:
  sai số bám, **median** và phân vị 90, RMS sau khi ổn định.
- Đạt: tỉ lệ ngã ≤ 2 % và median ≤ 0.03 rad (pitch) / 0.08 m/s (velocity); Upright: RMS pitch ≤ 0.10 rad.
- Thoát bằng `os._exit` (đóng Kit sẽ giết process trước khi in kết quả).

Checklist:
- [ ] `cmd_active` đọc **trước** `env.step`: lệnh mà action được chọn theo; `env.step` có thể rút lệnh mới.
- [ ] Policy **bắt buộc** `--checkpoint` (dòng 92): trước đây lấy run mới nhất, có thể là `model_0.pt` chưa train **[B10: đã sửa]**.
- [ ] `RslRlVecEnvWrapper(clip_actions=agent_cfg.clip_actions)`: giờ PID/LQR cũng bị kẹp (vốn đã tự kẹp, nên không đổi).

### 7.3 Công cụ

| File | Làm gì |
|---|---|
| `car/tools/plot_training.py` | vẽ reward và độ dài episode từ TensorBoard (dùng bởi `train_cascade.py`) |
| `car/tools/summarize_training_run.py` | chép từ Drone: bảng scalar, % trần từng reward, lý do kết thúc, bảng LaTeX, nháp nhận xét |
| `tools/fetch_pololu_cad.sh` | tải STEP/DXF vào `cad/`, PDF vào `docs/hardware/balboa/` |

Lệnh: `$PY src/Balance_Car_RL/car/tools/summarize_training_run.py --experiment balance_car_pitch | head -40`.

---

## Cấp 8 — Triển khai ROS 2 (★★★)

| File | Làm gì |
|---|---|
| `bridge_node.py` | nghe `imu` (thô) và `joint_states`; mỗi chu kỳ 50 Hz: ước lượng → policy → `wheel_torque_cmd`; dừng gửi khi dữ liệu cũ hơn 3 chu kỳ |
| `estimator.py` | cấp 2.2 |
| `policy.py` | ONNX, obs 6 số, kẹp ±1 rồi nhân `TORQUE_SCALE_NM`; giá trị **đã kẹp** đưa vào `last_action` (khớp `clip_actions = 1.0`) |

Checklist:
- [ ] `TORQUE_SCALE_NM` = stall (`test_ros_torque_scale_matches_training`).
- [ ] `rate_hz` khác 50 chỉ cảnh báo, vẫn chạy.
- [ ] Node chỉ chạy được policy **Upright** (6 obs, 2 action). Cascade pitch + velocity chưa có đường triển khai: thiếu
  node chạy hai mạng nối tiếp (obs 7/5 số, pitch target, kẹp). Guard tốc độ không còn cản trở: nó dùng ước lượng encoder + IMU **[B3: đã sửa]**.
- [ ] `models/` **trống** (policy cũ đã xoá, 0.4); `test_policy.py` bị skip tới khi bạn xuất ONNX mới (`guide/06` §6.1). Policy Upright phải train lại
  vì nó học với action không kẹp.
- [ ] Chưa chạy trên robot thật (`guide/06`).

---

## Cấp 9 — Toàn hệ thống theo thời gian: một bước của stage velocity (★★★★★)

Đã kiểm theo `isaaclab/envs/manager_based_rl_env.py` (`step`, dòng ~200–300).

```text
policy velocity chọn a_t từ obs_t (tính ở cuối bước t−1)
│
├─ action_manager.process_action(a_t)                      1 lần
│     FrozenPitchAction.process_actions:
│        θ* = clamp(a_t)·0.08, guard (v ước lượng = axle_speed) ← lệnh tốc độ c_t vẫn là cái obs_t đã thấy
│        obs_pitch = [env.car_imu.cache (cuối bước t−1), q̇ (hiện tại = cuối bước t−1), last_pitch, θ*]
│        u = clamp(policy_pitch(obs_pitch)) · stall
│
├─ lặp 4 lần (decimation):  apply_action (ghi u) → sim.step (5 ms) → scene.update (IMU đo)
│
├─ common_step_counter += 1
├─ termination_manager.compute → reward_manager.compute (trạng thái thật, ×0.02)
├─ reset các env đã xong (ImuPitchAndRate.reset khởi tạo từ trọng lực thật; FrozenPitchAction.reset)
├─ command_manager.compute: hết thời gian thì rút lệnh mới; speed guard (stage pitch)
├─ event interval: cú đẩy (đặt vận tốc gốc)
└─ observation_manager.compute → ImuPitchAndRate.__call__ (lọc 1 bước), wheel_speed_estimate, lệnh, last_action
      = obs_{t+1}
```

### 9.1 Bất biến xuyên suốt (mỗi điều phải đúng ở MỌI nơi)

- [ ] Policy pitch đóng băng thấy đúng thứ nó thấy lúc train: cùng thứ tự, cùng thời điểm đo (cuối bước trước), cùng
  tần số 50 Hz, cùng quy ước kẹp (sau khi train lại).
- [ ] Mọi chỗ đổi `[−1, 1]` → mô-men dùng cùng stall torque: `ActionsCfg`, `FrozenPitchAction`, ROS.
- [ ] `ψ̇ = q̇ + θ̇` ở mọi nơi tính tốc độ trục: `axle_speed` (obs và guard), `ImuPitchAndRate`, ROS estimator, PID, LQR.
- [ ] Ước lượng sim ≡ ước lượng ROS (test), trừ cách khởi tạo.
- [ ] Không chỗ nào của controller (obs hoặc action) dùng trạng thái thật: guard của `FrozenPitchAction` dùng `axle_speed` (đã sửa, **[B3]**).
  Chỉ reward, termination, evaluate và guard của `ScalarCommand` (bộ sinh tín hiệu train) dùng trạng thái thật.

### 9.2 Quy trình train → đóng băng → stage trên (logic, không chạy)

1. `train_cascade.py --stages pitch`: train, play, đánh giá, đạt thì ghi `frozen/pitch/`.
2. `train_cascade.py --stages velocity`: cần `frozen/pitch/policy.pt`; `FrozenPitchAction` load nó lúc dựng env.
3. Train lại pitch mà **không** train lại velocity: velocity cũ chạy trên pitch mới. Giờ có người báo: `train_cascade.py` in cảnh báo sau khi đóng băng, và
   `test_frozen_velocity_was_trained_on_the_frozen_pitch` fail (`needs_sha256`).

---

## Cấp 10 — Test (★★)

| File | Bảo vệ | Không bảo vệ |
|---|---|---|
| `test_car_cfg.py` | mesh tồn tại, chiều khớp bánh, bán kính/bề rộng, COM trên trục, quán tính hợp lệ, khối lượng URDF = cfg, hằng số động cơ, scale ROS | hình dạng collision (xem lệnh ở 1.2), giá trị rotor |
| `test_estimation.py` | hằng số ROS = sim, phép quay, quaternion, nhiễu, torch = numpy, dấu pitch, không mẫu thì chỉ gyro, bù gia tốc | `ImuPitchAndRate` (cần sim): reset, `has_sample`, `_last_step` |
| `test_pid.py`, `test_pid_control.py` | PID cơ bản; PID cascade hồi phục trên mô hình tuyến tính, dấu, reset | |
| `test_lqr_control.py` | một cực không ổn định, LQR ổn định mô hình, hồi phục, dấu | |
| `test_registration.py` | task Upright đăng ký, cfg dựng được | |
| `test_cascade.py` | bảng stage = registry, thứ tự phụ thuộc, thứ tự obs của cả 3 task, cùng dải/scale giữa 2 stage, thứ tự ghép obs, fingerprint (bỏ chú thích, phủ đủ file), `load_frozen` với actor giả (kích thước, thiếu file, cảnh báo thiếu/lệch fingerprint), `stale_dependents`; **khi có policy thật**: fingerprint khớp, velocity train trên đúng pitch, policy pitch phản ứng đúng dấu | policy có hành vi tốt (cần `evaluate.py`) |
| `test_mdp_terms.py` | `axle_speed` (cộng pitch rate, báo lỗi khi thiếu IMU), `yaw_rate_l2`, guard của `ScalarCommand` (đổi dấu, không đè lệnh gốc, không guard, phạm vi lấy mẫu), `FrozenPitchAction` (thang, kẹp, thứ tự obs, kẹp last action, guard dùng ước lượng) trên env giả | các term trong simulator thật (đã chạy thử 2 vòng PPO, xem dưới) |
| ROS `test_estimator.py`, `test_policy.py` | ước lượng numpy; ONNX ra mô-men hữu hạn, bị chặn (skip khi `models/` trống) | ONNX có khớp quy ước kẹp không |

Checklist:
- [ ] Chạy lệnh 0.1 → 61 passed, 4 skipped.
- [ ] Đọc `tests/test_mdp_terms.py`: nó dựng `FrozenPitchAction.__new__` và `ScalarCommand.__new__` với env giả (cách của `Drone_RL/tests/test_rl_layers.py`) **[B8: đã sửa]**.

Những gì chỉ kiểm được khi chạy mô phỏng (bạn tự quyết khi nào chạy):
- Video train và demo: xe có xoay tại chỗ không (xem [B4]), có bám lệnh không.
- `scripts/evaluate.py` cho từng stage; `summarize_training_run.py` cho % trần từng reward.

---

## Phụ lục A — File ngoài luồng chính

| File | Tình trạng |
|---|---|
| `src/Balance_Car_RL/assets/__init__.py` | `BALANCE_CAR_RL_ASSETS_DIR` |
| USD của URDF | Isaac Lab sinh vào `/tmp/IsaacLab/usd_*/` mỗi lần chạy khi hash (cấu hình + nội dung URDF) đổi; bản cũ trong `assets/data/balboa/balboa/` đã xoá |
| `scripts/list_envs.py` | template Isaac Lab |
| `models/` | trống: policy Upright cho ROS đã xoá (0.4), tự xuất lại sau khi train (`guide/06`) |
| `cad/` | STEP, DXF và export Onshape, git-ignored, cần cho `build_balboa_urdf.py` |
| `docs/hardware/balboa/*.pdf` | nguồn của các số (`docs/readme.md`) |
| `logs/review.md` | ghi chú review của phiên trước, không phải code |

## Phụ lục B — Vấn đề của lần review trước và trạng thái

| # | Vấn đề | Trạng thái | Cách xử lý / bằng chứng |
|---|---|---|---|
| B1 | Mọi policy đã train đều stale và không cơ chế nào báo | **Đã sửa** | Xoá policy cũ. `load_frozen` cảnh báo khi fingerprint thiếu hoặc lệch; `train_cascade` dừng nếu stage dưới cũ; `meta.json` mới luôn có `contract_sha256`. **Việc còn lại của bạn: train lại** |
| B2 | Fingerprint thiếu file và quá thô | **Đã sửa** | 12 file (thêm URDF, `imu_mount.json`, collision mesh, `rewards.py`, 2 file PPO), băm cây cú pháp nên chú thích/docstring không ảnh hưởng |
| B3 | Guard tốc độ của `FrozenPitchAction` đọc vận tốc thật | **Đã sửa** | dùng `axle_speed` (encoder + IMU); có test với stub. Smoke test 2 vòng PPO của stage velocity trên pitch tạm chạy được |
| B4 | Không reward nào phạt quay yaw | **Đã sửa (cần train để xác nhận)** | `yaw_rate_l2` −0.05; mốc: LQR 0.016, PID 0.029 rad/s. Sau train lại, đọc `RMS turn rate` của `evaluate.py` |
| B5 | Sai số ước lượng pitch lớn ngang RMS pitch | **Đã đo, không phải lỗi của bộ lọc** | 0.010 (LQR), 0.011 (PID), 0.013 (LQR + nhiễu mô-men σ = 0.1). Số 0.05–0.06 cũ có từ bộ lọc khởi tạo từ một mẫu nhiễu. Không thêm bù `θ̈·h` khi chưa có bằng chứng cần; đo lại với policy mới |
| B6 | `MOTOR_ROTOR_INERTIA` là ước lượng | **Còn mở (không có dữ liệu)** | Độ nhạy trên mô hình tuyến tính: cực không ổn định 11.9 / 10.6 / 9.2 / 8.1 / 7.4 rad/s ứng với rotor ×0.25 / ×0.5 / ×1 / ×2 / ×4; hệ số LQR đổi ×3 theo cùng dải. Số 7e-9 kg m² ứng với rotor ~1.5 g, bán kính 3 mm, hợp lý nhưng không có nguồn |
| B7 | Hộp collision chassis mỏng hơn mesh | **Đã sửa** | một hull lồi: 1.43 % đỉnh nằm ngoài (tối đa 0.63 mm), điểm thấp nhất khớp mesh; nhập vào Isaac Lab thành `convexHull`, smoke test 3 task chạy |
| B8 | Thiếu test cho `ScalarCommand` và `FrozenPitchAction` | **Đã sửa** | `tests/test_mdp_terms.py` |
| B9 | `env.car_imu` là biến chung của env | **Giảm nhẹ** | báo lỗi rõ khi thiếu; vẫn là một thuộc tính, nên chỉ một term `ImuPitchAndRate` mỗi env |
| B10 | `evaluate.py` có thể lấy run chưa train | **Đã sửa** | `--checkpoint` bắt buộc cho policy |
| B11 | Giả định chưa đo | **Còn mở** | khối lượng thân/bánh (không nguồn nào công bố: đã kiểm user's guide, datasheet, trang Pololu), rotor, hiệu suất hộp số 100 %, offset IMU còn lại 10 %, vị trí chip IMU; không có nhiễu encoder, trễ, ngẫu nhiên hoá khối lượng/ma sát. Cách đo ở `guide/01` §1.5–1.6 |
| B12 | (mới) velocity không biết pitch đã train lại | **Đã sửa** | `needs_sha256` + cảnh báo sau khi đóng băng + test |

Còn lại sau lần này, đều cần việc ngoài code: **train lại** (pitch → velocity, upright), **cân robot thật** (B11), **đo quán tính rotor** (B6), **đưa cascade lên robot** (node chạy 2 mạng).

## Phụ lục C — Bảng theo dõi

| Cấp | Mục | Xong | Ghi chú |
|---|---|---|---|
| 0 | quy ước, bản đồ, trạng thái | [ ] | |
| 1.1 | `car_cfg.py` | [ ] | |
| 1.2 | `build_balboa_urdf.py`, URDF | [ ] | |
| 2.1 | `estimation.py` | [ ] | |
| 2.2 | ROS `estimator.py` | [ ] | |
| 3.1 | `pid.py` | [ ] | |
| 3.2 | `cascade_pid.py` | [ ] | |
| 3.3 | `model.py` | [ ] | |
| 3.4 | `lqr.py` | [ ] | |
| 4 | `frozen_policy.py` | [ ] | |
| 5.1 | `observations.py` | [ ] | |
| 5.2 | `commands.py` | [ ] | |
| 5.3 | `pitch_action.py` | [ ] | |
| 5.4 | `rewards.py` | [ ] | |
| 6.1 | `car_env_cfg.py` | [ ] | |
| 6.2 | 3 task | [ ] | |
| 6.3 | PPO | [ ] | |
| 6.4 | đăng ký | [ ] | |
| 7.1 | `train_cascade.py` | [ ] | |
| 7.2 | `evaluate.py` | [ ] | |
| 7.3 | công cụ | [ ] | |
| 8 | ROS | [ ] | |
| 9.1 | bất biến | [ ] | |
| 9.2 | quy trình train/đóng băng | [ ] | |
| 10 | test | [ ] | |
