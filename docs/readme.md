# docs: sources

This folder holds the documents the numbers of the simulation come from (datasheets, drawings, papers). The
documentation itself is [guide/](../guide/readme.md); every value taken from these files is listed with its source in
[guide/01_robot.md](../guide/01_robot.md) and in `src/Balance_Car_RL/car/car_cfg.py`. `tools/fetch_pololu_cad.sh`
downloads them again.

| File | Used for |
|---|---|
| `hardware/balboa/balboa-32u4-robot-users-guide.pdf` | wheel, gearmotor options, encoder counts, battery (guide 1.2) |
| `hardware/balboa/balboa-32u4-balancing-robot-kit-dimensions.pdf` | wheel diameter, ground clearance (guide 1.2) |
| `hardware/balboa/balboa-32u4-control-board-dimensions.pdf` | control board outline; the IMU is assumed at its centre (guide 1.4) |
| `hardware/balboa/balboa-kit-gear-ratio-chart.pdf` | exact 50:1 gearmotor ratio 3344/65, 49:17 external gearbox, total 148.3:1 (guide 1.2) |
| `hardware/balboa/pololu-micro-metal-gearmotors-rev-6-2.pdf` | 50:1 HPCB 6 V: 650 rpm no-load, 0.74 kg cm stall torque at 6 V (guide 1.2, 3.3) |
| `hardware/balboa/LSM6DS33.pdf` | IMU noise densities, zero-g and zero-rate offsets, Table 3 (guide 1.2, 2.1) |

Papers: none yet (no number of the simulation comes from a paper). Put them in `papers/`, named
`<author>_<year>_<title>.pdf`, and add a row above.
