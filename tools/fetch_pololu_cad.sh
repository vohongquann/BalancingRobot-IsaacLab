#!/usr/bin/env bash
# Download the official Pololu Balboa 32U4 CAD (STEP, DXF) into cad/balboa/ (reference only, about 50 MB, git-ignored)
# and the datasheets and drawings (PDF) into docs/hardware/balboa/.
# The simulation does not need them: the URDF and meshes are already in src/Balance_Car_RL/assets/data/balboa/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="https://www.pololu.com"
mkdir -p "$ROOT/cad/balboa" "$ROOT/docs/hardware/balboa"

fetch() {  # fetch <url path> <output file>
  [ -s "$2" ] && { echo "skip  $2"; return; }
  echo "fetch $1"
  curl -fL --retry 3 -o "$2" "$BASE/$1"
}

for f in file/0J1309/balboa-balancing-robot-kit.step file/0J1269/balboa-32u4-control-board.step \
         file/0J1265/bal01a-drill.dxf; do
  fetch "$f" "$ROOT/cad/balboa/$(basename "$f")"
done

for f in file/0J1283/balboa-32u4-balancing-robot-kit-dimensions.pdf \
         file/0J1264/balboa-32u4-control-board-dimensions.pdf \
         file/0J1266/balboa-kit-gear-ratio-chart.pdf \
         file/0J1087/LSM6DS33.pdf \
         file/0J1487/pololu-micro-metal-gearmotors-rev-6-2.pdf; do
  fetch "$f" "$ROOT/docs/hardware/balboa/$(basename "$f")"
done
fetch docs/pdf/0J70/balboa_32u4_robot.pdf "$ROOT/docs/hardware/balboa/balboa-32u4-robot-users-guide.pdf"
