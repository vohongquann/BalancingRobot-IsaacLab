#!/usr/bin/env bash
# Download the official Pololu Balboa 32U4 CAD (STEP) and drawings (PDF, DXF) into cad/balboa/.
# Reference only, about 52 MB, git-ignored.
# The simulation does not need them: the URDF and meshes are already in src/Balance_Car_RL/assets/data/balboa/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="https://www.pololu.com/file"
mkdir -p "$ROOT/cad/balboa"

for f in \
  0J1309/balboa-balancing-robot-kit.step \
  0J1269/balboa-32u4-control-board.step \
  0J1283/balboa-32u4-balancing-robot-kit-dimensions.pdf \
  0J1264/balboa-32u4-control-board-dimensions.pdf \
  0J1265/bal01a-drill.dxf \
  0J1266/balboa-kit-gear-ratio-chart.pdf \
  0J1087/LSM6DS33.pdf; do
  out="$ROOT/cad/balboa/$(basename "$f")"
  [ -s "$out" ] && { echo "skip  $out"; continue; }
  echo "fetch $f"
  curl -fL --retry 3 -o "$out" "$BASE/$f"
done
