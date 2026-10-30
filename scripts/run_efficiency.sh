#!/usr/bin/env bash
# Efficiency of the comparison set (scripts/benchmark_efficiency.py) on this
# machine's CPU and/or GPU, one JSON per config and device in
# runs/efficiency/<host>/<config>_<device>.json; summarise with
# scripts/collect_efficiency.py.
#
#   bash scripts/run_efficiency.sh [DEVICE...]      # default: cpu cuda
#
# $CPUS pins the CPU runs (default 0,2,4,6: one thread per P-core of a
# Core Ultra 5 135H; see the benchmark's docstring). Extra benchmark options
# go in $BENCH_ARGS, e.g. BENCH_ARGS="--rounds 5 --seconds 20".
# For CPU energy, make RAPL readable first (resets on reboot):
#   sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj
set -euo pipefail
cd "$(dirname "$0")/.."

devices=("$@")
[ ${#devices[@]} -eq 0 ] && devices=(cpu cuda)
cpus="${CPUS:-0,2,4,6}"
out_dir="runs/efficiency/$(hostname -s)"

# name -> hydra overrides
declare -A configs=(
  [yolo26_rgb_fp32]="model=yolo26"
  [ms_yolov8_fp32]="model=ms_yolov8"
  [early_fusion_fp32]="model=yolo26_early_fusion"
  [early_fusion_ib_guided]="model=yolo26_early_fusion_bnn"
  [early_fusion_ib_guided_attn]="model=yolo26_early_fusion_bnn model.binarization.regions.attention=binary"
  [midfusion_fp32]="model=yolo26_midfusion"
  [midfusion_ib_guided]="model=yolo26_midfusion_bnn"
  [midfusion_ib_guided_attn]="model=yolo26_midfusion_bnn model.binarization.regions.attention=binary"
)

for device in "${devices[@]}"; do
  device_args=(--device "$device")
  [ "$device" = cpu ] && device_args+=(--cpus "$cpus")
  for name in $(printf '%s\n' "${!configs[@]}" | sort); do
    echo "== $name on $device"
    # shellcheck disable=SC2086  # overrides and BENCH_ARGS are word lists
    python scripts/benchmark_efficiency.py "$out_dir/${name}_${device}.json" \
      "${device_args[@]}" ${BENCH_ARGS:-} -- ${configs[$name]}
  done
done
