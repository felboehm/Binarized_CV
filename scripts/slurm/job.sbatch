#!/usr/bin/env bash
# Generic GPU job: runs one of the repo's Python entry points with the
# cluster's data location and worker count appended as Hydra overrides.
#
#   sbatch [sbatch options] scripts/slurm/job.sbatch SCRIPT [ARGS...]
#
# Examples:
#   sbatch -J fp32_control scripts/slurm/job.sbatch scripts/train_eval.py \
#       yolo26_early_fusion_fp32_control runs/bnn/fp32_control.json model=yolo26_early_fusion ...
#   sbatch -J sweep --array=0-2 scripts/slurm/job.sbatch scripts/binarization_sweep.py \
#       --model yolo26_early_fusion_bnn --init-checkpoint ... --epochs 3 train.seed={task}
#
# `{task}` anywhere in ARGS becomes $SLURM_ARRAY_TASK_ID. data.raw_root and
# data.num_workers are set through $BCV_OVERRIDES (read by load_config before
# the explicit overrides), so passing them in ARGS still overrides them.
#
# Environment (export before sbatch, or edit the defaults):
#   BCV_DATA       shared copy of data/raw (default: <repo>/data/raw)
#   BCV_STAGE      1 = copy BCV_DATA to node-local /tmp first (default 1)
#   BCV_OVERRIDES  extra default overrides, appended to the ones set here
#
# Partition: a100q (n013/n014, x86_64, 2x A100 each). Not a40q: its nodes
# (n009-n012) are aarch64, so the x86_64 .venv doesn't run there.
# No --mem: the GPU nodes are configured with RealMemory=1 (MB), so any
# memory request is rejected ("Memory specification can not be satisfied").
# Memory isn't tracked there; the job can use the node's RAM.
#
#SBATCH --partition=a100q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/slurm/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
[ $# -ge 1 ] || { echo "usage: sbatch scripts/slurm/job.sbatch SCRIPT [ARGS...]" >&2; exit 2; }

BCV_DATA="${BCV_DATA:-$PWD/data/raw}"
BCV_STAGE="${BCV_STAGE:-1}"
export TQDM_MININTERVAL=60  # one progress line per minute in the log, not per batch

# Fail before staging if the venv can't run here: .venv is built on the
# x86_64 login node, and some GPU nodes (a40q = n009-n012) are aarch64.
if ! .venv/bin/python -c "" 2>/dev/null; then
    echo "error: .venv/bin/python does not run on $(hostname) ($(uname -m)); use an x86_64 partition" >&2
    exit 1
fi

# Node-local copy of the dataset: reading ~120k JPEGs from shared storage
# would starve the GPU. The copy persists in /tmp/$USER across jobs, so later
# jobs on the same node only pay rsync's file-list check. flock serialises
# jobs that start on the same node at the same time.
raw_root="$BCV_DATA"
if [ "$BCV_STAGE" = 1 ]; then
    raw_root="/tmp/$USER/bcv-data/raw"
    mkdir -p "$raw_root"
    echo "[$(date +%T)] staging $BCV_DATA -> $raw_root"
    flock "/tmp/$USER/bcv-data.lock" rsync -a --delete "$BCV_DATA/" "$raw_root/"
    echo "[$(date +%T)] staged"
fi

# Read by load_config ahead of the explicit overrides, so ARGS still win.
export BCV_OVERRIDES="data.raw_root=$raw_root data.num_workers=$(( ${SLURM_CPUS_PER_TASK:-8} - 2 )) ${BCV_OVERRIDES:-}"

args=()
for a in "$@"; do args+=("${a//\{task\}/${SLURM_ARRAY_TASK_ID:-0}}"); done

echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "BCV_OVERRIDES=$BCV_OVERRIDES"
echo "cmd: python ${args[*]}"
exec .venv/bin/python "${args[@]}"
