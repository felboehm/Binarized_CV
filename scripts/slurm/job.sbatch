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
#   BCV_DATA       shared copy of data/raw (default: /global/D1/homes/$USER/bcv-data/raw)
#   BCV_STAGE      1 = copy BCV_DATA to node-local /tmp first (default 1)
#   BCV_OVERRIDES  extra default overrides, appended to the ones set here
#
# Partition: a100q (n013/n014, x86_64, 2x A100 each) by default. a40q
# (n009-n012, 1x A40 each) is aarch64 and runs from .venv-aarch64, built by
# setup_env.sh on one of its nodes (see there); submit with -p a40q.
# No --mem: the GPU nodes are configured with RealMemory=1 (MB), so any
# memory request is rejected ("Memory specification can not be satisfied").
# Memory isn't tracked there; the job can use the node's RAM.
# --propagate=NONE: don't inherit the login node's ulimits, in particular its
# 16 GB virtual-memory cap (ulimit -v), which makes torch fail with
# "DefaultCPUAllocator: can't allocate memory" before training starts.
#
#SBATCH --partition=a100q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --propagate=NONE
#SBATCH --output=runs/slurm/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
[ $# -ge 1 ] || { echo "usage: sbatch scripts/slurm/job.sbatch SCRIPT [ARGS...]" >&2; exit 2; }

BCV_DATA="${BCV_DATA:-/global/D1/homes/$USER/bcv-data/raw}"
BCV_STAGE="${BCV_STAGE:-1}"
export TQDM_MININTERVAL=60  # one progress line per minute in the log, not per batch

# One venv per architecture (setup_env.sh): .venv on x86_64, .venv-<arch>
# otherwise (a40q = n009-n012 is aarch64). Fail before staging if it can't run.
venv=.venv
[ "$(uname -m)" = x86_64 ] || venv=".venv-$(uname -m)"
if ! "$venv/bin/python" -c "" 2>/dev/null; then
    echo "error: $venv/bin/python does not run on $(hostname) ($(uname -m)); build it there with" >&2
    echo "  srun -p ${SLURM_JOB_PARTITION:-<partition>} --gres=gpu:1 -c 8 --propagate=NONE bash scripts/slurm/setup_env.sh" >&2
    exit 1
fi

# The repo's own data/raw holds only .gitkeep files on the cluster; staging
# from it (once the default when BCV_DATA wasn't exported) emptied the node
# copy and crashed the job, and any other job reading that copy.
if [ -z "$(find "$BCV_DATA/" -name '*.jp*g' -print -quit 2>/dev/null)" ]; then
    echo "error: BCV_DATA=$BCV_DATA contains no images; export BCV_DATA=<shared copy of data/raw>" >&2
    exit 1
fi

# Node-local copy of the dataset: reading ~120k JPEGs from shared storage
# would starve the GPU. The copy persists in /tmp/$USER across jobs, so later
# jobs on the same node only pay rsync's file-list check. flock serialises
# jobs that start on the same node at the same time. No --delete: other jobs
# on the node may be reading the copy, and a stale extra file is harmless.
raw_root="$BCV_DATA"
if [ "$BCV_STAGE" = 1 ]; then
    raw_root="/tmp/$USER/bcv-data/raw"
    mkdir -p "$raw_root"
    echo "[$(date +%T)] staging $BCV_DATA -> $raw_root"
    flock "/tmp/$USER/bcv-data.lock" rsync -a "$BCV_DATA/" "$raw_root/"
    echo "[$(date +%T)] staged"
fi

# Read by load_config ahead of the explicit overrides, so ARGS still win.
export BCV_OVERRIDES="data.raw_root=$raw_root data.num_workers=$(( ${SLURM_CPUS_PER_TASK:-8} - 2 )) ${BCV_OVERRIDES:-}"

args=()
for a in "$@"; do args+=("${a//\{task\}/${SLURM_ARRAY_TASK_ID:-0}}"); done

echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "BCV_OVERRIDES=$BCV_OVERRIDES"
echo "cmd: python ${args[*]}"
exec "$venv/bin/python" "${args[@]}"
