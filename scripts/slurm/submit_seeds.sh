#!/usr/bin/env bash
# Submit the comparison (fp32 control, ib_guided, ib_guided + binary
# attention; all warm-started from the fp32 early fusion checkpoint) as one
# Slurm job array per config, one task per seed. Metrics land in
# runs/bnn/<config>_<schedule>_seed<N>.json; summarise with
# scripts/collect_results.py.
#
# $SCHEDULE picks the training length:
#   20ep     (default) 20 epochs, cosine, the schedule of the seed-0 runs
#   plateau  train to convergence: LR x0.1 after 3 epochs without a val AP50
#            gain, stop at the stall after the 2nd cut (cap 60 epochs, EDE
#            anneal over 10 epochs as in 20ep). Reports best-val weights:
#            compare test_ap50 for these runs.
#
#   export BCV_DATA=/global/D1/homes/$USER/bcv-data/raw
#   export FP32=runs/checkpoints/yolo26_early_fusion/<ts>/best.pt
#   bash scripts/slurm/submit_seeds.sh [SEEDS] [CONFIG...]
#
# SEEDS is an sbatch --array spec (default 1-2; seed 0 ran before). CONFIGs
# default to all three: fp32_control ib_guided ib_guided_attn. Extra sbatch
# options go in $SBATCH_ARGS, e.g. SBATCH_ARGS="-p hgx2q". $FP32 is the
# warm-start checkpoint (required).
#
#   SCHEDULE=plateau bash scripts/slurm/submit_seeds.sh 0-2
set -euo pipefail
cd "$(dirname "$0")/../.."

seeds="${1:-1-2}"
shift || true
configs=("$@")
[ ${#configs[@]} -gt 0 ] || configs=(fp32_control ib_guided ib_guided_attn)

FP32="${FP32:?set FP32 to the fp32 early fusion warm-start checkpoint (trained on the flight-wise split)}"
schedule="${SCHEDULE:-20ep}"
common=(train.lr=0.001 train.optimizer=adam train.warmup_epochs=1 "train.seed={task}")
case "$schedule" in
    20ep)    SCHED=(train.epochs=20 train.scheduler=cosine "${common[@]}") ;;
    # ~6 min/epoch on an A100 incl. val, so the 60-epoch cap fits in 1 day.
    plateau) SCHED=(train.epochs=60 train.scheduler=plateau train.binarize_anneal_epochs=10 "${common[@]}") ;;
    *) echo "unknown SCHEDULE: $schedule (20ep | plateau)" >&2; exit 2 ;;
esac
[ -f "$FP32" ] || { echo "missing warm-start checkpoint $FP32" >&2; exit 1; }

for c in "${configs[@]}"; do
    case "$c" in
        fp32_control)    model=(model=yolo26_early_fusion "+model.init_checkpoint=$FP32") ;;
        ib_guided)       model=(model=yolo26_early_fusion_bnn "model.init_checkpoint=$FP32") ;;
        ib_guided_attn)  model=(model=yolo26_early_fusion_bnn "model.init_checkpoint=$FP32" model.binarization.regions.attention=binary) ;;
        *) echo "unknown config: $c" >&2; exit 2 ;;
    esac
    # Seed in the run name: array tasks starting in the same second would
    # otherwise share runs/checkpoints/<name>/<timestamp>/.
    # shellcheck disable=SC2086
    sbatch -J "$c" --array="$seeds" ${SBATCH_ARGS:-} scripts/slurm/job.sbatch scripts/train_eval.py \
        "yolo26_early_fusion_${c}_${schedule}_seed{task}" "runs/bnn/${c}_${schedule}_seed{task}.json" \
        "${model[@]}" "${SCHED[@]}"
done
