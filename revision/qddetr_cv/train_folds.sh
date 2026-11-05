#!/bin/bash
# Train the five held-out QD-DETR models and the in-sample reference model (see make_folds.py).
#
# Every hyperparameter is identical to the checkpoint used in the original
# submission (qd_detr/scripts/train_olympic.sh defaults: 200 epochs, bsz 32,
# lr 1e-4, seed 2018, SlowFast+CLIP video features, CLIP text features of the
# event-specific training queries). As in the original run, model selection
# ("model_best.ckpt") only uses the training split of the fold.
#
# Usage: bash revision/qddetr_cv/train_folds.sh [GPU_A] [GPU_B] [FOLD_DIR] [EXP_PREFIX]
#   default: folds/ and exp_id olympic_bench_heldout_fold{k} (each fold model is trained on
#   the 16 benchmark broadcasts outside its fold); the in-sample reference model
#   olympic_bench_all is trained on all 20 benchmark broadcasts.

set -u
QDDETR_DIR=/home/mrt/Projects/QD-DETR
GPU_A=${1:-0}
GPU_B=${2:-1}
FOLD_DIR=${3:-$(cd "$(dirname "$0")" && pwd)/folds}
EXP_PREFIX=${4:-olympic_bench_heldout_fold}

train_fold() {
    local fold=$1 gpu=$2
    local split="${FOLD_DIR}/fold${fold}_train_specific.jsonl"
    (
        cd "$QDDETR_DIR" || exit 1
        source venv/bin/activate
        CUDA_VISIBLE_DEVICES=$gpu bash qd_detr/scripts/train_olympic.sh \
            --train_path "$split" \
            --eval_path "$split" \
            --exp_id "${EXP_PREFIX}${fold}" \
            --seed 2018 \
            > "${FOLD_DIR}/train_fold${fold}.log" 2>&1
    )
    echo "fold ${fold} finished (exit $?)"
}

# Two folds at a time, one per GPU
train_fold 0 "$GPU_A" & train_fold 1 "$GPU_B" & wait
train_fold 2 "$GPU_A" & train_fold 3 "$GPU_B" & wait
train_fold 4 "$GPU_A" &
(
    cd "$QDDETR_DIR" || exit 1
    source venv/bin/activate
    CUDA_VISIBLE_DEVICES=$GPU_B bash qd_detr/scripts/train_olympic.sh \
        --train_path "${FOLD_DIR}/all_benchmark_train_specific.jsonl" \
        --eval_path "${FOLD_DIR}/all_benchmark_train_specific.jsonl" \
        --exp_id olympic_bench_all --seed 2018 > "${FOLD_DIR}/train_all.log" 2>&1
) &
wait
