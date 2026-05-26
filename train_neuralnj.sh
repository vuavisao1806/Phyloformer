#!/usr/bin/env bash
# Train Phyloformer on the NeuralNJ dataset.
#
# Early stopping protocol (NeuralNJ paper, Section B.1):
#   - Validation is checked every 10,000 training steps.
#   - Training stops when val_loss shows no improvement for
#     5 consecutive validation checks (= 50,000 steps of no progress).
#
# Data splits:
#   train/      → training examples
#   validation/ → early-stopping monitor (independent from train and test)
#   test/       → NEVER touched during training; reserved for final benchmark
#
# Usage:
#   bash train_neuralnj.sh [output_dir] [run_name] [base_model.ckpt]
#
# Examples:
#   bash train_neuralnj.sh
#   bash train_neuralnj.sh ./runs/exp1 my_run
#   bash train_neuralnj.sh ./runs/exp2 finetune ./models/pf.ckpt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_DIR="${SCRIPT_DIR}/data_from_NeuralNJ/train"
VAL_DIR="${SCRIPT_DIR}/data_from_NeuralNJ/validation"
# test/ is intentionally omitted — reserved for final evaluation only.

OUTPUT_DIR="${1:-${SCRIPT_DIR}/runs/neuralnj}"
RUN_NAME="${2:-phyloformer_neuralnj}"
BASE_MODEL="${3:-}"          # optional: path to a .ckpt to fine-tune from

# ── Architecture (Phyloformer original parameters) ────────────────────────────
NB_BLOCKS=6
NB_HEADS=4
EMBED_DIM=64
DROPOUT=0.0

# ── Optimiser ────────────────────────────────────────────────────────────────
LEARNING_RATE=1e-4
WARMUP_STEPS=5000

# ── Training schedule ────────────────────────────────────────────────────────
NB_EPOCHS=300               # generous ceiling; early stopping will cut this short
BATCH_SIZE=4

# ── Early stopping (NeuralNJ protocol, Section B.1) ──────────────────────────
CHECK_VAL_EVERY=10000       # validate every 10,000 training steps
NO_IMPROVE_STOP=5           # stop after 5 consecutive checks with no improvement

# ── Divergence guard (Phyloformer safety mechanism, not from NeuralNJ) ────────
# Aborts training if train_loss exceeds this value, indicating divergence.
# 3.0 is the train_distributed.py default. With L1 loss on DNA pairwise
# distances (typical range 0–1.5), anything above 3.0 signals a broken run.
HARD_LOSS_CEILING=3.0

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_EVERY=100

# ── Sanity checks ────────────────────────────────────────────────────────────
if [ ! -d "${TRAIN_DIR}" ]; then
    echo "ERROR: training directory not found: ${TRAIN_DIR}" >&2
    exit 1
fi
if [ ! -d "${VAL_DIR}" ]; then
    echo "ERROR: validation directory not found: ${VAL_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "  Phyloformer × NeuralNJ dataset — training run"
echo "============================================================"
echo "  Train      : ${TRAIN_DIR}"
echo "  Validation : ${VAL_DIR}"
echo "  Test       : (reserved — not used during training)"
echo "  Output     : ${OUTPUT_DIR}"
echo "  Run name   : ${RUN_NAME}"
echo ""
echo "  Early stopping (NeuralNJ protocol):"
echo "    check every   ${CHECK_VAL_EVERY} steps"
echo "    patience      ${NO_IMPROVE_STOP} consecutive checks"
echo "    loss ceiling  ${HARD_LOSS_CEILING} (divergence guard)"
echo "============================================================"

# ── Build argument list ───────────────────────────────────────────────────────
ARGS=(
    --train-trees      "${TRAIN_DIR}"
    --train-alignments "${TRAIN_DIR}"
    --val-trees        "${VAL_DIR}"
    --val-alignments   "${VAL_DIR}"
    --nb-blocks        "${NB_BLOCKS}"
    --nb-heads         "${NB_HEADS}"
    --embed-dim        "${EMBED_DIM}"
    --dropout          "${DROPOUT}"
    --learning-rate    "${LEARNING_RATE}"
    --warmup-steps     "${WARMUP_STEPS}"
    --nb-epochs        "${NB_EPOCHS}"
    --batch-size       "${BATCH_SIZE}"
    --check-val-every  "${CHECK_VAL_EVERY}"
    --no-improvement-stop "${NO_IMPROVE_STOP}"
    --hard-loss-ceiling   "${HARD_LOSS_CEILING}"
    --log-every        "${LOG_EVERY}"
    --output-dir       "${OUTPUT_DIR}"
    --run-name         "${RUN_NAME}"
)

if [ -n "${BASE_MODEL}" ]; then
    ARGS+=(--base-model "${BASE_MODEL}")
fi

python "${SCRIPT_DIR}/train_distributed.py" "${ARGS[@]}"
