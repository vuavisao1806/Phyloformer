#!/usr/bin/env bash
# Train Phyloformer on the NeuralNJ dataset.
#
# Parameters follow Phyloformer paper (main text Online Methods + supplementary Table 1, PF_Base):
#
#   Four parameters DIFFER from train_distributed.py defaults — must be explicit:
#     --learning-rate   1e-3   (code default: 1e-4   | paper: "maximum learning rate of 10^-3")
#     --warmup-steps    3000   (code default: 5000   | paper: "3,000 linear warmup steps")
#     --check-val-every 3000   (code default: 10000  | paper: "five successive 3,000 step intervals")
#     --batch-size      1      (code default: 4      | NeuralNJ mixes sequence lengths → batch_size=1
#                               avoids collation errors; paper batch_size=4 assumed fixed-length data)
#
#   All other hyperparameters already match paper values in train_distributed.py:
#     nb_epochs=100   → ~161,000 total schedule steps; early stopping terminates sooner
#     nb_blocks=6  | embed_dim=64 | nb_heads=4 | dropout=0.0
#     loss=MAE (L1Loss) | no_improvement_stop=5 | hard_loss_ceiling=3.0
#
# Data splits:
#   train/      → training (6,438 pairs)
#   validation/ → early-stopping monitor (96 pairs, independent from test)
#   test/       → NEVER touched during training; reserved for final benchmark
#
# Usage:
#   bash train_neuralnj.sh [output_dir] [run_name] [base_model.ckpt]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_DIR="${SCRIPT_DIR}/data_from_NeuralNJ/train"
VAL_DIR="${SCRIPT_DIR}/data_from_NeuralNJ/validation"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/runs/neuralnj}"
RUN_NAME="${2:-phyloformer_neuralnj}"
BASE_MODEL="${3:-}"

if [ ! -d "${TRAIN_DIR}" ]; then
    echo "ERROR: training directory not found: ${TRAIN_DIR}" >&2; exit 1
fi
if [ ! -d "${VAL_DIR}" ]; then
    echo "ERROR: validation directory not found: ${VAL_DIR}" >&2; exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "  Phyloformer × NeuralNJ dataset — training run"
echo "  Train      : ${TRAIN_DIR}"
echo "  Validation : ${VAL_DIR}"
echo "  Test       : (reserved — not used during training)"
echo "  Output     : ${OUTPUT_DIR}"
echo "  lr=1e-3 | warmup=3000 | check_val_every=3000 (Phyloformer paper)"
echo "  All other hyperparameters: train_distributed.py defaults"
echo "============================================================"

ARGS=(
    --train-trees      "${TRAIN_DIR}"
    --train-alignments "${TRAIN_DIR}"
    --val-trees        "${VAL_DIR}"
    --val-alignments   "${VAL_DIR}"
    --learning-rate    1e-3
    --warmup-steps     3000
    --check-val-every  3000
    --batch-size       1
    --output-dir       "${OUTPUT_DIR}"
    --run-name         "${RUN_NAME}"
)

if [ -n "${BASE_MODEL}" ]; then
    ARGS+=(--base-model "${BASE_MODEL}")
fi

python "${SCRIPT_DIR}/train_distributed.py" "${ARGS[@]}"
