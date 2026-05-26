#!/usr/bin/env bash
# setup_env.sh — Create and configure the conda environment for Phyloformer.
#
# Requirements: conda (or mamba) must be installed.
# Python version: 3.9 (required by Phyloformer; setup.py enforces < 3.10)
#
# Usage:
#   bash setup_env.sh [env_name]
#
# Examples:
#   bash setup_env.sh          # creates env named 'phylo'
#   bash setup_env.sh myenv    # creates env named 'myenv'

set -euo pipefail

ENV_NAME="${1:-phylo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 0. Locate conda ───────────────────────────────────────────────────────────
if command -v mamba &>/dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &>/dev/null; then
    CONDA_CMD="conda"
else
    echo "ERROR: conda (or mamba) not found. Please install Anaconda or Miniconda first." >&2
    exit 1
fi

echo "Using: ${CONDA_CMD}"

# ── 1. Create conda environment ───────────────────────────────────────────────
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' already exists — skipping creation."
else
    echo "Creating conda environment '${ENV_NAME}' with Python 3.9..."
    ${CONDA_CMD} create -n "${ENV_NAME}" python=3.9 -y
fi

# ── 2. Activate environment ───────────────────────────────────────────────────
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "Activated environment: ${ENV_NAME}"

# ── 3. Detect CUDA version and select PyTorch wheel ──────────────────────────
if command -v nvidia-smi &>/dev/null; then
    # nvidia-smi output: "| ... CUDA Version: 12.8   |"
    # grep -oP extracts only the version number, ignoring the table border "|"
    CUDA_FULL=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)

    if [[ -z "${CUDA_FULL}" ]]; then
        echo "WARN: Found nvidia-smi but could not parse CUDA version — falling back to CPU-only PyTorch." >&2
        TORCH_EXTRA="cpu"
    else
        CUDA_MAJOR=$(echo "${CUDA_FULL}" | cut -d. -f1)
        CUDA_MINOR=$(echo "${CUDA_FULL}" | cut -d. -f2)
        echo "Detected GPU with CUDA ${CUDA_FULL}"

        # Map CUDA version to the closest PyTorch wheel index.
        # PyTorch publishes separate wheel sets for each CUDA release.
        if   [[ "${CUDA_MAJOR}" -eq 12 ]] && [[ "${CUDA_MINOR}" -ge 8 ]]; then
            TORCH_EXTRA="cu128"   # CUDA 12.8+ → torch 2.7+
        elif [[ "${CUDA_MAJOR}" -eq 12 ]] && [[ "${CUDA_MINOR}" -ge 4 ]]; then
            TORCH_EXTRA="cu124"   # CUDA 12.4–12.7 → torch 2.4+
        elif [[ "${CUDA_MAJOR}" -eq 12 ]]; then
            TORCH_EXTRA="cu121"   # CUDA 12.0–12.3 → torch 2.1+
        elif [[ "${CUDA_MAJOR}" -eq 11 ]] && [[ "${CUDA_MINOR}" -ge 8 ]]; then
            TORCH_EXTRA="cu118"   # CUDA 11.8–11.x → torch 2.0+
        elif [[ "${CUDA_MAJOR}" -eq 11 ]] && [[ "${CUDA_MINOR}" -ge 7 ]]; then
            TORCH_EXTRA="cu117"   # CUDA 11.7
        else
            echo "WARN: CUDA ${CUDA_FULL} is older than 11.7 — falling back to CPU-only PyTorch." >&2
            TORCH_EXTRA="cpu"
        fi
    fi
else
    echo "No NVIDIA GPU detected — installing CPU-only PyTorch."
    TORCH_EXTRA="cpu"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_EXTRA}"
echo "Installing latest stable torch for ${TORCH_EXTRA} ..."
# No version pin: installs the newest torch compatible with the detected CUDA.
# Pinning to 2.0.1 is intentionally avoided because newer CUDA versions
# (12.4, 12.8) have no 2.0.1 wheels; torch 2.x APIs used here are stable.
pip install torch --index-url "${TORCH_INDEX}"

# ── 4. Install remaining dependencies ────────────────────────────────────────
# Pin lightning first so pip's resolver doesn't downgrade it via transitive
# dependencies (e.g. wandb<=0.15.x → lightning-cloud → lightning==2.0.x).
echo "Installing lightning>=2.2.0 ..."
pip install 'lightning>=2.2.0,<3'

echo "Installing Python dependencies from requirements.txt ..."
pip install --upgrade -r "${SCRIPT_DIR}/requirements.txt"

# Force the correct lightning version last. wandb<=0.15.x pulls lightning-cloud
# which downgrades lightning to 2.0.x even if requirements.txt says >=2.2.0.
# Re-installing after all other packages wins the version contest.
echo "Ensuring lightning>=2.2.0 (overrides any transitive downgrade) ..."
pip install 'lightning>=2.2.0,<3' --upgrade

# ── 5. Install the phyloformer package in editable mode ──────────────────────
echo "Installing phyloformer package (editable) ..."
pip install -e "${SCRIPT_DIR}" --no-deps

# ── 6. Smoke-test the training stack ─────────────────────────────────────────
echo ""
echo "Running import checks..."

python - <<'EOF'
import sys

checks = {
    "torch":        "import torch; print(f'  torch         {torch.__version__}  |  CUDA: {torch.cuda.is_available()}')",
    "lightning":    "import lightning; print(f'  lightning     {lightning.__version__}')",
    "dendropy":     "import dendropy; print(f'  dendropy      {dendropy.__version__}')",
    "scipy":        "import scipy; print(f'  scipy         {scipy.__version__}')",
    "transformers": "import transformers; print(f'  transformers  {transformers.__version__}')",
    "wandb":        "import wandb; print(f'  wandb         {wandb.__version__}')",
    "phyloformer":  "from phyloformer.data import load_alignment; from phyloformer.model import Phyloformer; print('  phyloformer   OK')",
}

failed = []
for name, stmt in checks.items():
    try:
        exec(stmt)
    except Exception as e:
        print(f"  {name:<14} FAILED — {e}")
        failed.append(name)

if failed:
    print(f"\nERROR: {len(failed)} package(s) failed to import: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\nAll checks passed.")
EOF

# ── 7. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Activate the environment with:"
echo ""
echo "      conda activate ${ENV_NAME}"
echo ""
echo "  Then start training:"
echo ""
echo "      bash train_neuralnj.sh"
echo "============================================================"
