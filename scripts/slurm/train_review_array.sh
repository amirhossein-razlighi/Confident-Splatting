#!/usr/bin/env bash
#SBATCH --job-name=cs_review_train
#SBATCH --account=def-amahdavi
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

MANIFEST="${MANIFEST:-experiments/review_manifest.tsv}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH_ROOT}/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${SCRATCH_ROOT}/.local/share/uv/python}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${HF_HOME}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 2
fi

ROW="$(awk -F '\t' -v idx="${TASK_ID}" 'NR == idx + 2 {print}' "${MANIFEST}")"
if [[ -z "${ROW}" ]]; then
  echo "No manifest row for SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
  exit 2
fi

IFS=$'\t' read -r SCENE VARIANT CONFIG DATA_DIR RESULT_DIR EXTRA_ARGS_STR <<< "${ROW}"
mkdir -p "${RESULT_DIR}"

COMMON_TRAIN_ARGS="${COMMON_TRAIN_ARGS:---disable-viewer --tb-every 250 --max-steps 30000 --eval-steps 30000 --save-steps 30000}"
read -r -a COMMON_ARGS <<< "${COMMON_TRAIN_ARGS}"
read -r -a EXTRA_ARGS <<< "${EXTRA_ARGS_STR}"

echo "Scene: ${SCENE}"
echo "Variant: ${VARIANT}"
echo "Config: ${CONFIG}"
echo "Data: ${DATA_DIR}"
echo "Results: ${RESULT_DIR}"
echo "Common args: ${COMMON_TRAIN_ARGS}"
echo "Extra args: ${EXTRA_ARGS_STR}"

uv sync --frozen

uv run python trainer.py "${CONFIG}" \
  --data-dir "${DATA_DIR}" \
  --result-dir "${RESULT_DIR}" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
