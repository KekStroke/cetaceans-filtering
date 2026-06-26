#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

ENV_ROOT="${ANIMAL2VEC_ENV_ROOT:-/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/envs/animal2vec-torch2}"
LOG_DIR="${ANIMAL2VEC_LOG_DIR:-/home/jovyan/iskhakov_anvar_local/logs/animal2vec_train_logs}"
mkdir -p "${ENV_ROOT}" "${LOG_DIR}"

PYTHON="${ENV_ROOT}/bin/python"

env_ready() {
  [[ -x "${PYTHON}" ]] || return 1
  "${PYTHON}" - <<'PY'
from packaging import version
import torch
assert version.parse(torch.__version__.split("+", 1)[0]) >= version.parse("2.0")
import fairseq  # noqa: F401
import hydra  # noqa: F401
import soundfile  # noqa: F401
PY
}

if ! env_ready; then
  rm -rf "${ENV_ROOT}"
  python3 -m venv --system-site-packages "${ENV_ROOT}"
  "${PYTHON}" -m pip install --upgrade "pip<24.1" wheel setuptools
  "${PYTHON}" -m pip install -r "${REPO_ROOT}/animal2vec/requirements-torch2.txt"
  env_ready
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG_NAME="${ANIMAL2VEC_CONFIG_NAME:-my_domain/a2v_smoke_16khz_5s_torch2}"
LOG_FILE="${LOG_DIR}/${CONFIG_NAME//\//_}_$(date +%Y%m%d_%H%M%S).log"
TRAIN_ARGS=("$@")

if [[ "${CONFIG_NAME}" == *"a2v_smoke"* ]]; then
  FULL_MANIFEST_DIR="${ANIMAL2VEC_FULL_MANIFEST_DIR:-/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/datasets/audio/marine-mammal/animal2vec_pretraining_data/cetaceans_16khz_5s_2026-06-25_a2v/manifest}"
  SMOKE_ROWS="${ANIMAL2VEC_SMOKE_ROWS:-256}"
  SMOKE_MANIFEST_DIR="${ANIMAL2VEC_SMOKE_MANIFEST_DIR:-/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/datasets/audio/marine-mammal/animal2vec_pretraining_data/cetaceans_16khz_5s_2026-06-25_a2v/smoke_${SMOKE_ROWS}_manifest}"
  mkdir -p "${SMOKE_MANIFEST_DIR}"
  head -n "$((SMOKE_ROWS + 1))" "${FULL_MANIFEST_DIR}/pretrain.tsv" > "${SMOKE_MANIFEST_DIR}/pretrain.tsv"
  : > "${SMOKE_MANIFEST_DIR}/valid_0.tsv"
  TRAIN_ARGS=("task.data=${SMOKE_MANIFEST_DIR}" "${TRAIN_ARGS[@]}")
fi

echo "repo_root=${REPO_ROOT}"
echo "env_root=${ENV_ROOT}"
echo "config_name=${CONFIG_NAME}"
echo "log_file=${LOG_FILE}"
echo "train_args=${TRAIN_ARGS[*]}"
"${PYTHON}" -m animal2vec.train --config-name "${CONFIG_NAME}" "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
