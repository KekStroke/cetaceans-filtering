#!/usr/bin/env bash
set -Eeuo pipefail

OUTER_RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-0}}}"
if [[ "${OUTER_RANK}" != "0" ]]; then
  echo "Outer launcher rank ${OUTER_RANK} is not the training owner; exiting to avoid duplicate training processes."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_SCRIPT="${SCRIPT_DIR}/../smoke_16khz_5s_torch2/run_job.sh"

export ANIMAL2VEC_CONFIG_NAME="${ANIMAL2VEC_CONFIG_NAME:-my_domain/a2v_16khz_5s_lr1e4_epoch1_torch2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec "${SMOKE_SCRIPT}" "$@"
