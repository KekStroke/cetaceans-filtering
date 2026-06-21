#!/bin/bash
# torch-2.x port smoke (a few pretraining updates). Portable: set A2V_REPO + A2V_PY.
#   A2V_REPO=~/a2v A2V_PY=~/a2v_env/bin/python bash run_smoke.sh task.data=/path/to/manifest
set +e
A2V_REPO="${A2V_REPO:-$HOME/a2v}"; PY="${A2V_PY:-python}"
cd "$A2V_REPO" || exit 1
export PYTHONPATH="$A2V_REPO:$PYTHONPATH"           # auto-loads sitecustomize.py
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 HYDRA_FULL_ERROR=1
"$PY" animal2vec_train.py --config-name smoke_pretrain optimization.max_update=20 "$@"
echo "EXITCODE=$?"
