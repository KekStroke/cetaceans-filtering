#!/bin/bash
# torch.compile OFF-vs-ON training benchmark. Portable: set A2V_REPO (animal2vec repo) and A2V_PY (env python).
#   A2V_REPO=~/a2v A2V_PY=~/a2v_env/bin/python bash run_bench.sh
# Point the config's task.data at your manifest (or override below), e.g. task.data=/path/to/manifest.
set +e
A2V_REPO="${A2V_REPO:-$HOME/a2v}"; PY="${A2V_PY:-python}"
cd "$A2V_REPO" || exit 1
export PYTHONPATH="$A2V_REPO:$PYTHONPATH"           # auto-loads sitecustomize.py
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 HYDRA_FULL_ERROR=1
OV="optimization.max_update=120 common.log_interval=1 model.clone_batch=4 dataset.max_tokens=160000 ${*}"

echo "### compile OFF ###"
A2V_COMPILE_BLOCKS=0 "$PY" animal2vec_train.py --config-name smoke_pretrain $OV 2>&1 | tee bench_off.log | grep -aE "done training|pretrain_inner" | tail -2
echo "### compile ON (default) ###"
A2V_COMPILE_BLOCKS=1 A2V_COMPILE_MODE=default "$PY" animal2vec_train.py --config-name smoke_pretrain $OV 2>&1 | tee bench_on.log | grep -aE "done training|compile blocks ON" | tail -2

# steady-state ms/update (discard first 50 warmup), compare OFF vs ON
"$PY" - <<'PYEOF'
import json, re, statistics as st, os
def ms(p):
    u=[1/float(json.loads(m.group(1))["ups"]) for ln in open(p,errors='ignore')
       for m in [re.search(r'(\{.*"ups".*\})',ln)] if m and float(json.loads(m.group(1)).get("ups",0))>0]
    s=u[50:] or u[len(u)//2:] or u; return (st.mean(s)*1000 if s else 0, len(s))
for tag,p in [("OFF","bench_off.log"),("ON ","bench_on.log")]:
    if os.path.exists(p):
        m,n=ms(p); print(f"{tag}: {m:.1f} ms/update (n={n})")
PYEOF
