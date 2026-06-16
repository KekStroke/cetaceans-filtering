#!/bin/bash
# Turnkey validation-dynamics driver. Drop checkpoints from the parallel runs, get the comparison.
# Usage:
#   bash run_dynamics.sh RUN:STEP:CKPT [RUN:STEP:CKPT ...]
# Example (tomorrow, after Anvar sends checkpoints):
#   bash run_dynamics.sh \
#     "blue:30000:/home/yarix/a2v_ckpts/blue_30k.pt" \
#     "orange:30000:/home/yarix/a2v_ckpts/orange_30k.pt"
# Runs BOTH probes (Watkins + filtration) on each checkpoint, ONE process at a time (memory-safe,
# RAM watchdog), registers run+step, then regenerates a2v_dynamics.png + the table. Re-run any day;
# results accumulate. Slim the checkpoints first if they are the raw ~5GB ones (a2v_extract slims
# on load, but pre-slimming to ~/a2v_ckpts keeps RAM low).
set -e
cd /home/yarix/a2v
export TF_CPP_MIN_LOG_LEVEL=3
PY=/home/yarix/a2v_env/bin/python
VD=/mnt/c/Users/Iaroslav/CETACEANS/a2v_validation
for spec in "$@"; do
  RUN="${spec%%:*}"; rest="${spec#*:}"; STEP="${rest%%:*}"; CKPT="${rest#*:}"
  TAG=$(basename "$CKPT"); TAG="${TAG%_slim.pt}"; TAG="${TAG%.pt}"
  echo "############ $RUN  step=$STEP  tag=$TAG ############"
  # RAM watchdog for this checkpoint's GPU work
  ( while true; do A=$(free -m | awk 'NR==2{print $7}'); [ "$A" -lt 3500 ] && { echo "WATCHDOG KILL ${A}MB"; pkill -9 -f a2v_watkins.py; pkill -9 -f a2v_filter.py; break; }; sleep 5; done ) &
  WD=$!
  $PY "$VD/a2v_watkins.py" "$CKPT"     # species + clustering (accumulates into animal2vec_watkins.json)
  $PY "$VD/a2v_filter.py"  "$CKPT"     # signal/noise (accumulates into a2v_filter.json)
  kill $WD 2>/dev/null || true
  # register run+step for this tag
  $PY - "$TAG" "$RUN" "$STEP" <<'PY'
import json, os, sys
p = "/mnt/c/Users/Iaroslav/CETACEANS/a2v_validation/dynamics_registry.json"
r = json.load(open(p)) if os.path.exists(p) else {}
r[sys.argv[1]] = {"run": sys.argv[2], "step": int(sys.argv[3])}
json.dump(r, open(p, "w"), indent=2)
print(f"registered {sys.argv[1]} -> {sys.argv[2]} @ {sys.argv[3]}")
PY
done
$PY "$VD/a2v_dynamics.py"
echo "############ DYNAMICS UPDATED -> a2v_dynamics.png ############"
