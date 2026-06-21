set +e
cd /home/yarix/a2v
export PYTHONPATH=/home/yarix/a2v:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 HYDRA_FULL_ERROR=1
PY=/home/yarix/a2v_env/bin/python
OV="optimization.max_update=120 common.log_interval=1 model.clone_batch=4 dataset.max_tokens=160000 checkpoint.save_dir=/tmp/a2v_bench_ck"
( while true; do A=$(free -m|awk 'NR==2{print $7}'); [ "$A" -lt 3500 ]&&{ echo "WATCHDOG KILL ${A}MB"; pkill -9 -f animal2vec_train.py; break;}; sleep 4; done ) & WD=$!
trap "kill $WD 2>/dev/null" EXIT
echo "############ COMPILE OFF ############"
unset A2V_COMPILE_BLOCKS
$PY animal2vec_train.py --config-name smoke_pretrain $OV 2>&1 | tee /home/yarix/bench_off.log | grep -aE "pretrain_inner|Error|Traceback|done training|compile" | tail -3
echo "off exit=$?"
echo "############ COMPILE ON (default mode) ############"
export A2V_COMPILE_BLOCKS=1 A2V_COMPILE_MODE=default
$PY animal2vec_train.py --config-name smoke_pretrain $OV 2>&1 | tee /home/yarix/bench_on.log | grep -aE "pretrain_inner|Error|Traceback|done training|compile|graph break" | tail -3
echo "on exit=$?"
echo "BENCH-DONE"
