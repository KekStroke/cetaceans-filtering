set +e
cd /home/yarix/a2v
export PYTHONPATH=/home/yarix/a2v:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 HYDRA_FULL_ERROR=1
( while true; do A=$(free -m|awk 'NR==2{print $7}'); [ "$A" -lt 3500 ]&&{ echo "WATCHDOG KILL ${A}MB"; pkill -9 -f animal2vec_train.py; break;}; sleep 4; done ) & WD=$!
trap "kill $WD 2>/dev/null" EXIT
/home/yarix/a2v_env/bin/python animal2vec_train.py --config-name smoke_pretrain
echo "EXITCODE=$?"
echo "SMOKE-DONE"
