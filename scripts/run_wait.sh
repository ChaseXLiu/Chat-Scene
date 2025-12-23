# bash scripts/run_wait.sh
# nohup bash scripts/run_wait.sh > wait.log 2>&1 &

which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=$((54000 + $RANDOM % 10000))
export MASTER_ADDR=localhost
# export CUDA_LAUNCH_BLOCKING=1

# Function to check GPU memory
check_gpu_memory() {
    gpu_id=$1
    free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_id | xargs)
    echo $free_memory
}

# Wait for GPU memory to be sufficient
wait_for_gpu() {
    while true; do
        # Check GPU 0 first
        gpu_0_memory=$(check_gpu_memory 0)
        if [ "$gpu_0_memory" -gt "23000" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - GPU 0 has sufficient memory: ${gpu_0_memory}MB"
            export CUDA_VISIBLE_DEVICES=0
            return
        fi
        
        # Check GPU 1 if GPU 0 doesn't have enough memory
        gpu_1_memory=$(check_gpu_memory 1)
        if [ "$gpu_1_memory" -gt "23000" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - GPU 1 has sufficient memory: ${gpu_1_memory}MB"
            export CUDA_VISIBLE_DEVICES=1
            return
        fi
        
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Neither GPU 0 (${gpu_0_memory}MB) nor GPU 1 (${gpu_1_memory}MB) has sufficient memory. Waiting..."
        sleep 30
    done
}

# Wait for sufficient GPU memory
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting to wait for GPU memory..."
wait_for_gpu
echo "$(date '+%Y-%m-%d %H:%M:%S') - GPU memory is sufficient, starting training..."

# Source configuration from run.sh
source scripts/run.sh

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting training..."

# 方式1：前台运行（直接在当前终端显示输出，等同于执行 bash scripts/run.sh）
python tasks/train.py "${ARGS[@]}"

# 方式2：后台运行（使用nohup在后台运行，输出日志到train.log文件）
# nohup python tasks/train.py "${ARGS[@]}" > ${OUTPUT_DIR}/train.log 2>&1 &

echo "$(date '+%Y-%m-%d %H:%M:%S') - Training started in background with PID $!"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Check log file at ${OUTPUT_DIR}/train.log for training progress"