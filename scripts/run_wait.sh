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

epoch=3
batch_size=8
lr=5e-6
train_emb=True
train_img_proj=True
add_img_token=True
train_spatial_attn=True
add_scene_token=False
no_obj=False
input_dim=1024 # 1024
bidirection=False
different_lr=False
max_obj_num=100
lora_r=16
lora_alpha=16
add_pos_emb=False
feat_fusion=False
fuse_with_id=False
config="/home/lcx/chat-scene/Chat-Scene/scripts/"
max_grad_norm=0.01
seed=42
use_location_token=False

llama_model_path="/home/lcx/HuggingFace-Download-Accelerator/hf_hub/models--lmsys--vicuna-7b-v1.5"

# train_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref#nr3d_caption#obj_align" 

train_tag="scanrefer#obj_align#nr3d_caption#scan2cap#scanqa#multi3dref"
# train_tag="scanqa"
# val_tag="scanqa#scan2cap#sqa3d#multi3dref"
val_tag="scanrefer#scan2cap#scanqa"

# evaluate=True
evaluate=False

debug=False
if [ $debug = "True" ]; then
    enable_wandb=False
    gpu_num=1
    do_save=False
    other_info="debug"
else
    enable_wandb=False
    gpu_num=1
    do_save=True
    other_info="chatscene"
fi

tag="${train_tag}__${val_tag}__${other_info}"

pretrained_path="/home/lcx/chat-scene/Chat-Scene/pretrained_models/ckpt_01_3446.pth"
# pretrained_path="/home/lcx/chat-scene/Chat-Scene/outputs/20250506_201448_lr5e-6_ep3_scanrefer#multi3dref#nr3d_caption#obj_align__scanrefer#multi3dref__chatscene/ckpt_00_15075.pth"


OUTPUT_DIR=outputs/"$(date +"%Y%m%d_%H%M%S")"_lr"$lr"_ep"$epoch"_"$tag"
mkdir -p ${OUTPUT_DIR}

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting training with nohup..."
# 使用nohup在后台运行训练脚本，输出日志到train.log文件
nohup python tasks/train.py \
    "${config}config.py" \
    output_dir "$OUTPUT_DIR" \
    scheduler.epochs "$epoch" \
    optimizer.lr "$lr" \
    model.add_scene_token "$add_scene_token" \
    model.add_img_token "$add_img_token" \
    pretrained_path "$pretrained_path" \
    evaluate "$evaluate" \
    wandb.enable "$enable_wandb" \
    gpu_num "$gpu_num" \
    do_save "$do_save" \
    batch_size "$batch_size" \
    model.train_emb "$train_emb" \
    model.train_img_proj "$train_img_proj" \
    train_tag "$train_tag" \
    val_tag "$val_tag" \
    model.no_obj "$no_obj" \
    segmentor "$segmentor" \
    pc_encoder "$pc_encoder" \
    model.input_dim "$input_dim" \
    model.bidirection "$bidirection" \
    optimizer.different_lr.enable "$different_lr" \
    model.max_obj_num "$max_obj_num" \
    lora.lora_r "$lora_r" \
    lora.lora_alpha "$lora_alpha" \
    model.add_pos_emb "$add_pos_emb" \
    model.feat_fusion "$feat_fusion" \
    optimizer.max_grad_norm "$max_grad_norm" \
    seed "$seed" \
    model.fuse_with_id "$fuse_with_id" \
    model.llama_model_path "$llama_model_path" \
    model.use_location_token "$use_location_token" > ${OUTPUT_DIR}/train.log 2>&1 &

echo "$(date '+%Y-%m-%d %H:%M:%S') - Training started in background with PID $!"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Check log file at ${OUTPUT_DIR}/train.log for training progress"

