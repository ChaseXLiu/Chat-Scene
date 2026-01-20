# conda activate chat-scene
# CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh
# CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh

# nohup bash scripts/run.sh > output.log 2>&1 &
# nohup bash -c "CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh" > output.log 2>&1 &

# gpustat -ui
# pkill -u lcx

which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=$((54000 + $RANDOM % 10000))
export MASTER_ADDR=localhost

epoch=3
batch_size=16
lr=5e-6
# lr=5e-7
train_emb=False
train_img_proj=True
train_spatial_attn=True
add_img_token=True
add_scene_token=False
no_obj=False
input_dim=1024 # 1024
bidirection=False
different_lr=True
max_obj_num=100
lora_r=16
lora_alpha=16
add_pos_emb=False
feat_fusion=False
fuse_with_id=False
config="/data/ZXMIC/mic_lcx/Chat-Scene/Chat-Scene/scripts/"
max_grad_norm=0.01
seed=42
use_location_token=False
stage=2

llama_model_path="/data/ZXMIC/mic_lcx/HuggingFace-Download-Accelerator/hf_hub/models--lmsys--vicuna-7b-v1.5"

# train_tag="scanrefer#obj_align#nr3d_caption#scan2cap#scanqa#multi3dref"
# val_tag="scanrefer#multi3dref#scan2cap#scanqa"

# train_tag="scanrefer#obj_align#nr3d_caption#scan2cap#scanqa#sqa3d#multi3dref"
# val_tag="scanrefer#scanqa#scan2cap#sqa3d#multi3dref"

train_tag="scanrefer#obj_align#nr3d_caption#scanqa"
val_tag="scanrefer#scanqa"



# evaluate=True
evaluate=False

debug=False
if [ $debug = "True" ]; then
    enable_wandb=False
    gpu_num=1
    do_save=False
    other_info="debug"
else
    # enable_wandb=True
    enable_wandb=False
    gpu_num=2
    do_save=True
    other_info="chatscene"
fi

tag="${train_tag}__${val_tag}__${other_info}"

# pretrained_path="/data/ZXMIC/mic_lcx/Chat-Scene/Chat-Scene/pretrained_models/our_ckpt_00_7649.pth"
pretrained_path="/data/ZXMIC/mic_lcx/Chat-Scene/Chat-Scene/pretrained_models/ckpt_01_3446.pth"


# OUTPUT_DIR=outputs/"$(date +"%Y%m%d_%H%M%S")"_lr"$lr"_ep"$epoch"_"$tag"
OUTPUT_DIR=/data/ZXMIC/mic_lcx/Chat-Scene/Chat-Scene/outputs/"$(date +"%Y%m%d_%H%M%S")"_lr"$lr"_ep"$epoch"_"$tag"
# OUTPUT_DIR=/data/lcx/chat-scene/outputs/ours"$(date +"%Y%m%d_%H%M%S")"_lr"$lr"_ep"$epoch"_"$tag"
mkdir -p ${OUTPUT_DIR}

ARGS=(
    "${config}config.py"
    output_dir "$OUTPUT_DIR"
    scheduler.epochs "$epoch"
    optimizer.lr "$lr"
    model.add_scene_token "$add_scene_token"
    model.add_img_token "$add_img_token"
    pretrained_path "$pretrained_path"
    evaluate "$evaluate"
    wandb.enable "$enable_wandb"
    gpu_num "$gpu_num"
    do_save "$do_save"
    batch_size "$batch_size"
    model.train_emb "$train_emb"
    model.train_img_proj "$train_img_proj"
    train_tag "$train_tag"
    val_tag "$val_tag"
    model.no_obj "$no_obj"
    segmentor "$segmentor"
    pc_encoder "$pc_encoder"
    model.input_dim "$input_dim"
    model.bidirection "$bidirection"
    optimizer.different_lr.enable "$different_lr"
    model.max_obj_num "$max_obj_num"
    lora.lora_r "$lora_r"
    lora.lora_alpha "$lora_alpha"
    model.add_pos_emb "$add_pos_emb"
    model.feat_fusion "$feat_fusion"
    optimizer.max_grad_norm "$max_grad_norm"
    seed "$seed"
    model.fuse_with_id "$fuse_with_id"
    model.llama_model_path "$llama_model_path"
    model.use_location_token "$use_location_token"
    model.stage "$stage"
)

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # srun --partition=mozi-S1 --gres=gpu:${gpu_num} --ntasks-per-node=${gpu_num} --kill-on-bad-exit --quotatype=reserved \
    
    if [ "$gpu_num" -gt 1 ]; then
        # Multi-GPU training
        # Generate device string (e.g., "0,1" for 2 GPUs)
        devices=$(seq -s, 0 $(($gpu_num - 1)))
        export CUDA_VISIBLE_DEVICES=$devices
        echo "Running on $gpu_num GPUs (CUDA_VISIBLE_DEVICES=$devices) with torchrun..."
        torchrun --nproc_per_node=${gpu_num} --master_port=${MASTER_PORT} tasks/train.py "${ARGS[@]}"
    else
        # Single-GPU training
        export CUDA_VISIBLE_DEVICES=0
        echo "Running on single GPU (CUDA_VISIBLE_DEVICES=0) with python..."
        python tasks/train.py "${ARGS[@]}"
    fi
fi

