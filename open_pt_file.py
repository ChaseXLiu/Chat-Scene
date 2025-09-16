import torch
import os

# 首先打印文件大小信息
# model_path = "/home/lcx/chat-scene/Chat-Scene/annotations/scannet_mask3d_uni3d_feats.pt"
model_path = "/home/lcx/chat-scene/Chat-Scene/annotations/scannet_train_attributes.pt"
size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"模型文件大小: {size_mb:.2f} MB")

# 使用map_location='cpu'来加载模型，避免占用GPU内存
checkpoint = torch.load(model_path, map_location='cpu')

# 创建输出文件
output_file = "model_info.txt"

# 将输出重定向到文件
with open(output_file, 'w') as f:
    # 写入文件大小信息
    f.write(f"模型文件大小: {size_mb:.2f} MB\n")
    
    # 写入模型的基本信息
    f.write("\n模型信息概览:\n")
    if isinstance(checkpoint, dict):
        f.write("\n键值列表:\n")
        for key in checkpoint.keys():
            value = checkpoint[key]
            if isinstance(value, torch.Tensor):
                f.write(f"键名: {key}, 形状: {value.shape}, 数据类型: {value.dtype}\n")
            else:
                f.write(f"键名: {key}, 类型: {type(value)}\n")
    else:
        f.write(f"模型类型: {type(checkpoint)}\n")
    
    # 如果是state_dict格式，写入参数统计
    if hasattr(checkpoint, 'state_dict'):
        state_dict = checkpoint.state_dict()
        total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        f.write(f"\n总参数量: {total_params:,}\n")

print(f"模型信息已保存到 {output_file}")

# 清理内存
del checkpoint
torch.cuda.empty_cache()  # 如果使用了GPU