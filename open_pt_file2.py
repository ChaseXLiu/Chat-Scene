import torch

# 加载模型文件
model_path = "/home/lcx/chat-scene/Chat-Scene/annotations/scannet_mask3d_uni3d_feats.pt"
checkpoint = torch.load(model_path, map_location='cpu')

# 获取指定键名的张量
key = "scene0000_00_10"
if key in checkpoint:
    tensor = checkpoint[key]
    print(f"\n{key}的详细信息:")
    print(f"形状: {tensor.shape}")
    print(f"数据类型: {tensor.dtype}")
    print(f"数值内容:")
    print(tensor)  # 打印张量的具体数值
    
    # 打印一些统计信息
    print(f"\n数值统计:")
    print(f"最小值: {tensor.min().item()}")
    print(f"最大值: {tensor.max().item()}")
    print(f"平均值: {tensor.mean().item()}")
    print(f"标准差: {tensor.std().item()}")
else:
    print(f"找不到键名 {key}")