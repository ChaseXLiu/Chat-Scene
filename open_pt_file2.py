import torch
import sys

# 设置输出重定向到文件
output_file = "tensor_analysis_output.txt"
sys.stdout = open(output_file, 'w', encoding='utf-8')

# 设置GPU设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型文件
# model_path = "/home/lcx/chat-scene/Chat-Scene/annotations/scannet_mask3d_uni3d_feats.pt"
model_path = "/home/lcx/chat-scene/Chat-Scene/annotations/scannet_train_attributes.pt"
checkpoint = torch.load(model_path, map_location=device)

# 获取指定键名的张量
# key = "scene0000_00_10"
key = "scene0191_00"
if key in checkpoint:
    data = checkpoint[key]
    print(f"\n{key}的详细信息:")
    
    # 检查数据类型
    if isinstance(data, dict):
        print(f"数据类型: 字典")
        print(f"字典大小: {len(data)}")
        print(f"字典键: {list(data.keys())}")
        
        # 可以选择打印字典中的一些值
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                # 移动张量到指定设备
                v = v.to(device)
                print(f"\n子键 '{k}' 的详细信息:{v}")
                print(f"形状: {v.shape}")
                print(f"数据类型: {v.dtype}")
                print(f"数值统计:")
                print(f"最小值: {v.min().item()}")
                print(f"最大值: {v.max().item()}")
                print(f"平均值: {v.mean().item()}")
                print(f"标准差: {v.std().item()}")
            else:
                print(f"\n子键 '{k}' 的值: {v}")
                if k == 'objects':
                    # 查找'table'的索引
                    table_indices = [i for i, obj in enumerate(v) if obj == 'table']
                    if table_indices:
                        print(f"找到'table'的索引: {table_indices}")
                        # 获取对应的位置信息
                        if 'locs' in data:
                            table_locs = data['locs'][table_indices]
                            print(f"'table'对应的位置信息:\n{table_locs}")
    elif isinstance(data, torch.Tensor):
        # 移动张量到指定设备
        data = data.to(device)
        print(f"形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"数值内容:")
        print(data)
        
        # 打印一些统计信息
        print(f"\n数值统计:")
        print(f"最小值: {data.min().item()}")
        print(f"最大值: {data.max().item()}")
        print(f"平均值: {data.mean().item()}")
        print(f"标准差: {data.std().item()}")
    else:
        print(f"数据类型: {type(data)}")
        print(f"数值内容:")
        print(data)
else:
    print(f"找不到键名 {key}")

# 关闭输出文件
sys.stdout.close()
sys.stdout = sys.__stdout__