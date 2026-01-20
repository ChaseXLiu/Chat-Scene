import torch
import torch.nn.functional as F
import os
import sys

def load_file(path):
    print(f"正在加载 {path} ...")
    if not os.path.exists(path):
        print(f"错误: 文件 {path} 不存在。")
        return None
    try:
        data = torch.load(path, map_location='cpu')
        return data
    except Exception as e:
        print(f"加载文件出错: {e}")
        return None

def compare_pt_files(file1, file2):
    data1 = load_file(file1)
    data2 = load_file(file2)

    if data1 is None or data2 is None:
        return

    # --- 情况 1: 数据是字典格式 (例如 {'scene_00_0': tensor, ...}) ---
    if isinstance(data1, dict) and isinstance(data2, dict):
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())
        common_keys = sorted(list(keys1.intersection(keys2)))
        
        print(f"\n数据结构: 字典 (Dictionary)")
        print(f"文件 1 键数量: {len(keys1)}")
        print(f"文件 2 键数量: {len(keys2)}")
        print(f"共有 键数量: {len(common_keys)}")

        if len(common_keys) == 0:
            print("未找到共有的键！请检查两个文件的 Key 格式是否一致。")
            print(f"文件1 示例键: {list(keys1)[:3]}")
            print(f"文件2 示例键: {list(keys2)[:3]}")
            return

        print("正在根据共有键对齐特征...")
        feats1 = []
        feats2 = []
        
        for k in common_keys:
            v1 = data1[k]
            v2 = data2[k]
            
            # 转为 Tensor 并展平
            if not isinstance(v1, torch.Tensor): v1 = torch.tensor(v1)
            if not isinstance(v2, torch.Tensor): v2 = torch.tensor(v2)
            
            v1 = v1.flatten().float()
            v2 = v2.flatten().float()
            
            # 移除形状检查，改为在堆叠后统一处理
            # if v1.shape != v2.shape:
            #    print(f"警告: 键 {k} 的形状不匹配: {v1.shape} vs {v2.shape}, 跳过。")
            #    continue
                
            feats1.append(v1)
            feats2.append(v2)

        if not feats1:
            print("没有提取到有效的特征。")
            return

        feats1 = torch.stack(feats1)
        feats2 = torch.stack(feats2)

    # --- 情况 2: 数据是 Tensor 格式 (例如 [N, D]) ---
    elif isinstance(data1, torch.Tensor) and isinstance(data2, torch.Tensor):
        print(f"\n数据结构: Tensor")
        print(f"文件 1 形状: {data1.shape}")
        print(f"文件 2 形状: {data2.shape}")
        
        if data1.shape != data2.shape:
            print("形状不匹配！无法进行逐元素比较。")
            return
            
        feats1 = data1.float()
        feats2 = data2.float()
        if feats1.dim() > 2:
            feats1 = feats1.flatten(start_dim=1)
            feats2 = feats2.flatten(start_dim=1)

    else:
        print(f"类型不兼容: {type(data1)} vs {type(data2)}")
        return

    # --- 维度对齐处理 ---
    # 如果维度不一致，添加一个简单的投影层进行对齐（模拟蒸馏过程中的投影）
    # 注意：这只是为了计算相似度，实际上未经训练的投影没有意义，
    # 但如果之前有投影层权重，应该在这里加载。
    # 这里我们采用 "截断" 或 "补零" 的方式来强行对齐，或者提示用户。
    
    # 检测维度
    dim1 = feats1.shape[1]
    dim2 = feats2.shape[1]
    
    if dim1 != dim2:
        print(f"\n警告: 特征维度不一致 ({dim1} vs {dim2})。")
        print("正在尝试对齐维度...")
        
        # 策略 1: 截断 (取前 N 维)
        min_dim = min(dim1, dim2)
        print(f"-> 采用截断策略: 仅比较前 {min_dim} 维特征。")
        feats1 = feats1[:, :min_dim]
        feats2 = feats2[:, :min_dim]
        
        # 策略 2: 如果您希望用线性层投影，请取消下面注释并加载权重
        # projector = torch.nn.Linear(dim1, dim2)
        # feats1 = projector(feats1) 

    # --- 计算相似度 ---
    print(f"\n正在计算 {feats1.shape[0]} 个样本的余弦相似度 (特征维度: {feats1.shape[1]})...")
    
    # 归一化向量
    feats1_norm = F.normalize(feats1, p=2, dim=1)
    feats2_norm = F.normalize(feats2, p=2, dim=1)

    # 计算点积 (即余弦相似度)
    similarities = (feats1_norm * feats2_norm).sum(dim=1)
    
    # 截断数值以防止数值误差超出 [-1, 1]
    similarities = torch.clamp(similarities, -1.0, 1.0)

    # --- 统计结果 ---
    mean_sim = similarities.mean().item()
    max_sim = similarities.max().item()
    min_sim = similarities.min().item()
    std_sim = similarities.std().item()

    print(f"\n结果统计:")
    print(f"{'-'*30}")
    print(f"平均相似度 (Mean): {mean_sim:.4f}")
    print(f"标准差 (Std Dev):  {std_sim:.4f}")
    print(f"最小值 (Min):      {min_sim:.4f}")
    print(f"最大值 (Max):      {max_sim:.4f}")
    print(f"{'-'*30}")

    # 分布直方图
    print("\n相似度分布:")
    hist = torch.histc(similarities, bins=10, min=-1, max=1)
    for i in range(10):
        low = -1 + i * 0.2
        high = -1 + (i + 1) * 0.2
        count = int(hist[i].item())
        # 简单的进度条可视化
        bar_len = int(count / len(similarities) * 50) if len(similarities) > 0 else 0
        bar = '#' * bar_len
        if count > 0:
            print(f"[{low:4.1f}, {high:4.1f}): {count:6d} | {bar}")

if __name__ == "__main__":
    # 这里填入您的文件路径
    f1 = "/home/lcx/chat-scene/Chat-Scene/annotations/scannet_mask3d_uni3d_feats.pt"
    f2 = "/home/lcx/chat-scene/Chat-Scene/annotations/scannet_mask3d_obj_textfeat.pt"
    
    compare_pt_files(f1, f2)