#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_feats_dinov2_vs_dinov3.py
-----------------------------------
可视化并比较 DINOv2 与 DINOv3 的 Mask3D 特征分布

Usage:
    python visualize_feats_dinov2_vs_dinov3.py \
        --file_dinov2 annotations/scannet_mask3d_videofeats_dinov2.pt \
        --file_dinov3 annotations/scannet_mask3d_videofeats_dinov3.pt \
        --out_dir results/feat_vis/
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import seaborn as sns


# ---------------------------------------------------------------
# 加载特征
# ---------------------------------------------------------------
def load_feats(path, max_scenes=None):
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        all_feats = []
        for i, (k, v) in enumerate(data.items()):
            if max_scenes is not None and i >= max_scenes:
                break
            if not isinstance(v, torch.Tensor):
                continue
            all_feats.append(v)

        first_shape = all_feats[0].shape
        print(f"[DEBUG] First tensor shape: {first_shape}")

        if len(first_shape) == 1:
            feats = torch.stack(all_feats, dim=0)
        elif len(first_shape) == 2:
            feats = torch.cat(all_feats, dim=0)
        else:
            raise ValueError(f"Unexpected feature shape: {first_shape}")

        print(f"[INFO] Loaded {len(all_feats)} scenes, total {feats.shape[0]} features of dim {feats.shape[-1]}")
        return feats
    else:
        print(f"[INFO] Loaded tensor features of shape {data.shape}")
        return data


# ---------------------------------------------------------------
# 基础统计信息
# ---------------------------------------------------------------
def compute_stats(f1, f2, name1="DINOv2", name2="DINOv3"):
    f1, f2 = f1.numpy() if isinstance(f1, torch.Tensor) else f1, f2.numpy() if isinstance(f2, torch.Tensor) else f2
    print("\n=== Feature Statistics ===")
    print(f"{name1} mean: {f1.mean():.4f}, std: {f1.std():.4f}")
    print(f"{name2} mean: {f2.mean():.4f}, std: {f2.std():.4f}")

    sample_n = min(len(f1), len(f2), 2000)
    cos_sim = 1 - cdist(f1[:sample_n], f2[:sample_n], metric="cosine")
    print(f"Mean cosine similarity ({name1} vs {name2}): {cos_sim.mean():.4f}")

    l2_dist = np.linalg.norm(f1[:sample_n] - f2[:sample_n], axis=1).mean()
    print(f"Mean L2 distance: {l2_dist:.4f}")


# ---------------------------------------------------------------
# 可视化：t-SNE
# ---------------------------------------------------------------
def visualize_tsne(f1, f2, out_dir, name1="DINOv2", name2="DINOv3"):
    f1, f2 = f1.numpy() if isinstance(f1, torch.Tensor) else f1, f2.numpy() if isinstance(f2, torch.Tensor) else f2
    n = min(len(f1), len(f2), 3000)
    idx1 = np.random.choice(len(f1), n, replace=False)
    idx2 = np.random.choice(len(f2), n, replace=False)
    samples = np.concatenate([f1[idx1], f2[idx2]], axis=0)
    labels = np.array([0] * n + [1] * n)

    print("[INFO] Running PCA (pre-reduction for t-SNE)...")
    pca = PCA(n_components=min(50, samples.shape[1]))
    samples_50 = pca.fit_transform(samples)

    print("[INFO] Running t-SNE (2D)...")
    tsne = TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=1000, random_state=42)
    emb = tsne.fit_transform(samples_50)

    plt.figure(figsize=(8, 6))
    plt.scatter(emb[labels == 0, 0], emb[labels == 0, 1], s=5, alpha=0.5, label=name1, color='royalblue')
    plt.scatter(emb[labels == 1, 0], emb[labels == 1, 1], s=5, alpha=0.5, label=name2, color='orange')
    plt.legend()
    plt.title(f"t-SNE feature distribution: {name1} vs {name2}")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "tsne_dinov2_vs_dinov3.png"), dpi=300)
    plt.close()
    print(f"[INFO] Saved t-SNE figure to {out_dir}/tsne_dinov2_vs_dinov3.png")


# ---------------------------------------------------------------
# 可视化：PCA 2D + 3D
# ---------------------------------------------------------------
def visualize_pca(f1, f2, out_dir, name1="DINOv2", name2="DINOv3"):
    f1, f2 = f1.numpy() if isinstance(f1, torch.Tensor) else f1, f2.numpy() if isinstance(f2, torch.Tensor) else f2
    n = min(len(f1), len(f2), 5000)
    X = np.concatenate([f1[:n], f2[:n]], axis=0)
    labels = np.array([0] * n + [1] * n)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # 2D PCA
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[labels == 0, 0], X_pca[labels == 0, 1], s=5, alpha=0.5, label=name1, color='royalblue')
    plt.scatter(X_pca[labels == 1, 0], X_pca[labels == 1, 1], s=5, alpha=0.5, label=name2, color='orange')
    plt.title(f"2D PCA Feature Distribution: {name1} vs {name2}")
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "pca2d_dinov2_vs_dinov3.png"), dpi=300)
    plt.close()

    # 3D PCA
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_pca[labels == 0, 0], X_pca[labels == 0, 1], X_pca[labels == 0, 2],
               s=5, alpha=0.6, label=name1, color='royalblue')
    ax.scatter(X_pca[labels == 1, 0], X_pca[labels == 1, 1], X_pca[labels == 1, 2],
               s=5, alpha=0.6, label=name2, color='orange')
    ax.set_title(f"3D PCA Feature Distribution: {name1} vs {name2}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca3d_dinov2_vs_dinov3.png"), dpi=300)
    plt.close()
    print(f"[INFO] Saved PCA 2D/3D figures to {out_dir}")


# ---------------------------------------------------------------
# 可视化：密度分布热力图
# ---------------------------------------------------------------
def visualize_density(f1, f2, out_dir, name1="DINOv2", name2="DINOv3"):
    f1, f2 = f1.numpy() if isinstance(f1, torch.Tensor) else f1, f2.numpy() if isinstance(f2, torch.Tensor) else f2
    n = min(len(f1), len(f2), 3000)
    X = np.concatenate([f1[:n], f2[:n]], axis=0)
    labels = np.array([0] * n + [1] * n)
    X_pca = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=X_pca[labels == 0, 0], y=X_pca[labels == 0, 1],
                fill=True, cmap="Blues", alpha=0.4, label=name1)
    sns.kdeplot(x=X_pca[labels == 1, 0], y=X_pca[labels == 1, 1],
                fill=True, cmap="Oranges", alpha=0.4, label=name2)
    plt.title(f"Density Heatmap (PCA 2D): {name1} vs {name2}")
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "density_heatmap_dinov2_vs_dinov3.png"), dpi=300)
    plt.close()
    print(f"[INFO] Saved density heatmap to {out_dir}/density_heatmap_dinov2_vs_dinov3.png")


# ---------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------
def main(args):
    f2 = load_feats(args.file_dinov2, args.max_scenes)
    f3 = load_feats(args.file_dinov3, args.max_scenes)

    compute_stats(f2, f3, "DINOv2", "DINOv3")
    visualize_pca(f2, f3, args.out_dir, "DINOv2", "DINOv3")
    visualize_tsne(f2, f3, args.out_dir, "DINOv2", "DINOv3")
    visualize_density(f2, f3, args.out_dir, "DINOv2", "DINOv3")

    print(f"\n[INFO] All visualizations saved to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dinov2", required=True, help="Path to DINOv2 feature file")
    parser.add_argument("--file_dinov3", required=True, help="Path to DINOv3 feature file")
    parser.add_argument("--max_scenes", type=int, default=100, help="Max number of scenes to load")
    parser.add_argument("--out_dir", type=str, default="results/feat_vis", help="Output directory")
    args = parser.parse_args()
    main(args)
