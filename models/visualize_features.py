def _visualize_features(multi_scale_feats, orig_feat, img_feat, locs, sample_idx=None):
        """
        可视化多尺度特征分布
        
        Args:
            multi_scale_feats: 多尺度特征列表
            orig_feat: 原始3D特征
            img_feat: 图像特征
            locs: 物体位置信息
            sample_idx: 要可视化的样本索引，None表示使用第一个样本
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            
            # 尝试导入sklearn
            try:
                from sklearn.manifold import TSNE
                from sklearn.decomposition import PCA
            except ImportError:
                print("警告: sklearn未安装，无法进行特征可视化")
                return
            
            # 创建保存目录
            save_dir = '/home/lcx/chat-scene/Chat-Scene/vis/feature_dist'
            os.makedirs(save_dir, exist_ok=True)
            
            # 确定要可视化的样本索引
            batch_idx = 0 if sample_idx is None else sample_idx
            if batch_idx >= multi_scale_feats[0].shape[0]:
                batch_idx = 0
                
            # 提取要可视化的特征
            features_to_vis = {}
            
            # 检查维度并进行投影，确保所有特征具有相同的维度
            orig_dim = orig_feat.shape[-1]
            llama_dim = multi_scale_feats[0].shape[-1]
            
            # 创建临时投影层，用于将原始特征投影到相同维度
            if orig_dim != llama_dim:
                temp_proj = nn.Linear(orig_dim, llama_dim).to(orig_feat.device)
                features_to_vis['original'] = temp_proj(orig_feat[batch_idx]).detach().cpu().numpy()
                
                # 如果图像特征维度也不同，同样进行投影
                if img_feat.shape[-1] != llama_dim:
                    img_temp_proj = nn.Linear(img_feat.shape[-1], llama_dim).to(img_feat.device)
                    features_to_vis['image'] = img_temp_proj(img_feat[batch_idx]).detach().cpu().numpy()
                else:
                    features_to_vis['image'] = img_feat[batch_idx].detach().cpu().numpy()
            else:
                features_to_vis['original'] = orig_feat[batch_idx].detach().cpu().numpy()
                features_to_vis['image'] = img_feat[batch_idx].detach().cpu().numpy()
            
            # 添加多尺度特征
            features_to_vis['semantic'] = multi_scale_feats[0][batch_idx].detach().cpu().numpy()
            features_to_vis['geometry'] = multi_scale_feats[1][batch_idx].detach().cpu().numpy()
            features_to_vis['texture'] = multi_scale_feats[2][batch_idx].detach().cpu().numpy()
            
            # 获取当前时间戳作为文件名
            import time
            timestamp = int(time.time())
            
            # 使用t-SNE进行降维可视化
            plt.figure(figsize=(20, 10))
            
            # 1. t-SNE可视化
            plt.subplot(1, 2, 1)
            combined_features = np.vstack([feat for name, feat in features_to_vis.items()])
            combined_labels = np.concatenate([np.full(feat.shape[0], i) for i, (name, feat) in enumerate(features_to_vis.items())])
            
            # 如果特征维度太高，先用PCA降维
            if combined_features.shape[1] > 50:
                pca = PCA(n_components=50)
                combined_features = pca.fit_transform(combined_features)
            
            # 应用t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, combined_features.shape[0]//5))
            tsne_results = tsne.fit_transform(combined_features)
            
            # 绘制t-SNE结果
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            markers = ['o', 's', '^', 'D', 'x']
            feature_names = list(features_to_vis.keys())
            
            start_idx = 0
            for i, (name, feat) in enumerate(features_to_vis.items()):
                end_idx = start_idx + feat.shape[0]
                plt.scatter(
                    tsne_results[start_idx:end_idx, 0],
                    tsne_results[start_idx:end_idx, 1],
                    c=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    alpha=0.7,
                    label=name
                )
                start_idx = end_idx
            
            plt.title('t-SNE Visualization of Multi-scale Features')
            plt.legend()
            
            # 2. PCA可视化
            plt.subplot(1, 2, 2)
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(combined_features)
            
            start_idx = 0
            for i, (name, feat) in enumerate(features_to_vis.items()):
                end_idx = start_idx + feat.shape[0]
                plt.scatter(
                    pca_results[start_idx:end_idx, 0],
                    pca_results[start_idx:end_idx, 1],
                    c=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    alpha=0.7,
                    label=name
                )
                start_idx = end_idx
            
            plt.title('PCA Visualization of Multi-scale Features')
            plt.legend()
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(f'{save_dir}/feature_dist_{timestamp}.png')
            plt.close()
            
            # 3. 特征统计信息可视化
            plt.figure(figsize=(15, 10))
            
            # 3.1 特征范数分布
            plt.subplot(2, 2, 1)
            for i, (name, feat) in enumerate(features_to_vis.items()):
                norms = np.linalg.norm(feat, axis=1)
                plt.hist(norms, alpha=0.5, bins=20, label=name)
            plt.title('Feature Norm Distribution')
            plt.xlabel('L2 Norm')
            plt.ylabel('Count')
            plt.legend()
            
            # 3.2 特征余弦相似度热图
            plt.subplot(2, 2, 2)
            cosine_sim = np.zeros((len(feature_names), len(feature_names)))
            for i, (name1, feat1) in enumerate(features_to_vis.items()):
                for j, (name2, feat2) in enumerate(features_to_vis.items()):
                    # 计算平均特征向量
                    mean_feat1 = np.mean(feat1, axis=0)
                    mean_feat2 = np.mean(feat2, axis=0)
                    # 计算余弦相似度
                    sim = np.dot(mean_feat1, mean_feat2) / (np.linalg.norm(mean_feat1) * np.linalg.norm(mean_feat2) + 1e-8)
                    cosine_sim[i, j] = sim
            
            plt.imshow(cosine_sim, cmap='viridis')
            plt.colorbar()
            plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45)
            plt.yticks(np.arange(len(feature_names)), feature_names)
            plt.title('Cosine Similarity Between Feature Types')
            
            # 3.3 特征维度激活分布
            plt.subplot(2, 2, 3)
            for i, (name, feat) in enumerate(features_to_vis.items()):
                # 计算每个维度的平均激活值
                mean_activation = np.mean(feat, axis=0)
                plt.plot(mean_activation[:100], label=f'{name} (first 100 dims)', alpha=0.7)
            plt.title('Mean Activation of First 100 Feature Dimensions')
            plt.xlabel('Dimension')
            plt.ylabel('Mean Activation')
            plt.legend()
            
            # 3.4 特征方差分布
            plt.subplot(2, 2, 4)
            for i, (name, feat) in enumerate(features_to_vis.items()):
                # 计算每个维度的方差
                var = np.var(feat, axis=0)
                plt.plot(np.sort(var)[::-1][:50], label=f'{name} (top 50 dims)', alpha=0.7)
            plt.title('Top 50 Dimensions with Highest Variance')
            plt.xlabel('Dimension Rank')
            plt.ylabel('Variance')
            plt.legend()
            
            # 保存统计图
            plt.tight_layout()
            plt.savefig(f'{save_dir}/feature_stats_{timestamp}.png')
            plt.close()
            
            # 如果有位置信息，可视化3D空间中的特征分布
            if locs is not None:
                self._visualize_3d_features(multi_scale_feats, locs, batch_idx, save_dir, timestamp)
            
            print(f"特征可视化已保存到 {save_dir}/feature_dist_{timestamp}.png 和 feature_stats_{timestamp}.png")
        except Exception as e:
            print(f"特征可视化失败: {e}")
            # 出错时不影响模型正常运行
            pass

def _visualize_3d_features(multi_scale_feats, locs, batch_idx, save_dir, timestamp):
    """可视化3D空间中的特征分布"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        # 提取位置信息和特征
        positions = locs[batch_idx, :, :3].detach().cpu().numpy()
        
        # 创建3D图
        fig = plt.figure(figsize=(20, 15))
        
        # 为每个尺度创建一个子图
        for i in range(len(multi_scale_feats)):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            
            # 获取当前尺度的特征
            features = multi_scale_feats[i][batch_idx].detach().cpu().numpy()
            
            # 计算特征的主成分或某种统计量作为颜色
            if features.shape[1] > 3:
                try:
                    # 使用PCA降到3维
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=3)
                    feature_colors = pca.fit_transform(features)
                    # 归一化到[0,1]范围
                    feature_colors = (feature_colors - feature_colors.min(axis=0)) / (feature_colors.max(axis=0) - feature_colors.min(axis=0) + 1e-10)
                except ImportError:
                    # 如果没有sklearn，使用特征的前3个维度
                    feature_colors = features[:, :3]
                    if feature_colors.shape[1] < 3:
                        # 如果维度不足3，填充
                        padding = np.zeros((feature_colors.shape[0], 3 - feature_colors.shape[1]))
                        feature_colors = np.hstack([feature_colors, padding])
                    # 归一化
                    feature_colors = (feature_colors - feature_colors.min(axis=0)) / (feature_colors.max(axis=0) - feature_colors.min(axis=0) + 1e-10)
            else:
                feature_colors = features
            
            # 绘制3D散点图，颜色表示特征值
            scatter = ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=np.linalg.norm(features, axis=1),  # 使用特征范数作为颜色
                cmap='viridis',
                s=50,
                alpha=0.7
            )
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax, label='Feature Norm')
            
            # 设置标题和标签
            ax.set_title(f'{self.token_groups[i]} Features in 3D Space')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        # 添加总体标题
        plt.suptitle('Multi-scale Features Distribution in 3D Space')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(f'{save_dir}/3d_feature_dist_{timestamp}.png')
        plt.close()
    except Exception as e:
        print(f"3D特征可视化失败: {e}")