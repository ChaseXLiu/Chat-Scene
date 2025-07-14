import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import os
from tqdm import tqdm

def initialize_codebook(features_dict, feature_key, codebook_size, codebook_dim, output_file):
    """
    Initialize codebook using K-Means clustering on 1024-dim features and save to .pt file.
    
    Args:
        features_dict (dict): Dictionary from .pt file, keys are sceneXXXX_YY_ZZ, values are dicts with feature_key.
        feature_key (str): Key for feature type ('global_feature', 'local_features', 'texture_features').
        codebook_size (int): Number of codebook vectors (e.g., 16, 32).
        codebook_dim (int): Dimension of each vector (1024).
        output_file (str): Path to save the initialized codebook.
    """
    # Extract features for the specified key
    features = []
    print(f"Extracting {feature_key} features...")
    for scene_id in tqdm(features_dict.keys(), desc=f"Processing {feature_key}", unit="object"):
        scene_data = features_dict[scene_id]
        feature = scene_data[feature_key]  # Shape: [1024]
        assert feature.shape == (codebook_dim,), \
            f"Feature dimension mismatch for {scene_id}, {feature_key}: expected {codebook_dim}, got {feature.shape[0]}"
        features.append(feature)
    features = torch.stack(features).numpy()  # Shape: [num_objects, 1024]
    print(f"Extracted {len(features)} features for {feature_key}")
    
    # Run K-Means
    print(f"Running K-Means for {feature_key} with {codebook_size} clusters...")
    kmeans = KMeans(n_clusters=codebook_size, random_state=0, n_init=10, verbose=1)
    kmeans.fit(features)
    codebook = torch.from_numpy(kmeans.cluster_centers_).float()  # Shape: [codebook_size, 1024]
    
    # L2 normalize
    codebook = F.normalize(codebook, dim=-1)
    
    # Save codebook
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(codebook, output_file)
    print(f"Codebook for {feature_key} saved to {output_file}")
    return codebook

def initialize_all_codebooks(features_file="/home/lcx/chat-scene/Chat-Scene/annotations/scannet_mask3d_uni3d_feats.pt"):
    """
    Initialize and save codebooks for all scales.
    
    Args:
        features_file (str): Path to .pt file containing feature dictionary.
    """
    codebook_dir = "/home/lcx/chat-scene/Chat-Scene/annotations/codebooks/"
    
    # Load feature dictionary
    features_dict = torch.load(features_file) 
    print(f"Loaded features from {features_file}, num_objects: {len(features_dict)}")
    
    # Initialize codebooks
    codebook_object = initialize_codebook(
        features_dict=features_dict,
        feature_key="global_feature",
        codebook_size=16,
        codebook_dim=1024,
        output_file=f"{codebook_dir}/codebook_object.pt"
    )
    codebook_mid = initialize_codebook(
        features_dict=features_dict,
        feature_key="local_features",
        codebook_size=32,
        codebook_dim=1024,
        output_file=f"{codebook_dir}/codebook_mid.pt"
    )
    codebook_detail = initialize_codebook(
        features_dict=features_dict,
        feature_key="texture_features",
        codebook_size=16,
        codebook_dim=1024,
        output_file=f"{codebook_dir}/codebook_detail.pt"
    )
    return codebook_object, codebook_mid, codebook_detail

if __name__ == "__main__":
    # Run initialization
    initialize_all_codebooks()