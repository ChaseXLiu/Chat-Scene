"""
Custom Twin-Transformer Module for processing text, 3D and 2D features
- Text features: from text encoder (e.g., BERT)
- 3D features: Uni3D (1024-dim, geometric-semantic features)
- 2D features: DINOv2 (1024-dim, fine-grained visual features)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from .med import BertLayer  # Assuming you have the custom BertLayer from BridgeQA


class TwinTransformerEncoder(nn.Module):
    """Custom Twin Transformer Encoder that processes text, 2D and 3D features in parallel streams"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_layers_twin = getattr(config, 'num_hidden_layers_twin', config.num_hidden_layers)
        
        # Text stream layers
        self.layer_text = nn.ModuleList([BertLayer(config, i) for i in range(self.num_hidden_layers)])
        
        # 2D stream layers
        self.layer_2d = nn.ModuleList([BertLayer(config, i) for i in range(self.num_hidden_layers)])
        
        # 3D stream layers (twin layers)
        self.layer_3d = nn.ModuleList([BertLayer(config, i) for i in range(self.num_hidden_layers_twin)])
        
        # Feature adapters to convert input features to hidden size
        self.adapter_text = nn.Linear(config.input_text_dim, config.hidden_size)
        self.adapter_2d = nn.Linear(config.input_2d_dim, config.hidden_size)
        self.adapter_3d = nn.Linear(config.input_3d_dim, config.hidden_size)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
    def forward(self, features_text, features_2d, features_3d, 
                attention_mask_text=None, attention_mask_2d=None, attention_mask_3d=None):
        """
        Forward pass for Custom Twin Transformer
        
        Args:
            features_text: [batch_size, seq_len_text, text_dim] - Text features
            features_2d: [batch_size, seq_len_2d, 1024] - DINOv2 features
            features_3d: [batch_size, seq_len_3d, 1024] - Uni3D features
            attention_mask_text: [batch_size, seq_len_text] - Attention mask for text features
            attention_mask_2d: [batch_size, seq_len_2d] - Attention mask for 2D features
            attention_mask_3d: [batch_size, seq_len_3d] - Attention mask for 3D features
            
        Returns:
            fused_features: [batch_size, hidden_size] - Fused features
            features_text_out: [batch_size, seq_len_text, hidden_size] - Processed text features
            features_2d_out: [batch_size, seq_len_2d, hidden_size] - Processed 2D features
            features_3d_out: [batch_size, seq_len_3d, hidden_size] - Processed 3D features
        """
        # Adapt input features to hidden size
        hidden_states_text = self.adapter_text(features_text)  # [B, L_text, H]
        hidden_states_2d = self.adapter_2d(features_2d)  # [B, L_2D, H]
        hidden_states_3d = self.adapter_3d(features_3d)  # [B, L_3D, H]
        
        # Prepare attention masks
        if attention_mask_text is not None:
            attention_mask_text = self._prepare_attention_mask(attention_mask_text, hidden_states_text.device)
            
        if attention_mask_2d is not None:
            attention_mask_2d = self._prepare_attention_mask(attention_mask_2d, hidden_states_text.device)
            
        if attention_mask_3d is not None:
            attention_mask_3d = self._prepare_attention_mask(attention_mask_3d, hidden_states_text.device)
            
        # Process through twin transformer layers
        for i in range(min(self.num_hidden_layers, self.num_hidden_layers_twin)):
            # Get layer modules
            layer_module_text = self.layer_text[i]
            layer_module_2d = self.layer_2d[i]
            layer_module_3d = self.layer_3d[i]
            
            # 准备交叉注意力的 K 和 V
            encoder_hidden_states_text = torch.cat([hidden_states_text, hidden_states_2d, hidden_states_3d], dim=1)
            encoder_hidden_states_2d = torch.cat([hidden_states_2d, hidden_states_3d, hidden_states_text], dim=1)
            encoder_hidden_states_3d = torch.cat([hidden_states_3d, hidden_states_2d, hidden_states_text], dim=1)
            
            # 准备交叉注意力的 mask
            encoder_attention_mask_text = self._concat_masks([attention_mask_text, attention_mask_2d, attention_mask_3d])
            encoder_attention_mask_2d = self._concat_masks([attention_mask_2d, attention_mask_3d, attention_mask_text])
            encoder_attention_mask_3d = self._concat_masks([attention_mask_3d, attention_mask_2d, attention_mask_text])
            
            # --- 文本流 ---
            layer_outputs_text = layer_module_text(
                hidden_states_text,
                attention_mask=attention_mask_text,
                encoder_hidden_states=encoder_hidden_states_text,
                encoder_attention_mask=encoder_attention_mask_text,
                mode='multimodal'
            )
            hidden_states_text = layer_outputs_text[0]
            
            # --- 2D 流 ---
            layer_outputs_2d = layer_module_2d(
                hidden_states_2d,
                attention_mask=attention_mask_2d,
                encoder_hidden_states=encoder_hidden_states_2d,
                encoder_attention_mask=encoder_attention_mask_2d,
                mode='multimodal'
            )
            hidden_states_2d = layer_outputs_2d[0]
            
            # --- 3D 流 ---
            layer_outputs_3d = layer_module_3d(
                hidden_states_3d,
                attention_mask=attention_mask_3d,
                encoder_hidden_states=encoder_hidden_states_3d,
                encoder_attention_mask=encoder_attention_mask_3d,
                mode='multimodal'
            )
            hidden_states_3d = layer_outputs_3d[0]
            
        # Fuse the features
        # Option 1: Concatenate and project
        # fused_features = self.output_projection(
        #     torch.cat([hidden_states_2d.mean(dim=1), hidden_states_3d.mean(dim=1)], dim=-1)
        # )  # [B, H]
        fused_features = None # 设为 None
        
        return fused_features, hidden_states_text, hidden_states_2d, hidden_states_3d
    
    def _prepare_attention_mask(self, attention_mask, device):
        """将 [B, L] 掩码转换为 [B, 1, 1, L] 并且反转 (0 -> 0.0, 1 -> -10000.0)"""
        if attention_mask.dim() == 4:
            return attention_mask # 已经处理过了
        # 确保 mask 是 float 类型
        extended_attention_mask = attention_mask.to(dtype=torch.float32)
        # 反转 mask: 1.0 (attend) -> 0.0, 0.0 (ignore) -> -10000.0
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # 扩展维度
        return extended_attention_mask.unsqueeze(1).unsqueeze(2).to(device)

    def _concat_masks(self, mask_list):
        """安全地拼接多个掩码，即使某些为 None"""
        valid_masks = [m for m in mask_list if m is not None]
        if not valid_masks:
            return None
        # 假设所有 mask 都是 [B, 1, 1, L]
        return torch.cat(valid_masks, dim=-1)


class TwinTransformer(nn.Module):
    """Complete Custom Twin Transformer module for text-2D-3D feature fusion"""
    
    def __init__(self, 
                 input_text_dim=768,   # Text feature dimension
                 input_2d_dim=1024,    # DINOv2 feature dimension
                 input_3d_dim=1024,    # Uni3D feature dimension
                 hidden_size=768,      # Hidden size for transformer
                 num_hidden_layers=6,  # Number of layers for text/2D stream
                 num_hidden_layers_twin=6,  # Number of layers for 3D stream
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1):
        super().__init__()
        
        # Create config for transformer
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        
        # 添加自定义属性
        config.input_text_dim = input_text_dim
        config.input_2d_dim = input_2d_dim
        config.input_3d_dim = input_3d_dim
        config.num_hidden_layers_twin = num_hidden_layers_twin
        config.encoder_width = hidden_size # 用于 BertLayer 的交叉注意力
        config.add_cross_attention = True   # 确保 BertLayer 启用交叉注意力
        
        self.config = config
        self.twin_encoder = TwinTransformerEncoder(config)
        
    def forward(self, features_text, features_2d, features_3d, 
                attention_mask_text=None, attention_mask_2d=None, attention_mask_3d=None):
        """
        Forward pass for Custom Twin Transformer
        
        Args:
            features_text: [batch_size, seq_len_text, text_dim] - Text features
            features_2d: [batch_size, seq_len_2d, 1024] - DINOv2 features
            features_3d: [batch_size, seq_len_3d, 1024] - Uni3D features
            attention_mask_text: [batch_size, seq_len_text] - Attention mask for text features
            attention_mask_2d: [batch_size, seq_len_2d] - Attention mask for 2D features
            attention_mask_3d: [batch_size, seq_len_3d] - Attention mask for 3D features
            
        Returns:
            fused_features: [batch_size, hidden_size] - Fused features
            features_text_out: [batch_size, seq_len_text, hidden_size] - Processed text features
            features_2d_out: [batch_size, seq_len_2d, hidden_size] - Processed 2D features
            features_3d_out: [batch_size, seq_len_3d, hidden_size] - Processed 3D features
        """
        return self.twin_encoder(
            features_text, features_2d, features_3d, 
            attention_mask_text, attention_mask_2d, attention_mask_3d
        )


# # Example usage
# if __name__ == "__main__":
#     # Create model
#     model = CustomTwinTransformer(
#         input_text_dim=768,
#         input_2d_dim=1024,
#         input_3d_dim=1024,
#         hidden_size=768,
#         num_hidden_layers=6,
#         num_hidden_layers_twin=6
#     )
    
#     # Create sample inputs
#     batch_size = 2
#     seq_len_text = 20   # Text tokens
#     seq_len_2d = 256    # DINOv2 patch tokens
#     seq_len_3d = 128    # Uni3D point tokens
    
#     features_text = torch.randn(batch_size, seq_len_text, 768)
#     features_2d = torch.randn(batch_size, seq_len_2d, 1024)
#     features_3d = torch.randn(batch_size, seq_len_3d, 1024)
    
#     # Forward pass
#     fused_features, features_text_out, features_2d_out, features_3d_out = model(
#         features_text, features_2d, features_3d
#     )
    
    # print(f"Fused features shape: {fused_features.shape}")      # [batch_size, hidden_size]
    # print(f"Text features shape: {features_text_out.shape}")    # [batch_size, seq_len_text, hidden_size]
    # print(f"2D features shape: {features_2d_out.shape}")        # [batch_size, seq_len_2d, hidden_size]
    # print(f"3D features shape: {features_3d_out.shape}")        # [batch_size, seq_len_3d, hidden_size]