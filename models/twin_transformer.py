"""
Custom Twin-Transformer Module (BridgeQA Style)
- 2D Stream: Processes 2D features (Conditioned on 2D + 3D context)
- 3D Stream: Processes 3D features (Conditioned on 3D + 2D context)
- Consistent with BridgeQA paper: No independent Text Stream in the Twin module.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from .med import BertLayer  # Assuming you have the custom BertLayer from BridgeQA


class TwinTransformerEncoder(nn.Module):
    """
    Custom Twin Transformer Encoder that processes 2D and 3D features in parallel streams.
    Implements the 'Twin' fusion: each stream attends to the concatenation of both streams.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_layers_twin = getattr(config, 'num_hidden_layers_twin', config.num_hidden_layers)
        
        # 2D stream layers
        self.layer_2d = nn.ModuleList([BertLayer(config, i) for i in range(self.num_hidden_layers)])
        
        # 3D stream layers (twin layers)
        self.layer_3d = nn.ModuleList([BertLayer(config, i) for i in range(self.num_hidden_layers_twin)])
        
        # Feature adapters to convert input features to hidden size
        self.adapter_text = nn.Linear(config.input_text_dim, config.hidden_size)
        self.adapter_2d = nn.Linear(config.input_2d_dim, config.hidden_size)
        self.adapter_3d = nn.Linear(config.input_3d_dim, config.hidden_size)
        
    def forward(self, features_text, features_2d, features_3d, attention_mask_text=None, attention_mask_2d=None, attention_mask_3d=None):
        """
        Forward pass for Custom Twin Transformer (2D + 3D)
        
        Args:
            features_text: [batch_size, seq_len_text, text_dim]
            features_2d: [batch_size, seq_len_2d, 1024]
            features_3d: [batch_size, seq_len_3d, 1024]
            attention_mask_2d: [batch_size, seq_len_2d]
            attention_mask_3d: [batch_size, seq_len_3d]
            features_text: Optional, kept for compatibility but not used in Twin loop by default
            
        Returns:
            features_2d_out: [batch_size, seq_len_2d, hidden_size]
            features_3d_out: [batch_size, seq_len_3d, hidden_size]
        """
        # Adapt input features to hidden size
        hidden_states_text = self.adapter_text(features_text) # [B, L_text, H]
        hidden_states_2d = self.adapter_2d(features_2d)       # [B, L_2d, H]
        hidden_states_3d = self.adapter_3d(features_3d)       # [B, L_3d, H]
        
        # Prepare attention masks
        if attention_mask_text is not None:
            attention_mask_text = self._prepare_attention_mask(attention_mask_text, hidden_states_text.device)

        if attention_mask_2d is not None:
            attention_mask_2d = self._prepare_attention_mask(attention_mask_2d, hidden_states_2d.device)
            
        if attention_mask_3d is not None:
            attention_mask_3d = self._prepare_attention_mask(attention_mask_3d, hidden_states_2d.device)
            
        # Process through twin transformer layers
        for i in range(min(self.num_hidden_layers, self.num_hidden_layers_twin)):
            # Get layer modules
            layer_module_2d = self.layer_2d[i]
            layer_module_3d = self.layer_3d[i]
            
            # --- Twin-Transformer Fusion Mechanism---
            # 2D Stream sees: [3D, text]
            encoder_hidden_states_2d = torch.cat([hidden_states_3d, hidden_states_text], dim=1)
            encoder_attention_mask_2d = self._concat_masks([attention_mask_3d, attention_mask_text])
            
            # 3D Stream sees: [2D, text]
            encoder_hidden_states_3d = torch.cat([hidden_states_2d, hidden_states_text], dim=1)
            encoder_attention_mask_3d = self._concat_masks([attention_mask_2d, attention_mask_text])
            
            # --- Forward 2D ---
            layer_outputs_2d = layer_module_2d(
                hidden_states_2d,
                attention_mask=attention_mask_2d,
                encoder_hidden_states=encoder_hidden_states_2d, # K, V = text + 3D
                encoder_attention_mask=encoder_attention_mask_2d,
                mode='multimodal'
            )
            hidden_states_2d = layer_outputs_2d[0]
            
            # --- Forward 3D ---
            layer_outputs_3d = layer_module_3d(
                hidden_states_3d,
                attention_mask=attention_mask_3d,
                encoder_hidden_states=encoder_hidden_states_3d, # K, V = text + 2D
                encoder_attention_mask=encoder_attention_mask_3d,
                mode='multimodal'
            )
            hidden_states_3d = layer_outputs_3d[0]
            
        return hidden_states_2d, hidden_states_3d
    
    def _prepare_attention_mask(self, attention_mask, device):
        """将 [B, L] 掩码转换为 [B, 1, 1, L] 并且反转"""
        if attention_mask.dim() == 4:
            return attention_mask 
        extended_attention_mask = attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask.unsqueeze(1).unsqueeze(2).to(device)

    def _concat_masks(self, mask_list):
        """安全地拼接多个掩码"""
        valid_masks = [m for m in mask_list if m is not None]
        if not valid_masks:
            return None
        return torch.cat(valid_masks, dim=-1)


class TwinTransformer(nn.Module):
    """Complete Custom Twin Transformer module for 2D-3D feature fusion"""
    
    def __init__(self, 
                 input_text_dim=768,   # Keeping config args for compatibility
                 input_2d_dim=1024,    
                 input_3d_dim=1024,    
                 hidden_size=768,      
                 num_hidden_layers=6,  
                 num_hidden_layers_twin=6,  
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
        
        config.input_text_dim = input_text_dim
        config.input_2d_dim = input_2d_dim
        config.input_3d_dim = input_3d_dim
        config.num_hidden_layers_twin = num_hidden_layers_twin
        config.encoder_width = hidden_size
        config.add_cross_attention = True 
        
        self.config = config
        self.twin_encoder = TwinTransformerEncoder(config)
        
    def forward(self, features_text, features_2d, features_3d, 
                attention_mask_text=None, attention_mask_2d=None, attention_mask_3d=None):
        """
        Forward pass
        """
        return self.twin_encoder(
            features_text, features_2d, features_3d, 
            attention_mask_text, attention_mask_2d, attention_mask_3d
        )