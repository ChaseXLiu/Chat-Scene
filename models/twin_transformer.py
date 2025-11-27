import torch
import torch.nn as nn
import os
from transformers import BertConfig
from .med import BertLayer 
import torch.utils.checkpoint as checkpoint # 引入 checkpoint 用于节省显存

class TwinTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers 
        self.gradient_checkpointing = False # 如果显存不够(OOM)，可以在外部设为 True
        
        self.layer_2d = nn.ModuleList([BertLayer(config, i) for i in range(self.num_layers)])
        self.layer_3d = nn.ModuleList([BertLayer(config, i) for i in range(self.num_layers)])
        
        # Feature adapters
        self.adapter_2d = nn.Linear(config.input_2d_dim, config.hidden_size)
        self.adapter_3d = nn.Linear(config.input_3d_dim, config.hidden_size)
        
        # Pre-LayerNorm 
        self.ln_2d = nn.LayerNorm(config.hidden_size)
        self.ln_3d = nn.LayerNorm(config.hidden_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.adapter_2d.weight)
        nn.init.zeros_(self.adapter_2d.bias)
        nn.init.xavier_uniform_(self.adapter_3d.weight)
        nn.init.zeros_(self.adapter_3d.bias)

    def forward(self, features_2d, features_3d, attention_mask_2d=None, attention_mask_3d=None):            
        # 1. Adapt & Norm
        hidden_states_2d = self.ln_2d(self.adapter_2d(features_2d)) 
        hidden_states_3d = self.ln_3d(self.adapter_3d(features_3d)) 
        
        # 2. Prepare masks
        if attention_mask_2d is not None:
            extended_mask_2d = self._prepare_attention_mask(attention_mask_2d, hidden_states_2d)
        else:
            extended_mask_2d = None
            
        if attention_mask_3d is not None:
            extended_mask_3d = self._prepare_attention_mask(attention_mask_3d, hidden_states_3d)
        else:
            extended_mask_3d = None
            
        # 3. Twin Loop
        for i in range(self.num_layers):
            layer_module_2d = self.layer_2d[i]
            layer_module_3d = self.layer_3d[i]
            
            # 【关键修改】保存当前层之前的状态，确保双流是对称交互的
            prev_hidden_states_2d = hidden_states_2d
            prev_hidden_states_3d = hidden_states_3d

            def run_layer(module, hidden_states, attention_mask, encoder_hidden, encoder_mask):
                return module(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=encoder_mask,
                    mode='multimodal'
                )[0]

            # --- 2D Stream ---
            # 使用 gradient_checkpointing 节省 RTX 3090 显存
            if self.training and self.gradient_checkpointing:
                new_hidden_states_2d = checkpoint.checkpoint(
                    run_layer, layer_module_2d, prev_hidden_states_2d, extended_mask_2d, prev_hidden_states_3d, extended_mask_3d
                )
            else:
                new_hidden_states_2d = run_layer(
                    layer_module_2d, prev_hidden_states_2d, extended_mask_2d, prev_hidden_states_3d, extended_mask_3d
                )

            # --- 3D Stream ---
            # 注意：CrossAttention 的 Key/Value 输入必须是 prev_hidden_states_2d (未经本层更新的)
            if self.training and self.gradient_checkpointing:
                new_hidden_states_3d = checkpoint.checkpoint(
                    run_layer, layer_module_3d, prev_hidden_states_3d, extended_mask_3d, prev_hidden_states_2d, extended_mask_2d
                )
            else:
                new_hidden_states_3d = run_layer(
                    layer_module_3d, prev_hidden_states_3d, extended_mask_3d, prev_hidden_states_2d, extended_mask_2d
                )
            
            # 更新状态
            hidden_states_2d = new_hidden_states_2d
            hidden_states_3d = new_hidden_states_3d
            
        return hidden_states_2d, hidden_states_3d

    def _prepare_attention_mask(self, attention_mask, input_tensor):
        if attention_mask.dim() == 4:
            return attention_mask 
        # 跟随输入 tensor 的类型 (fp16/fp32) 和设备
        extended_attention_mask = attention_mask.to(dtype=input_tensor.dtype, device=input_tensor.device)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask.unsqueeze(1).unsqueeze(2)


class TwinTransformer(nn.Module):
    def __init__(self, 
                 input_2d_dim=1024,    
                 input_3d_dim=1024,    
                 hidden_size=768,       
                 num_hidden_layers=2,
                 num_attention_heads=12,
                 hidden_dropout_prob=0.1,
                 bert_weights_path=None):              
        super().__init__()
        
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=hidden_dropout_prob,
        )
        
        config.input_2d_dim = input_2d_dim
        config.input_3d_dim = input_3d_dim
        
        # 必须开启 cross attention
        config.add_cross_attention = True
        # 必须设置 encoder_width 等于 hidden_size，因为用了 Adapter
        config.encoder_width = hidden_size
        config.chunk_size_feed_forward = 0
        config.output_attentions = False
        
        self.twin_encoder = TwinTransformerEncoder(config)

        self.load_bert_weights(bert_weights_path)

    def load_bert_weights(self, weight_path):
        if not os.path.exists(weight_path):
            print(f"Warning: BERT weight file not found at {weight_path}. Using random init.")
            return

        print(f"Loading BERT weights from {weight_path}...")
        # map_location='cpu' 保证在任何环境下都能读
        bert_state_dict = torch.load(weight_path, map_location='cpu')
        
        num_layers = len(self.twin_encoder.layer_2d)
        
        for i in range(num_layers):
            prefix = f"bert.encoder.layer.{i}."
            layer_state_dict = {}
            for key, value in bert_state_dict.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    layer_state_dict[new_key] = value
            
            # strict=False 忽略不匹配的键 (如 LayerNorms)
            self.twin_encoder.layer_2d[i].load_state_dict(layer_state_dict, strict=False)
            self.twin_encoder.layer_3d[i].load_state_dict(layer_state_dict, strict=False)
            
        print(f"Successfully initialized first {num_layers} layers from BERT.")
        
    def forward(self, features_2d, features_3d, attention_mask_2d=None, attention_mask_3d=None):
        return self.twin_encoder(features_2d, features_3d, attention_mask_2d, attention_mask_3d)