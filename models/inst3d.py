from pickletools import genops
import random
import logging
from abc import ABC
from typing import Optional

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
from models.position_embedding import PositionEmbeddingCoordsSine
from peft import LoraConfig, get_peft_model
# from models.load_llama import init_llama_model
from torch.nn.utils.rnn import pad_sequence

import contextlib
from dataset.base_dataset import update_caption, recover_caption


# Initialize the logger for the module
logger = logging.getLogger(__name__)

def nclamp(input, min_value, max_value):
    """Clamp the input tensor between min and max values while retaining the gradient."""
    clamped = input.clamp(min=min_value, max=max_value)
    return clamped.detach() + input - input.detach()

def print_grad_status(model):
    """
    After performing losses.backward(), this function checks and logs the status
    of each parameter in the model. It indicates whether the parameter is 
    trainable, and whether it has received gradients.
    """
    for name, param in model.named_parameters():
        status = (
            "(Trainable)" if param.requires_grad else "(Fixed)",
            "(Has grad)" if param.grad is not None else "(No grad backward)"
        )
        logger.info(f"{name:<80}{status[0]:<20}{status[1]:<20}{list(param.shape)}")

# MCMF_part1: vanilla self-attention with a learnable token for key and value
class SelfAttentionWithLearnableToken(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SelfAttentionWithLearnableToken, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim

        # Define a global token for learning
        self.global_key_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.global_value_token = nn.Parameter(torch.randn(1, 1, input_dim))

        # Define projection matrices for generating Q, K, V
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)

        # Define the linear layer to be used for output
        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Generate Q, K, V
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Add global token to key and value
        global_key_token = self.global_key_token.expand(batch_size, -1, -1)  # (batch_size, 1, input_dim)
        global_value_token = self.global_value_token.expand(batch_size, -1, -1)  # (batch_size, 1, input_dim)

        # Add a global token at the beginning of the sequence
        K = torch.cat([global_key_token, K], dim=1)
        V = torch.cat([global_value_token, V], dim=1)

        # Calculating the Self-Attention score
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.input_dim ** 0.5)

        # Calculating Attention Weights
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # weighted sum
        attention_output = torch.matmul(attention_weights, V)

        # Output via linear layer
        output = self.output_proj(attention_output)
        return output
    
# MCMF_part2: cross-modal feature fusion between 3D and 2D
class FeatureCrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, output_dim, num_heads):
        super(FeatureCrossAttention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.q_proj_1 = nn.Linear(query_dim, query_dim, bias=False) #1024->1024


        k_modules = [nn.Linear(768, 1024)]
        for _ in range(1,2):
            k_modules.append(nn.GELU())
            k_modules.append(nn.Linear(1024, 1024))
        self.k_proj_1 = nn.Sequential(*k_modules) #768->1024

        v_modules = [nn.Linear(768, 1024)]
        for _ in range(1,2):
            v_modules.append(nn.GELU())
            v_modules.append(nn.Linear(1024, 1024))
        self.v_proj_1 = nn.Sequential(*v_modules) #768->1024

        self.attention = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads)
        
        # self.fc = nn.Linear(query_dim + query_dim, output_dim)
        self.fc = nn.Linear(query_dim, output_dim)

    def forward(self, query, key, value):
        # query: (batch_size, seq_len, query_dim)
        # key, value: (batch_size, seq_len, key_dim)
        
        query = self.q_proj_1(query).transpose(0, 1)  # (seq_len, batch_size, query_dim)
        key = self.k_proj_1(key).transpose(0, 1)      # (seq_len, batch_size, key_dim)
        value = self.v_proj_1(value).transpose(0, 1)  # (seq_len, batch_size, key_dim)
        
        attn_output, _ = self.attention(query, key, value)
        
        attn_output = attn_output.transpose(0, 1)
        
        combined_output = attn_output + query.transpose(0, 1)
        
        final_output = self.fc(combined_output)
        
        return final_output
    
# 3D_ISR: spatial condition self-attention1
# 空间条件自注意力机制类，用于处理3D点云数据的空间关系
class MultiHeadAttentionSpatial(nn.Module):
    def __init__(
        self, d_model, n_head, dropout=0.1, spatial_multihead=True, spatial_dim=5,
        spatial_attn_fusion='mul',
    ):
        # 调用父类初始化方法
        super().__init__()
        # 确保模型维度能被头数整除
        assert d_model % n_head == 0, 'd_model: %d, n_head: %d' %(d_model, n_head)

        # 设置模型参数
        self.n_head = n_head  # 注意力头数
        self.d_model = d_model  # 模型维度
        self.d_per_head = d_model // n_head  # 每个头的维度
        self.spatial_multihead = spatial_multihead  # 是否使用多头空间注意力
        self.spatial_dim = spatial_dim  # 空间维度
        self.spatial_attn_fusion = spatial_attn_fusion  # 空间注意力融合方式

        # 定义线性变换层，用于查询、键、值的投影
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        
        # 输出线性层和正则化层
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)  # Dropout层
        self.layer_norm = nn.LayerNorm(d_model)  # LayerNorm层

        # 空间注意力头数设置
        self.spatial_n_head = n_head if spatial_multihead else 1
        # 根据不同的融合方式设置不同的线性层
        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            # 用于位置信息的线性变换
            self.pairwise_loc_fc = nn.Linear(spatial_dim, self.spatial_n_head)
        elif self.spatial_attn_fusion == 'ctx':
            # 上下文相关的线性变换
            self.pairwise_loc_fc = nn.Linear(spatial_dim, d_model)
        elif self.spatial_attn_fusion == 'cond':
            # 条件相关的线性变换
            self.lang_cond_fc = nn.Linear(d_model, self.spatial_n_head * (spatial_dim + 1))
        else:
            # 不支持的融合方式
            raise NotImplementedError('unsupported spatial_attn_fusion %s' % (self.spatial_attn_fusion))

    def forward(self, q, k, v, pairwise_locs, key_padding_mask=None, txt_embeds=None):
        # 保存残差连接
        residual = q
        # 对查询、键、值进行线性变换并重新排列维度
        q = genops.rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
        k = genops.rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
        v = genops.rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)
        # 计算注意力分数，使用爱因斯坦求和约定
        attn = torch.einsum('hblk,hbtk->hblt', q, k) / np.sqrt(q.shape[-1])

        # 根据不同的融合方式计算位置注意力
        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            # 通过线性层处理位置信息
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            # 重新排列维度以匹配注意力矩阵
            loc_attn = genops.rearrange(loc_attn, 'b l t h -> h b l t') 
            # 如果是乘法融合，应用ReLU激活函数
            if self.spatial_attn_fusion == 'mul':
                loc_attn = F.relu(loc_attn)
            # 如果不使用多头空间注意力，则复制到所有头
            if not self.spatial_multihead:
                loc_attn = genops.repeat(loc_attn, 'h b l t -> (h nh) b l t', nh=self.n_head)
        elif self.spatial_attn_fusion == 'ctx':
            # 上下文相关的处理
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            # 重新排列维度
            loc_attn = genops.rearrange(loc_attn, 'b l t (h k) -> h b l t k', h=self.n_head)
            # 计算查询和位置信息的点积
            loc_attn = torch.einsum('hblk,hbltk->hblt', q, loc_attn) / np.sqrt(q.shape[-1])
        elif self.spatial_attn_fusion == 'cond':
            # 条件相关的处理
            # 计算空间权重
            spatial_weights = self.lang_cond_fc(residual + txt_embeds.unsqueeze(1))
            # 重新排列维度
            spatial_weights = genops.rearrange(spatial_weights, 'b l (h d) -> h b l d', h=self.spatial_n_head, d=self.spatial_dim+1)
            # 如果空间头数为1，则复制到所有头
            if self.spatial_n_head == 1:
                spatial_weights = genops.repeat(spatial_weights, '1 b l d -> h b l d', h=self.n_head)
            # 分离偏置项和权重
            spatial_bias = spatial_weights[..., :1]
            spatial_weights = spatial_weights[..., 1:]
            # 计算位置注意力
            loc_attn = torch.einsum('hbld,bltd->hblt', spatial_weights, pairwise_locs) + spatial_bias
            # 应用sigmoid激活函数
            loc_attn = torch.sigmoid(loc_attn)

        # 如果有键填充掩码，则应用掩码
        if key_padding_mask is not None:
            # 扩展掩码以匹配注意力矩阵的维度
            mask = genops.repeat(key_padding_mask, 'b t -> h b l t', h=self.n_head, l=q.size(2))
            # 将掩码应用于注意力分数
            attn = attn.masked_fill(mask, -np.inf)
            # 根据融合方式应用不同的掩码策略
            if self.spatial_attn_fusion in ['mul', 'cond']:
                loc_attn = loc_attn.masked_fill(mask, 0)
            else:
                loc_attn = loc_attn.masked_fill(mask, -np.inf)

        # 根据融合方式计算最终的融合注意力
        if self.spatial_attn_fusion == 'add':
            # 加法融合：对注意力分数和位置注意力分别进行softmax后求平均
            fused_attn = (torch.softmax(attn, 3) + torch.softmax(loc_attn, 3)) / 2
        else:
            # 其他融合方式
            if self.spatial_attn_fusion in ['mul', 'cond']:
                # 对数域加法：将位置注意力转换到对数域后与注意力分数相加
                fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn
            else:
                # 直接相加
                fused_attn = loc_attn + attn
            # 对融合后的注意力进行softmax
            fused_attn = torch.softmax(fused_attn, 3)
        
        # 检查是否有NaN值
        assert torch.sum(torch.isnan(fused_attn) == 0), print(fused_attn)

        # 使用融合注意力对值进行加权求和
        output = torch.einsum('hblt,hbtv->hblv', fused_attn, v)
        # 重新排列输出维度
        output = genops.rearrange(output, 'head b l v -> b l (head v)')
        # 应用线性变换、dropout和层归一化
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        # 返回输出和融合注意力矩阵
        return output, fused_attn
    
# Transformer编码器层类
class TransformerEncoderLayer(nn.Module):

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
        activation="relu"
    ):
        # 调用父类初始化方法
        super().__init__()
        # 定义自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 前馈网络的实现
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 第一个线性层
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 第二个线性层

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        self.activation = _get_activation_fn(activation)

    def forward(
        self, src, src_mask: Optional[Tensor] = None,# type: ignore
        src_key_padding_mask: Optional[Tensor] = None,# type: ignore
    ):
        # 第一个归一化层
        src2 = self.norm1(src)
        # 自注意力计算
        src2 = self.self_attn(
            src2, src2, value=src2, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        # 残差连接和dropout
        src = src + self.dropout1(src2)
        # 第二个归一化层
        src2 = self.norm2(src)
        # 前馈网络：线性变换 -> 激活函数 -> dropout -> 线性变换
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # 残差连接和dropout
        src = src + self.dropout2(src2)
        # 返回处理后的序列
        return src

    def forward_post(
        self, src, src_mask: Optional[Tensor] = None,# type: ignore
        src_key_padding_mask: Optional[Tensor] = None,# type: ignore
    ):
        # 自注意力计算
        src2 = self.self_attn(
            src, src, value=src, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        # 残差连接和dropout
        src = src + self.dropout1(src2)
        # 第一个归一化层
        src = self.norm1(src)
        # 前馈网络：线性变换 -> 激活函数 -> dropout -> 线性变换
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # 残差连接和dropout
        src = src + self.dropout2(src2)
        # 第二个归一化层
        src = self.norm2(src)
        # 返回处理后的序列
        return src
    
# 3D_ISR: spatial condition self-attention2
# 空间条件Transformer解码器层，继承自TransformerDecoderLayer
class TransformerSpatialDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
        spatial_multihead=True, spatial_dim=5, spatial_attn_fusion='mul'
    ):
        # 调用父类初始化方法
        super().__init__(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        # 删除父类的自注意力机制
        del self.self_attn
        # 使用自定义的空间条件自注意力机制
        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dropout=dropout, 
            spatial_multihead=spatial_multihead, 
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )
        
    # 计算成对位置信息的方法
    def calc_pairwise_locs(self, obj_centers, eps=1e-10, pairwise_rel_type='center'):
        # 计算对象中心之间的相对位置
        pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
                        - einops.repeat(obj_centers, 'b l d -> b 1 l d')
        # 计算成对距离
        pairwise_dists = torch.sqrt(torch.sum(pairwise_locs ** 2, 3) + eps)  # (b, l, l)

        # 计算最大距离并进行归一化
        max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
        norm_pairwise_dists = pairwise_dists / einops.repeat(max_dists, 'b -> b 1 1')

        # 计算2D距离
        pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, 3) + eps)
        # 构建包含多种位置信息的张量
        pairwise_locs = torch.stack(
            [norm_pairwise_dists, pairwise_locs[..., 2] / pairwise_dists,
             pairwise_dists_2d / pairwise_dists, pairwise_locs[..., 1] / pairwise_dists_2d,
             pairwise_locs[..., 0] / pairwise_dists_2d],
            dim=3
        )
        # 返回计算得到的位置信息
        return pairwise_locs
        
    # 前向传播方法
    def forward(
        self, tgt, memory, tgt_pairwise_locs,
        tgt_mask: Optional[Tensor] = None, # type: ignore
        memory_mask: Optional[Tensor] = None, # type: ignore
        tgt_key_padding_mask: Optional[Tensor] = None, # type: ignore
        memory_key_padding_mask: Optional[Tensor] = None, # type: ignore
    ):

        # 第一个归一化层
        tgt2 = self.norm1(tgt)
        # 空间条件自注意力计算
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, tgt2, tgt_pairwise_locs,
            key_padding_mask=tgt_key_padding_mask,
            txt_embeds=memory[:, 0],
        )
        # 残差连接和dropout
        tgt = tgt + self.dropout1(tgt2)
        # 第二个归一化层
        tgt2 = self.norm2(tgt)
        # 多头注意力计算（交叉注意力）
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        # 残差连接和dropout
        tgt = tgt + self.dropout2(tgt2)
        # 第三个归一化层
        tgt2 = self.norm3(tgt)
        # 前馈网络：线性变换 -> 激活函数 -> dropout -> 线性变换
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        # 残差连接和dropout
        tgt = tgt + self.dropout3(tgt2)
        # 返回处理后的序列以及自注意力和交叉注意力矩阵
        return tgt, self_attn_matrices, cross_attn_matrices
    
class Inst3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._initialize_params()
        self._load_llama_model()
        self._initialize_projections()
        self._initialize_scene_processing()
        self._load_instructions()
        
        if not self.debug:
            self.p_0_embed, self.p_1_embed = self.prepare_fixed_embed()

        self.last_embed = None

    def _initialize_params(self):
        """Initialize basic parameters from the config."""
        self.add_mcmf = True
        self.add_3d_isr = True
        self.low_resource = self.config.model.low_resource
        self.max_txt_len = self.config.model.max_txt_len
        self.end_sym = self.config.model.end_sym
        self.system_path = self.config.model.system_path
        self.instruction_path = self.config.model.instruction_path
        self.role = self.config.model.role
        self.no_obj = self.config.model.no_obj
        self.add_scene_token = self.config.model.add_scene_token
        self.add_img_token = self.config.model.add_img_token
        self.obj_norm_scale = self.config.model.obj_norm_scale
        self.scene_norm_scale = self.config.model.scene_norm_scale
        self.grad_scale = self.config.model.grad_scale
        self.train_emb = self.config.model.train_emb
        self.train_img_proj = self.config.model.train_img_proj

        self.input_dim = self.config.model.input_dim
        self.img_input_dim = self.config.model.img_input_dim
        self.attr_dim = self.config.model.attr_dim
        self.scene_dim = self.config.model.scene_dim

        self.debug = self.config.debug

    def _load_llama_model(self):
        """Load and configure the LLAMA model."""
        if self.debug:
            self.llama_model = None
            self.llama_dim = 4096
            return

        logger.info('Loading LLAMA')
        llama_model_path = self.config.model.llama_model_path
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False, legacy=False)
        self.llama_dim = self._initialize_llama_model(llama_model_path)
        self._extend_llama_tokenizer()
        logger.info('Loading LLAMA Done')

    def _initialize_llama_model(self, llama_model_path):
        """Initialize the LLAMA model with the specified configurations."""
        llama_dtype = torch.bfloat16
        llama_model_kwargs = {
            "torch_dtype": llama_dtype,
            "attn_implementation": "flash_attention_2"
        }
        
        if self.low_resource:
            llama_model_kwargs.update({
                "load_in_8bit": True,
                "device_map": "auto"
            })
        
        self.llama_model = LlamaForCausalLM.from_pretrained(llama_model_path, **llama_model_kwargs)
        self._freeze_llama_parameters()

        if self.config.model.use_lora:
            self._apply_lora()

        self.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return self.llama_model.config.hidden_size

    def _freeze_llama_parameters(self):
        """Freeze the parameters of the LLAMA model except for trainable parts."""
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False

    def _apply_lora(self):
        """Apply LoRA (Low-Rank Adaptation) to specific layers of the LLAMA model."""
        lora_target_modules = self._find_lora_target_modules()
        lora_config = LoraConfig(
            r=self.config.lora.lora_r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llama_model = get_peft_model(self.llama_model, lora_config)
        self._enable_lora_trainable_parameters()

    def _find_lora_target_modules(self):
        """Identify target modules for LoRA adaptation."""
        def find_linear_layers(model, lora_target_modules):
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(x in name for x in lora_target_modules):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))
        
        return find_linear_layers(self.llama_model, self.config.lora.lora_target_modules)

    def _enable_lora_trainable_parameters(self):
        """Enable training for specific parameters after applying LoRA."""
        self.llama_model.print_trainable_parameters()
        self.llama_model.model.lm_head.weight.requires_grad = True
        self.llama_model.model.lm_head.weight.data = self.llama_model.model.lm_head.weight.data.float()
        self.llama_model.model.model.embed_tokens.weight.requires_grad = True
        self.llama_model.model.model.embed_tokens.weight.data = self.llama_model.model.model.embed_tokens.weight.data.float()

    def _extend_llama_tokenizer(self):
        """Extend the LLAMA tokenizer with additional tokens for objects."""
        objid_tokens = [f"<OBJ{i:03}>" for i in range(200)]
        self.objid_start_idx = len(self.llama_tokenizer)
        self.llama_tokenizer.add_tokens(objid_tokens, special_tokens=True)
        self.objid_end_idx = len(self.llama_tokenizer)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

    def _initialize_projections(self):
        """Initialize the projection layers for objects and images."""
        self.object_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        self.object_img_proj = nn.Sequential(
            nn.Linear(self.img_input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        self.img2obj_proj = nn.Sequential(
            nn.Linear(self.img_input_dim, self.input_dim)
        )
        if not self.train_img_proj:
            self._freeze_projection_layer(self.object_img_proj)

    def _freeze_projection_layer(self, layer):
        """Freeze the parameters of a given projection layer."""
        for p in layer.parameters():
            p.requires_grad = False

    def _initialize_scene_processing(self):
        """Initialize components for scene processing and relations."""
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=self.scene_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(self.scene_dim, self.scene_dim)
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.scene_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.05,
            norm_first=True,
            batch_first=True
        )
        self.relation_module = nn.TransformerEncoder(self.encoder_layer, num_layers=self.config.model.encoder_num_layers)
        self.scene_init_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.scene_dim)
        )
        self.scene_proj = nn.Sequential(
            nn.Linear(self.scene_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )

        if not self.add_scene_token:
            self._freeze_scene_processing_layers()

    def _freeze_scene_processing_layers(self):
        """Freeze the parameters of layers used for scene processing."""
        for p in self.relation_module.parameters():
            p.requires_grad = False
        for p in self.scene_init_proj.parameters():
            p.requires_grad = False
        for p in self.scene_proj.parameters():
            p.requires_grad = False
        for p in self.pos_proj.parameters():
            p.requires_grad = False

    def _load_instructions(self):
        """Load system and instruction texts from specified paths."""
        with open(self.system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])
        with open(self.instruction_path, "r") as f:
            self.instruction = "\n".join([x.strip() for x in f.readlines()])

    def llama_embed_tokens(self, token_ids):
        """Embed tokens using the LLAMA model's token embedding layer."""
        if self.config.model.use_lora:
            return self.llama_model.model.model.embed_tokens(token_ids)
        else:
            return self.llama_model.model.embed_tokens(token_ids)

    def prepare_fixed_embed(self):
        prompt = self.system + " " + self.instruction + " " + self.role[0] + ": " 
        p_0, p_1 = prompt.split("<REPLACE>")
        p_0_token = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=True)
        p_1_token = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False)
        p_0_embed = self.llama_embed_tokens(p_0_token.input_ids).squeeze(0).detach()
        logger.info(f'p_0_embed is {p_0_embed},the length of p_0_embed is {len(p_0_embed[0])}')
        p_1_embed = self.llama_embed_tokens(p_1_token.input_ids).squeeze(0).detach()
        logger.info(f'p_1_embed is {p_1_embed},the length of p_1_embed is {len(p_1_embed[0])}')
        return p_0_embed, p_1_embed

    def get_text_emb(self, text, device="cpu"):
        text_tokens = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        # print(f'llama\'s text_tokens is {text_tokens}')
        embeds = self.llama_embed_tokens(text_tokens.input_ids)
        # print(f'llama\'s embeds is {embeds}')
        if self.train_emb:
            indices = text_tokens.input_ids >= self.ori_vocab_size
            indices = (indices * 1).unsqueeze(-1)
            embeds = (1 - indices) * embeds.detach() + indices * embeds
        else:
            embeds = embeds.detach()
        return embeds

    def encode_object_feat(self, feat, img_feat, locs):
        feat = torch.nn.functional.normalize(feat, dim=-1)
        img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
        return feat, img_feat
    
    @staticmethod
    def get_dist_attention(pos, dist_exp=1):
        # pos (bs, obj_num, 3)
        dist = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.sum(dist.abs()**dist_exp, dim=-1)
        dist_attn = torch.nn.functional.softmax(-dist, dim=-1)
        return dist_attn
    
    def get_fused_object_list(self, embed_obj, fused_embed_obj, embed_scene, scene_mask, obj_id, assigned_ids):
        valid_ids = torch.where(scene_mask)[0].tolist() 
        if self.config.model.use_lora:
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx] # 200 * 4096
        else:
            objid_embeds = self.llama_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]
        if len(valid_ids) == 1:
            object_list_embed = []
            object_list_embed.append(objid_embeds[obj_id])
            if not self.no_obj:
                object_list_embed.append(fused_embed_obj[valid_ids[0]])
            object_list_embed = torch.stack(object_list_embed, dim=0)
            return object_list_embed
        random.shuffle(valid_ids)
        assigned_ids = assigned_ids[valid_ids]
        if not self.train_emb:
            objid_embeds = objid_embeds.detach()
        selected_objid_embeds = objid_embeds[assigned_ids] 
        if self.no_obj:
            if fused_embed_obj is None:
                object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
                object_list_embed[0::2, :] = selected_objid_embeds
                object_list_embed[1::2, :] = embed_scene[valid_ids]
            else:
                object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
                object_list_embed[0::4, :] = selected_objid_embeds
                object_list_embed[1::4, :] = embed_obj[valid_ids]
                object_list_embed[2::4, :] = embed_scene[valid_ids]
                object_list_embed[3::4, :] = fused_embed_obj[valid_ids]
            return object_list_embed
        if fused_embed_obj is None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            object_list_embed[1::2, :] = embed_scene[valid_ids]
        if fused_embed_obj is not None and embed_scene is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::3, :] = selected_objid_embeds
            object_list_embed[1::3, :] = embed_obj[valid_ids]
            object_list_embed[2::3, :] = fused_embed_obj[valid_ids]
        if fused_embed_obj is not None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 4, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::4, :] = selected_objid_embeds
            object_list_embed[1::4, :] = embed_obj[valid_ids]
            object_list_embed[2::4, :] = embed_scene[valid_ids]
            object_list_embed[3::4, :] = fused_embed_obj[valid_ids]

        return object_list_embed # 4n*4096
    
    def get_min_max_coord(self, xyz, scene_mask):
        """Get the minimum and maximum coordinates from masked XYZ positions."""
        scene_mask = scene_mask.unsqueeze(-1).expand_as(xyz)
        masked_xyz_min = torch.where(scene_mask, xyz, torch.full_like(xyz, float('inf')))
        masked_xyz_max = torch.where(scene_mask, xyz, torch.full_like(xyz, float('-inf')))
        mins = masked_xyz_min.min(dim=1)[0]
        maxs = masked_xyz_max.max(dim=1)[0]
        return mins, maxs

    def forward_train(self, scene_feat, scene_img_feat, scene_locs, scene_mask, obj_ids, assigned_ids, questions, answers, is_eval=False, **kwargs):
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        device = object_embed.device
        
        # add MCMF module
        if self.add_mcmf:
            object_embed = object_embed.transpose(0, 1) # 100,32,1024
            object_img_embed = object_img_embed.transpose(0, 1)
            self_attention1 = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=8).to(device=device)
            self_attention_with_token2 = nn.SelfAttentionWithLearnableToken(embed_dim=self.img_input_dim, num_heads=8).to(device=device)
            object_embed = self_attention1(object_embed, object_embed, object_embed)[0]
            object_img_embed = self_attention_with_token2(object_img_embed, object_img_embed, object_img_embed)[0]
            object_embed = object_embed.transpose(0, 1) # 32,100,1024
            object_img_embed = object_img_embed.transpose(0, 1)

            cross_attention = FeatureCrossAttention(query_dim=self.input_dim, key_dim=self.img_input_dim, output_dim=self.llama_dim, num_heads=8)
            cross_attention.to(device=device)
            fused_object_embed = cross_attention(query=object_embed, key=object_img_embed, value=object_img_embed) #(32,100,4096)
            fused_object_embed.to(device)
            batch_size = object_embed.shape[0]
            proj_object_embed = self.object_proj(object_embed)
            proj_scene_embed = None


        # add 3D-ISR module
        if self.add_3d_isr:  
            if self.add_img_token:
                object_img_embed = self.img2obj_proj(object_img_embed) #768->1024
                object_embed = object_embed + object_img_embed

            obj_embed = self.scene_init_proj(object_embed) #1024->256
            
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask) 
            # position awareness
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed) #256->256

            
            pos_attn = self.get_dist_attention(scene_locs[:, :, :3])

            obj_pos_rel = torch.matmul(pos_attn, pos_embed)
            scene_embed =  obj_embed + obj_pos_rel

            spatial_embed = obj_embed + pos_embed
            decoder_layer = TransformerSpatialDecoderLayer(self.config.hidden_size, self.config.num_attention_heads,
            dim_feedforward=2048, dropout=0.1, activation='gelu', **kwargs)
            pairwise_locs = decoder_layer.calc_pairwise_locs(
                scene_locs[:, :, :3], scene_locs[:, :, 3:], 
                pairwise_rel_type=self.config.pairwise_rel_type
            )
            scene_embed = decoder_layer(scene_locs,pairwise_locs,spatial_embed,scene_feat)
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~scene_mask)
            proj_scene_embed = self.scene_proj(scene_embed) #256->4096
        
        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0
        p_0_embed = self.p_0_embed.to(device)
        p_1_embed = self.p_1_embed.to(device)

        for i, question in enumerate(questions):
            prompt = f" {question} {self.role[1]}: "
            prompt_embed = self.get_text_emb(prompt, device=device).squeeze(0)
            object_list_embed = self.get_fused_object_listt(
                proj_object_embed[i],
                fused_object_embed[i] if self.add_img_token else None,
                proj_scene_embed[i] if self.add_scene_token else None,
                scene_mask[i],
                obj_ids[i],
                assigned_ids[i]
            )
            
            wrapped_embed = torch.cat([p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=0)
            wrapped_attn = torch.ones(wrapped_embed.size()[:-1], dtype=torch.long).to(wrapped_embed.device)
            empty_target = (
                torch.ones(wrapped_attn.shape[0], dtype=torch.long).to(device).fill_(-100)
            )

            answer = answers[i] + self.end_sym
            to_regress_token = self.llama_tokenizer(answer, return_tensors="pt", add_special_tokens=False).to(device)

            answer_target = to_regress_token.input_ids.masked_fill(
                to_regress_token.input_ids == self.llama_tokenizer.pad_token_id, -100
            ).squeeze(0)
            to_regress_embed = self.llama_model.model.embed_tokens(to_regress_token.input_ids).squeeze(0).detach()
            to_regress_embed = self.get_text_emb(answer, device=device).squeeze(0)

            
            target = torch.cat([empty_target, answer_target], dim=0)
            
            input_embed = torch.cat([wrapped_embed, to_regress_embed], dim=0)
            attn = torch.cat([wrapped_attn, to_regress_token.attention_mask[0]], dim=0)
            input_embed_list.append(input_embed)
            attn_list.append(attn)
            target_list.append(target)
            max_seq_len = max(max_seq_len, target.shape[0])
        
        max_seq_len = min(768, max_seq_len)

        def pad_and_trim(tensor_list, max_len, batch_first=True, padding_value=0):
            padded = pad_sequence(tensor_list, batch_first=batch_first, padding_value=padding_value)
            if padded.shape[1] > max_len:
                return padded[:, :max_len]
            return padded
        
        input_embeds = pad_and_trim(input_embed_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        attention_mask = pad_and_trim(attn_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        targets = pad_and_trim(target_list, max_seq_len, batch_first=True, padding_value=-100).to(device)

        
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=input_embeds, 
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets 
            )

        return dict(
            loss=outputs.loss,
            obj_norm=fused_object_embed.norm(dim=-1).mean().detach().cpu(),
            scene_norm=proj_scene_embed.norm(dim=-1).mean().detach().cpu() if proj_scene_embed is not None else 0.,
            max_seq_len=max_seq_len
        )

    def evaluate(self, scene_feat, scene_img_feat, scene_locs, scene_mask, custom_prompt, obj_ids, assigned_ids, is_eval=True, **kwargs):
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        device = object_embed.device
        batch_size, obj_num = object_embed.shape[:2]
        proj_object_embed = self.object_proj(object_embed)
        
        # add MCMF module
        if self.add_mcmf:
            object_embed = object_embed.transpose(0, 1) # 100,32,1024
            object_img_embed = object_img_embed.transpose(0, 1)
            self_attention1 = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=8).to(device=device)
            self_attention_with_token2 = nn.SelfAttentionWithLearnableToken(embed_dim=self.img_input_dim, num_heads=8).to(device=device)
            object_embed = self_attention1(object_embed, object_embed, object_embed)[0]
            object_img_embed = self_attention_with_token2(object_img_embed, object_img_embed, object_img_embed)[0]
            object_embed = object_embed.transpose(0, 1) # 32,100,1024
            object_img_embed = object_img_embed.transpose(0, 1)

            cross_attention = FeatureCrossAttention(query_dim=self.input_dim, key_dim=self.img_input_dim, output_dim=self.llama_dim, num_heads=8)
            cross_attention.to(device=device)
            fused_object_embed = cross_attention(query=object_embed, key=object_img_embed, value=object_img_embed) #(32,100,4096)
            fused_object_embed.to(device)
            batch_size = object_embed.shape[0]
            proj_object_embed = self.object_proj(object_embed)
            proj_scene_embed = None
        
        # add 3D-ISR module
        if self.add_3d_isr:  
            if self.add_img_token:
                object_img_embed = self.img2obj_proj(object_img_embed) #768->1024
                object_embed = object_embed + object_img_embed

            obj_embed = self.scene_init_proj(object_embed) #1024->256
            
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask) 
            # position awareness
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed) #256->256

            
            pos_attn = self.get_dist_attention(scene_locs[:, :, :3])

            obj_pos_rel = torch.matmul(pos_attn, pos_embed)
            scene_embed =  obj_embed + obj_pos_rel

            spatial_embed = obj_embed + pos_embed
            decoder_layer = TransformerSpatialDecoderLayer(self.config.hidden_size, self.config.num_attention_heads,
            dim_feedforward=2048, dropout=0.1, activation='gelu', **kwargs)
            pairwise_locs = decoder_layer.calc_pairwise_locs(
                scene_locs[:, :, :3], scene_locs[:, :, 3:], 
                pairwise_rel_type=self.config.pairwise_rel_type
            )
            scene_embed = decoder_layer(scene_locs,pairwise_locs,spatial_embed,scene_feat)
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~scene_mask)
            proj_scene_embed = self.scene_proj(scene_embed) #256->4096

        output_texts = []
        p_0_embed = self.p_0_embed.to(device).unsqueeze(0)
        p_1_embed = self.p_1_embed.to(device).unsqueeze(0)
        for i in range(batch_size):
            tmp_prompt = f" {custom_prompt[i]} {self.role[1]}: "
            tmp_prompt = update_caption(tmp_prompt, assigned_ids[i])
            prompt_embed = self.get_text_emb(tmp_prompt, device=device)
            
            object_list_embed = self.get_fused_object_list(
                proj_object_embed[i],
                fused_object_embed[i] if self.add_img_token else None,
                proj_scene_embed[i] if self.add_scene_token else None,
                scene_mask[i],
                obj_ids[i],
                assigned_ids[i]
            )
            object_list_embed = object_list_embed.unsqueeze(0)
            wrapped_embed = torch.cat([p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=1)
            with self.maybe_autocast():
                outputs = self.llama_model.generate(
                    inputs_embeds=wrapped_embed,
                    max_new_tokens=self.max_txt_len,
                    num_beams=5,
                    do_sample=True,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=3.0,
                    length_penalty=1,
                    temperature=1.0,
                )
            output_token = outputs[0]
            output_text = self.llama_tokenizer.decode(output_token)
            output_text = output_text.split(self.end_sym)[0]
            output_text = output_text.replace('  ', ' ').replace(' .', '.').strip()
            output_text = recover_caption(output_text, assigned_ids[i].tolist())
            output_texts.append(output_text)
        return output_texts

    def forward(self, **kwargs):
        if "answers" in kwargs:
            return self.forward_train(**kwargs)
        if "custom_prompt" in kwargs:
            return self.evaluate(**kwargs)
        return None

    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt").input_ids.shape[1]

    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device
