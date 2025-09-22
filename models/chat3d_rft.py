import random
import logging
from abc import ABC
from typing import Optional
from collections import Counter

import torch
from torch import Tensor
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from torch.nn.utils.rnn import pad_sequence

from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
from models.position_embedding import PositionEmbeddingCoordsSine
from peft import LoraConfig, get_peft_model
# from models.load_llama import init_llama_model
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerDecoderLayer

import contextlib
from dataset.base_dataset import update_caption, recover_caption

# import visualize_features

logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)

def nclamp(input, min, max):
    return input.clamp(min=min, max=max).detach() + input - input.detach()


def print_grad_status(model):
    """Call this function after losses.backward()
    and it will find out all variables without grad, which
    means that the varaible is not in the graph.
    """
    """
    遍历模型的所有参数，打印每个参数的名称、是否可训练、是否有梯度以及形状信息
    """
    for name, p in model.named_parameters():
        print('{:80s}{:20s}{:20s}{}'.format(name,
            '(Trainable)' if p.requires_grad else '(Fixed)',
            '(Has grad):' if p.grad is not None else '(No grad backward):',
            list(p.shape)))

class SpatialRelationAttention(nn.Module):
    """
    3D空间关系注意力模块（使用标准Transformer机制）
    
    对于3D物体{O_1,O_2,...,O_n}，其中心点坐标为{C_1,C_2,...,C_n}。
    对每一对物体{O_i,O_j}，计算它们的欧式距离和角度关系，构建空间关系特征。
    通过空间条件注意力机制，捕获每个实例在整个3D场景中的成对空间关系。
    """
    def __init__(self, feat_dim=1024, pos_dim=5, num_heads=8, spatial_multihead=True, spatial_attn_fusion='mul'):
        """
        初始化空间关系注意力模块
        
        Args:
            feat_dim (int): 物体特征维度
            pos_dim (int): 位置特征维度，默认为5[sin(θ_h), cos(θ_h), sin(θ_v), cos(θ_v), d_ij]
            num_heads (int): 注意力头数
            spatial_multihead (bool): 是否使用多头空间注意力
            spatial_attn_fusion (str): 空间注意力融合方式 mul add bias
            融合方式选择：
                mul 强调空间关系的权重作用，可能适合空间关系主导的场景。
                add 平衡特征和空间信息，适合混合场景。
                bias 直接叠加，可能在空间信息较弱时效果更好。
        """
        super(SpatialRelationAttention, self).__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.spatial_multihead = spatial_multihead
        self.spatial_attn_fusion = spatial_attn_fusion
        
        # 标准Transformer注意力机制参数，用于生成查询（Query）、键（Key）和值（Value），实现标准的多头自注意力机制
        self.w_qs = nn.Linear(feat_dim, feat_dim)
        self.w_ks = nn.Linear(feat_dim, feat_dim)
        self.w_vs = nn.Linear(feat_dim, feat_dim)
        
        # 输出线性层和正则化层
        self.fc = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(feat_dim)
        
        # 空间注意力头数设置
        self.spatial_n_head = num_heads if spatial_multihead else 1
        
        # 用于计算spatial conditioned attention weight l_i = W_P^T(P_i + O_i)
        self.w_p = nn.Linear(3 + feat_dim, pos_dim, bias=False)
        
        # 场景级令牌生成模块
        # Transformer编码器层
        self.scene_transformer_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, 
            nhead=num_heads, 
            dim_feedforward=2048, 
            dropout=0.1, 
            batch_first=True
        )
        # Transformer编码器
        self.scene_transformer = nn.TransformerEncoder(self.scene_transformer_layer, num_layers=2)
        # 最大池化层
        self.scene_pool = nn.AdaptiveMaxPool1d(1)
        # 两层MLP
        self.scene_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        
    def forward(self, objects, positions):
        """
        前向传播，计算空间关系注意力
        
        Args:
            objects (Tensor): 3D物体特征 [batch_size, num_objects, feat_dim]
            positions (Tensor): 3D物体位置坐标 [batch_size, num_objects, 3] (x, y, z)
            

        Returns:
            enhanced_objects (Tensor): 增强后的物体特征 [batch_size, num_objects, feat_dim]
        """
        batch_size, num_objects, feat_dim = objects.shape
        
        # 保存残差连接
        residual = objects
        
        # 对查询、键、值进行线性变换并重新排列维度
        q = einops.rearrange(self.w_qs(objects), 'b l (head k) -> head b l k', head=self.num_heads)# [num_heads, batch_size, num_objects, head_dim]，head_dim = feat_dim / num_heads
        k = einops.rearrange(self.w_ks(objects), 'b t (head k) -> head b t k', head=self.num_heads)
        v = einops.rearrange(self.w_vs(objects), 'b t (head v) -> head b t v', head=self.num_heads)
        
        # 计算注意力分数，即 Q 和 K 的点积，缩放后得到注意力权重 [num_heads, batch_size, num_objects, num_objects]
        attn = torch.einsum('hblk,hbtk->hblt', q, k) / np.sqrt(q.shape[-1])
        
        # 计算成对关系特征 s_ij = [sin(θ_h), cos(θ_h), sin(θ_v), cos(θ_v), d_ij]
        # 扩展维度以便计算成对关系
        pos_expanded_1 = positions.unsqueeze(2)  # [batch_size, num_objects, 1, 3]
        pos_expanded_2 = positions.unsqueeze(1)  # [batch_size, 1, num_objects, 3]
        
        # 计算坐标差值
        delta = pos_expanded_2 - pos_expanded_1  # [batch_size, num_objects, num_objects, 3]
        
        # 计算欧式距离 d_ij = ||C_i - C_j||_2
        d_ij = torch.norm(delta, dim=-1)  # [batch_size, num_objects, num_objects]
        
        # 计算水平角度 θ_h = arctan2((y_j-y_i)/(x_j-x_i))
        # 注意：需要处理x_j=x_i的情况
        theta_h = torch.atan2(delta[..., 1], delta[..., 0] + 1e-8)  # [batch_size, num_objects, num_objects]
        
        # 计算垂直角度 θ_v = arcsin((z_j-z_i)/d_ij)
        # 注意：需要处理d_ij=0的情况
        theta_v = torch.asin(delta[..., 2] / (d_ij + 1e-8))  # [batch_size, num_objects, num_objects]
        
        # 构建成对关系特征 s_ij = [sin(θ_h), cos(θ_h), sin(θ_v), cos(θ_v), d_ij]
        s_ij = torch.stack([
            torch.sin(theta_h),
            torch.cos(theta_h),
            torch.sin(theta_v),
            torch.cos(theta_v),
            d_ij
        ], dim=-1)  # [batch_size, num_objects, num_objects, 5]
        
        # 计算spatial conditioned attention weight l_i = W_P^T(P_i + O_i)
        # 扩展维度以便计算
        pos_expanded = positions.unsqueeze(2).expand(-1, -1, num_objects, -1)  # [batch_size, num_objects, num_objects, 3]
        obj_expanded = objects.unsqueeze(1).expand(-1, num_objects, -1, -1)    # [batch_size, num_objects, num_objects, feat_dim]
        
        # 拼接位置和物体特征
        pos_obj_combined = torch.cat([pos_expanded, obj_expanded], dim=-1)     # [batch_size, num_objects, num_objects, pos_dim + feat_dim]
        
        # 计算l_i
        l_i = self.w_p(pos_obj_combined)  # [batch_size, num_objects, num_objects, pos_dim]
        
        # 计算w_ij = l_i * s_ij
        w_ij = torch.sum(l_i * s_ij, dim=-1)  # [batch_size, num_objects, num_objects]
        
        # 重新排列维度以匹配注意力矩阵
        w_ij = w_ij.unsqueeze(0).expand(self.spatial_n_head, -1, -1, -1)  # [spatial_n_head, batch_size, num_objects, num_objects]
        
        # 如果不使用多头空间注意力，则复制到所有头
        if not self.spatial_multihead:
            w_ij = einops.repeat(w_ij, 'h b l t -> (h nh) b l t', nh=self.num_heads)
        
        # 使用sigsoftmax融合函数: w_ij = sigmoid(w_ij) * exp(w_ij^o) / sum(sigmoid(w_il) * exp(w_il^o))
        # 先计算标准注意力的softmax
        attn_softmax = torch.softmax(attn, dim=3)  # [num_heads, batch_size, num_objects, num_objects]
        
        # 计算sigsoftmax
        sigmoid_w_ij = torch.sigmoid(w_ij)
        fused_attn = sigmoid_w_ij * attn_softmax
        
        # 归一化
        fused_attn = fused_attn / (torch.sum(fused_attn, dim=3, keepdim=True) + 1e-8)
        
        # 使用融合注意力对值进行加权求和
        output = torch.einsum('hblt,hbtv->hblv', fused_attn, v)
        # 重新排列输出维度
        output = einops.rearrange(output, 'head b l v -> b l (head v)')
        # 应用线性变换、dropout和层归一化
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        # 返回输出
        return output
    
    # def generate_scene_token(self, enhanced_objects):
    #     """
    #     生成场景级令牌
    #     Args:
    #         enhanced_objects (Tensor): 增强后的物体特征 [batch_size, num_objects, feat_dim]
    #     Returns:
    #         scene_token (Tensor): 场景级令牌 [batch_size, feat_dim]
    #     """
    #     # 通过Transformer编码器处理增强的实例级特征
    #     transformed_objects = self.scene_transformer(enhanced_objects)  # [batch_size, num_objects, feat_dim]
    #     # 转换维度以适应池化层
    #     batch_size, num_objects, feat_dim = transformed_objects.shape
    #     transformed_objects = transformed_objects.transpose(1, 2)  # [batch_size, feat_dim, num_objects]
    #     # 应用最大池化
    #     pooled_objects = self.scene_pool(transformed_objects)  # [batch_size, feat_dim, 1]
    #     # 压缩最后一个维度
    #     pooled_objects = pooled_objects.squeeze(-1)  # [batch_size, feat_dim]
    #     # 通过MLP生成最终的场景级令牌
    #     scene_token = self.scene_mlp(pooled_objects)  # [batch_size, feat_dim]
    #     return scene_token

class Chat3D(nn.Module):
    """
    VideoChat model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        llama_model_path = config.model.llama_model_path
        self.low_resource = config.model.low_resource
        self.max_txt_len = config.model.max_txt_len
        self.end_sym = config.model.end_sym
        self.system_path = config.model.system_path
        self.instruction_path = config.model.instruction_path
        self.role = config.model.role
        self.no_obj = config.model.no_obj
        self.add_scene_token = config.model.add_scene_token
        self.add_img_token = config.model.add_img_token
        self.train_emb = config.model.train_emb
        self.train_img_proj = config.model.train_img_proj
        self.input_dim = config.model.input_dim
        self.img_input_dim = config.model.img_input_dim
        self.attr_dim = config.model.attr_dim
        self.scene_dim = config.model.scene_dim
        self.pos_dim = config.model.pos_dim
        self.max_obj_num = config.model.max_obj_num
        self.bidirection = config.model.bidirection
        self.add_pos_emb = config.model.add_pos_emb
        self.feat_fusion = config.model.feat_fusion
        self.fuse_with_id = config.model.fuse_with_id
        self.use_location_token = config.model.use_location_token
        self.train_spatial_attn = config.model.train_spatial_attn

        # 空间多层级特征分组配置
        initial_weights = torch.tensor([0.5, 0.3, 0.2])  # 三层级权重
        self.multi_scale_weights = nn.Parameter(initial_weights)

        self.debug = config.debug
        if not self.debug:
            logger.info('Loading LLaMA')
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False, legacy=False)
            # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            if self.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map="auto",
                    attn_implementation="flash_attention_2"
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                )
            # print(torch.cuda.memory_allocated(device="cuda:0")/1e9)
            # self.llama_model = self.llama_model.to("cuda")
            # print(torch.cuda.memory_allocated(device="cuda:0")/1e9)
            # breakpoint()
            logger.info("freeze LLAMA")
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

            if config.model.use_lora:
                def find_linear_layers(model, lora_target_modules):
                    cls = torch.nn.Linear
                    lora_module_names = set()
                    for name, module in model.named_modules():
                        if (
                            isinstance(module, cls)
                            and all(
                                [
                                    x not in name
                                    for x in [
                                        "instance2embed",
                                        "hidden_state2query"
                                    ]
                                ]
                            )
                            and any([x in name for x in lora_target_modules])
                        ):
                            lora_module_names.add(name)
                    return sorted(list(lora_module_names))
            
                lora_target_modules = find_linear_layers(self.llama_model, config.lora.lora_target_modules)

                lora_config = LoraConfig(
                    r=config.lora.lora_r,
                    lora_alpha=config.lora.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=config.lora.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.llama_model = get_peft_model(self.llama_model, lora_config)
                self.llama_model.print_trainable_parameters()
                self.llama_model.model.lm_head.weight.requires_grad = True
                self.llama_model.model.lm_head.weight.data = self.llama_model.model.lm_head.weight.data.float()
                self.llama_model.print_trainable_parameters()
                self.llama_model.model.model.embed_tokens.weight.requires_grad = True
                self.llama_model.model.model.embed_tokens.weight.data = self.llama_model.model.model.embed_tokens.weight.data.float()
                self.llama_model.print_trainable_parameters()
            else:
                self.llama_model.lm_head.weight.requires_grad = True
                self.llama_model.lm_head.weight.data = self.llama_model.lm_head.weight.data.float()
                self.llama_model.model.embed_tokens.weight.requires_grad = True
                self.llama_model.model.embed_tokens.weight.data = self.llama_model.model.embed_tokens.weight.data.float()
            
            self.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            objid_tokens = []  # 生成物体ID token
            for i in range(self.max_obj_num):
                objid_tokens.append(f"<OBJ{i:03}>")  # 如<OBJ000>, <OBJ001>等
            self.objid_start_idx = self.ori_vocab_size = len(self.llama_tokenizer)  # 记录原始词表大小和添加后的索引范围
            self.llama_tokenizer.add_tokens(objid_tokens, special_tokens=True)  # 将生成的物体ID token添加到tokenizer中， special_tokens=True表示这些是特殊token，不会被普通文本分词处理
            self.objid_end_idx = len(self.llama_tokenizer)  # 新添加的物体token的结束索引
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))  # 调整模型embedding层大小
            # if self.use_location_token:
            #     location_tokens = ["<LOCATION>", "</LOCATION>"]
            #     for i in range(1000):
            #         location_tokens.append(f"<LOC{i:03}>")
            #     self.llama_tokenizer.add_tokens(location_tokens, special_tokens=True)
            #     self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

            self.llama_dim = self.llama_model.config.hidden_size  # 记录LLaMA模型的隐藏层维度
            logger.info('Loading LLAMA Done')
        else:
            self.llama_model = None
            self.llama_dim = 4096
        
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

        if not self.train_img_proj:
            for p in self.object_img_proj.parameters():
                p.requires_grad = False
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=self.pos_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(self.pos_dim, self.llama_dim)
        )
        # 初始化空间关系注意力模块
        self.spatial_relation_attention = SpatialRelationAttention(
            feat_dim=self.input_dim,
            pos_dim=5,  # [sin(θ_h), cos(θ_h), sin(θ_v), cos(θ_v), d_ij]
            num_heads=8,
            spatial_multihead=True,
            spatial_attn_fusion='mul'
        )
        # 根据配置决定是否冻结空间关系注意力模块
        if not self.train_spatial_attn:
            for p in self.spatial_relation_attention.parameters():
                p.requires_grad = False
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.scene_dim, nhead=8, dim_feedforward=2048, dropout=0.05, norm_first=True, batch_first=True)
        # self.relation_module = nn.TransformerEncoder(self.encoder_layer, num_layers=config.model.encoder_num_layers)
        # self.scene_init_proj = nn.Sequential(
        #     nn.Linear(self.input_dim, self.scene_dim)
        # )
        # self.scene_proj = nn.Sequential(
        #     nn.Linear(self.scene_dim, self.llama_dim),
        #     # nn.GELU(),
        #     # nn.Linear(self.llama_dim, self.llama_dim)
        # )
        
        # if not self.add_scene_token:
        #     for p in self.relation_module.parameters():
        #         p.requires_grad = False
        #     for p in self.scene_init_proj.parameters():
        #         p.requires_grad = False
        #     for p in self.scene_proj.parameters():
        #         p.requires_grad = False
                
        # 加载系统提示模板
        with open(self.system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])
        # 加载指令模板 
        with open(self.instruction_path, "r") as f:
            self.instruction = "\n".join([x.strip() for x in f.readlines()])

        if not self.debug:
            self.p_0_embed, self.p_1_embed = self.prepare_fixed_embed()
        self.last_embed = None
        
        # print_grad_status(self)

    def get_objid_embeds(self):
        """获取物体ID token的embedding"""
        # 判断是否使用LoRA适配器
        if self.config.model.use_lora:
            # 从LoRA适配的LLaMA模型中获取物体ID token的embedding
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx] # max_obj_num * 4096
        else:
            # 从原始LLaMA模型中获取物体ID token的embedding
            objid_embeds = self.llama_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]
        return objid_embeds
    
    def llama_embed_tokens(self, token_ids):
        """获取 token_ids 的 token embedding"""
        if self.config.model.use_lora:
            return self.llama_model.model.model.embed_tokens(token_ids)
        else:
            return self.llama_model.model.embed_tokens(token_ids)

    def prepare_fixed_embed(self):
        """预计算固定部分的embedding来减少运行时的计算开销"""
        # 组合系统提示、指令和角色前缀
        prompt = self.system + " " + self.instruction + " " + self.role[0] + ": " 
        print("prompt:", prompt)
        p_0, p_1 = prompt.split("<REPLACE>")
        print("p_0:", p_0)
        print("p_1:", p_1)
        # 对第一部分进行tokenize（添加特殊token）
        p_0_token = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=True)
        # 对第二部分进行tokenize（不添加特殊token）
        p_1_token = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False)
        # 获取第一部分的embedding表示并去除batch维度
        p_0_embed = self.llama_embed_tokens(p_0_token.input_ids).squeeze(0).detach()
        # 获取第二部分的embedding表示并去除batch维度
        p_1_embed = self.llama_embed_tokens(p_1_token.input_ids).squeeze(0).detach()
        return p_0_embed, p_1_embed

    def get_text_emb(self, text, device="cpu"):
        text_tokens = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        embeds = self.llama_embed_tokens(text_tokens.input_ids)
        if self.train_emb:
            indices = text_tokens.input_ids >= self.ori_vocab_size
            indices = (indices * 1).unsqueeze(-1)
            embeds = (1 - indices) * embeds.detach() + indices * embeds
        else:
            embeds = embeds.detach()
        return embeds

    def encode_object_feat(self, feat, img_feat, locs):
        """
        feat : 3D物体的原始特征
        img_feat : 对应的图像特征
        locs : 物体位置信息
        """
        # 特征归一化处理
        feat = torch.nn.functional.normalize(feat, dim=-1)
        img_feat = torch.nn.functional.normalize(img_feat, dim=-1)

        # 截取多层级信息
        feat_level_1, feat_level_2, feat_level_3 = torch.split(feat, 1024, dim=-1)

        # 创建多层级特征表示
        multi_scale_feats = []        
        # 第一个层级
        multi_scale_feats.append(feat_level_1)
        # 第二个层级
        multi_scale_feats.append(feat_level_2)
        # 第三个层级
        multi_scale_feats.append(feat_level_3)

        # 确保权重数量与特征数量匹配
        weights = self.multi_scale_weights[:len(multi_scale_feats)]
        # 归一化权重
        norm_weights = F.softmax(weights, dim=0)
        # 融合多层级特征
        fused_feat = torch.zeros_like(multi_scale_feats[0])
        for i, scale_feat in enumerate(multi_scale_feats):
            fused_feat += norm_weights[i] * scale_feat
        fused_feat = torch.nn.functional.normalize(fused_feat, dim=-1)
        return fused_feat, img_feat

    @staticmethod
    def get_dist_attention(pos, dist_exp=1):
        # pos (bs, obj_num, 3)
        dist = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.sum(dist.abs()**dist_exp, dim=-1)
        dist_attn = torch.nn.functional.softmax(-dist, dim=-1)
        return dist_attn

    def get_object_list_embed(self, embed_obj, embed_img, embed_scene, scene_mask, obj_id, assigned_ids):
        """构建多模态对象特征嵌入列表        
        参数:
            embed_obj: 3D对象特征 [num_objects, feat_dim]
            embed_img: 2D图像特征 [num_objects, feat_dim] 
            embed_scene: 场景级特征 [num_objects, feat_dim]
            scene_mask: 有效对象掩码 [num_objects]
            obj_id: 目标对象ID
            assigned_ids: 对象分配ID            
        返回:
            多模态对象特征组合 [num_valid_objects * num_modalities, feat_dim]
        """
        # 获取有效对象ID
        valid_ids = torch.where(scene_mask)[0].tolist()
        # object_list_embed = []
        # object_list_embed.append(embed_obj[obj_id])
        # object_list_embed = torch.stack(object_list_embed, dim=0)
        # return object_list_embed
        # 加载对象ID的基础嵌入
        if self.config.model.use_lora:
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx] # max_obj_num * 4096
        else:
            objid_embeds = self.llama_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]
        # if len(valid_ids) == 1:
        #     object_list_embed = []
        #     object_list_embed.append(objid_embeds[obj_id])
        #     if not self.no_obj:
        #         object_list_embed.append(embed_obj[valid_ids[0]])
        #     # if embed_scene is not None:
        #     #     object_list_embed.append(embed_scene[valid_ids[0]])
        #     # if embed_img is not None:
        #     #     object_list_embed.append(embed_img[valid_ids[0]])
        #     object_list_embed = torch.stack(object_list_embed, dim=0)
        #     return object_list_embed
        # random.shuffle(valid_ids)

        assigned_ids = assigned_ids[valid_ids]
        if not self.train_emb:
            objid_embeds = objid_embeds.detach()
        selected_objid_embeds = objid_embeds[valid_ids]

        if self.use_location_token:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] += embed_obj[assigned_ids]
            object_list_embed[1::2, :] += embed_img[assigned_ids]
            return object_list_embed
        if self.fuse_with_id:
            object_list_embed = selected_objid_embeds
            if not self.no_obj:
                object_list_embed += embed_obj[assigned_ids]
            if self.add_img_token:
                object_list_embed += embed_img[assigned_ids]
            return object_list_embed
        if self.feat_fusion:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            if not self.no_obj:
                object_list_embed[1::2, :] += embed_obj[assigned_ids]
            if self.add_img_token:
                object_list_embed[1::2, :] += embed_img[assigned_ids]
            return object_list_embed
        if self.no_obj:
            # if embed_img is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            object_list_embed[1::2, :] = embed_img[assigned_ids]
            # else:
            #     object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            #     object_list_embed[0::3, :] = selected_objid_embeds
            #     object_list_embed[1::3, :] = embed_scene[assigned_ids]
            #     object_list_embed[2::3, :] = embed_img[assigned_ids]
            return object_list_embed
        if embed_img is None and embed_scene is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            object_list_embed[1::2, :] = embed_obj[assigned_ids]
            return object_list_embed
            # object_list_embed = selected_objid_embeds + embed_obj[assigned_ids]
        if embed_img is None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::3, :] = selected_objid_embeds
            object_list_embed[1::3, :] = embed_obj[assigned_ids]
            object_list_embed[2::3, :] = embed_scene[assigned_ids]
            return object_list_embed
        if embed_img is not None and embed_scene is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::3, :] = selected_objid_embeds
            object_list_embed[1::3, :] = embed_obj[assigned_ids]
            object_list_embed[2::3, :] = embed_img[assigned_ids]
            return object_list_embed
        if embed_img is not None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 4, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::4, :] = selected_objid_embeds
            object_list_embed[1::4, :] = embed_obj[assigned_ids]
            object_list_embed[2::4, :] = embed_scene[assigned_ids]
            object_list_embed[3::4, :] = embed_img[assigned_ids]
            return object_list_embed
        return object_list_embed

    def get_min_max_coord(self, xyz, scene_mask):
        """计算场景中有效物体的坐标边界(最小/最大xyz值)
        参数:
            xyz: 物体3D坐标张量 [batch_size, num_objects, 3]
            scene_mask: 有效物体掩码 [batch_size, num_objects]
        返回:
            mins: 最小坐标值 [batch_size, 3]
            maxs: 最大坐标值 [batch_size, 3]
        """
        # 扩展掩码维度以匹配xyz形状
        scene_mask = scene_mask.unsqueeze(-1).expand_as(xyz)  # [bs, N, 3]
        # 计算最小坐标(忽略无效物体)
        masked_xyz_min = torch.where(scene_mask, xyz, torch.full_like(xyz, float('inf')))  # [bs, N, 3]
        # 计算最大坐标(忽略无效物体) 
        masked_xyz_max = torch.where(scene_mask, xyz, torch.full_like(xyz, float('-inf')))  # [bs, N, 3]
        # 物体维度求最小/最大值
        mins = masked_xyz_min.min(dim=1)[0]
        maxs = masked_xyz_max.max(dim=1)[0]
        return mins, maxs

    def forward_train(self, scene_feat, scene_img_feat, scene_locs, scene_mask, obj_ids, assigned_ids, questions, answers, is_eval=False, **kwargs):
        """3D场景对话模型的训练前向传播
        核心流程:
        1. 多模态特征编码 → 2. 空间位置处理 → 3. 注意力机制 → 4. 文本生成        
        参数:
            scene_feat: 3D场景特征 [bs, num_objs, feat_dim]
            scene_img_feat: 2D图像特征 [bs, num_objs, feat_dim]
            scene_locs: 物体3D坐标 [bs, num_objs, 3] 
            scene_mask: 有效物体掩码 [bs, num_objs]
            obj_ids: 目标物体ID [bs]
            assigned_ids: 物体分配ID [bs, num_objs]
            questions: 问题文本列表 [bs]
            answers: 答案文本列表 [bs]            
        返回:
            包含各项损失的字典
        """       
        # 获取对象嵌入
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        device = object_embed.device
        batch_size = object_embed.shape[0]
        # 空间关系注意力
        object_embed = self.spatial_relation_attention(object_embed, scene_locs[:, :, :3])
        object_embed = torch.nn.functional.normalize(object_embed, dim=-1)
        
        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = self.object_img_proj(object_img_embed)
        proj_scene_embed = None

        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0
        p_0_embed = self.p_0_embed.to(device)
        p_1_embed = self.p_1_embed.to(device)
        object_list_intervals = []

        for i, question in enumerate(questions):
            # 构建文本提示
            prompt = f"{question} {self.role[1]}: "
            prompt_embed = self.get_text_emb(prompt, device=device).squeeze(0)
            # 获取对象特征列表
            object_list_embed = self.get_object_list_embed(
                proj_object_embed[i], 
                proj_object_img_embed[i] if self.add_img_token else None,
                proj_scene_embed[i] if self.add_scene_token else None, 
                scene_mask[i],
                obj_ids[i],
                assigned_ids[i]
            )
            # object_list_embed = nclamp(object_list_embed, min=-0.05, max=0.05)
            object_list_intervals.append((p_0_embed.shape[0], p_0_embed.shape[0] + object_list_embed.shape[0]))
            # 组合文本和视觉特征
            wrapped_embed = torch.cat([
                p_0_embed, 
                object_list_embed, 
                p_1_embed, 
                prompt_embed
                ], dim=0)
            wrapped_attn = torch.ones(wrapped_embed.size()[:-1], dtype=torch.long).to(wrapped_embed.device)
            empty_target = (
                torch.ones(wrapped_attn.shape[0], dtype=torch.long).to(device).fill_(-100)
            )
            # 处理答案文本
            answer = answers[i] + self.end_sym
            to_regress_token = self.llama_tokenizer(answer, return_tensors="pt", add_special_tokens=False).to(device)
            # breakpoint()
            answer_target = to_regress_token.input_ids.masked_fill(
                to_regress_token.input_ids == self.llama_tokenizer.pad_token_id, -100
            ).squeeze(0)
            # to_regress_embed = self.llama_model.model.embed_tokens(to_regress_token.input_ids).squeeze(0).detach()
            to_regress_embed = self.get_text_emb(answer, device=device).squeeze(0)

            # 构建模型输入
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
        targets = pad_and_trim(target_list, max_seq_len, batch_first=True, padding_value=-100).to(device)
        attention_mask = pad_and_trim(attn_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        # 修改注意力掩码生成方式，考虑空间关系
        if self.bidirection:
            input_dtype = input_embeds.dtype
            causal_mask = torch.ones((max_seq_len, max_seq_len), dtype=input_dtype, device=device)
            causal_mask = torch.tril(causal_mask, diagonal=0)
            causal_mask = causal_mask[None, None, :, :].expand(input_embeds.shape[0], 1, -1, -1).clone()
            padding_mask = causal_mask[..., :].eq(1.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :] = causal_mask[..., :].masked_fill(padding_mask, 0.0)
            for i in range(causal_mask.shape[0]):
                st, ed = object_list_intervals[i]
                causal_mask[i, :, st:ed, st:ed] = 1.0
            attention_mask = causal_mask
        
        # label_weights = torch.ones(self.llama_model.config.vocab_size, device=device)
        # label_weights[self.objid_start_idx:self.objid_end_idx] = 10

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                # label_weights=label_weights
            )

        return dict(
            loss=outputs.loss,
            obj_norm=proj_object_embed.norm(dim=-1).mean().detach().cpu(),
            obj_img_norm=proj_object_img_embed.norm(dim=-1).mean().detach().cpu(),
            objid_norm=self.get_objid_embeds().norm(dim=-1).mean().detach().cpu(),
            scene_norm=proj_scene_embed.norm(dim=-1).mean().detach().cpu() if proj_scene_embed is not None else 0.,
            max_seq_len=max_seq_len
        )

    def evaluate(self, scene_feat, scene_img_feat, scene_locs, scene_mask, custom_prompt, obj_ids, assigned_ids, is_eval=True, **kwargs):
        """3D场景对话模型的评估方法
        核心流程:
        1. 多模态特征编码 → 2. 空间位置处理 → 3. 注意力机制 → 4. 文本生成        
        参数:
            scene_feat: 3D场景特征 [bs, num_objs, feat_dim]
            scene_img_feat: 2D图像特征 [bs, num_objs, feat_dim]
            scene_locs: 物体3D坐标 [bs, num_objs, 3]
            scene_mask: 有效物体掩码 [bs, num_objs]
            custom_prompt: 自定义提示文本列表 [bs]
            obj_ids: 目标物体ID [bs]
            assigned_ids: 物体分配ID [bs, num_objs]            
        返回:
            生成的回答文本列表 [bs]
        """
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        device = object_embed.device
        batch_size, obj_num = object_embed.shape[:2]
        # 空间关系注意力
        object_embed = self.spatial_relation_attention(object_embed, scene_locs[:, :, :3])
        object_embed = torch.nn.functional.normalize(object_embed, dim=-1)
        
        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = self.object_img_proj(object_img_embed)
        proj_scene_embed = None

        output_texts = []
        p_0_embed = self.p_0_embed.to(device).unsqueeze(0)
        p_1_embed = self.p_1_embed.to(device).unsqueeze(0)

        for i in range(batch_size):
            # 构建文本提示
            tmp_prompt = f" {custom_prompt[i]} {self.role[1]}: "
            tmp_prompt = update_caption(tmp_prompt, assigned_ids[i])
            prompt_embed = self.get_text_emb(tmp_prompt, device=device)                     
            # 获取对象特征列表
            object_list_embed = self.get_object_list_embed(
                proj_object_embed[i], 
                proj_object_img_embed[i] if self.add_img_token else None,
                proj_scene_embed[i] if self.add_scene_token else None, 
                scene_mask[i],
                obj_ids[i],
                assigned_ids[i]
            )
            object_list_embed = object_list_embed.unsqueeze(0)
            # 组合文本和视觉特征           
            wrapped_embed = torch.cat([
                p_0_embed, 
                object_list_embed, 
                p_1_embed, 
                prompt_embed
                ], dim=1)
            attention_mask=None
            if self.bidirection:
                seq_len = wrapped_embed.shape[1]
                attention_mask = torch.ones((seq_len, seq_len), dtype=wrapped_embed.dtype, device=device)
                attention_mask = torch.tril(attention_mask, diagonal=0)
                attention_mask = attention_mask[None, None, :, :].expand(1, 1, -1, -1).clone()
                st, ed = p_0_embed.shape[1], p_0_embed.shape[1] + object_list_embed.shape[1]
                attention_mask[:, :, st:ed, st:ed] = 1.0
            
            with self.maybe_autocast():
                outputs = self.llama_model.generate(
                    inputs_embeds=wrapped_embed,
                    max_new_tokens=self.max_txt_len,
                    # stopping_criteria=stopping_criteria,
                    num_beams=5,
                    # do_sample=True,
                    min_length=1,
                    # top_p=0.9,
                    repetition_penalty=3.0,
                    length_penalty=1,
                    temperature=1.0,
                    customized_mask=attention_mask
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
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device