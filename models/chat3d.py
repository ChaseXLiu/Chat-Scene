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

from .twin_transformer import TwinTransformer

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

class AsymmetricSpatialInjector(nn.Module):
    """
    3D位置嵌入
    - 融合绝对坐标 (x, y, z) 与相对位置关系 (距离+角度)
    非对称空间注入模块
    - 核心思想：只将空间信息注入到 3D 特征中，完全保留 2D 特征的纯净性，以保护 QA 任务。
    - 机制：Residual Connection + Gating Mechanism
    """
    def __init__(self, feature_dim=768, pos_input_dim=5, spatial_hidden_dim=256):
        """
        Args:
            feature_dim (int): 3D 特征的原始维度 (这里是 768)。
            pos_input_dim (int): 相对位置特征的基础维度 (默认5: [sin, cos, ...])。
            spatial_hidden_dim (int): 空间特征的中间映射维度。
        """
        super().__init__()

        # 1. 空间特征编码器
        # 将 [3(绝对坐标) + 5(相对聚合)] = 8维 映射到隐空间
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3 + pos_input_dim, spatial_hidden_dim),
            nn.LayerNorm(spatial_hidden_dim),
            nn.ReLU()
        )
        
        # 2. 注入 MLP
        # 输入: [3D特征(768) + 空间特征(256)] = 1024
        # 输出: [注入增量(768)] - 保持与输入维度一致以便相加
        self.injection_mlp = nn.Sequential(
            nn.Linear(feature_dim + spatial_hidden_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 3. 可学习门控参数 alpha
        # 初始化为 0，保证训练初期 feature_3d_new == feature_3d (Identity映射)
        self.gate_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, feature_3d, positions):
        """
        Args:
            feature_3d: [B, N, 768]  原始 Uni3D 特征
            positions:  [B, N, 3]    物体中心坐标
        
        Returns:
            feature_3d_new: [B, N, 768] 注入空间信息后的 3D 特征
        """
        B, N, _ = positions.shape

        # ---- Part A: 计算空间几何特征 ----
        
        # 1. 计算 Pairwise 相对关系
        pos1 = positions.unsqueeze(2)  # [B, N, 1, 3]
        pos2 = positions.unsqueeze(1)  # [B, 1, N, 3]
        delta = pos2 - pos1            # [B, N, N, 3]

        d_ij = torch.norm(delta, dim=-1)
        # 加上 1e-8 防止除零，clamp 防止 asin 越界
        theta_h = torch.atan2(delta[..., 1], delta[..., 0] + 1e-8)
        theta_v = torch.asin((delta[..., 2] / (d_ij + 1e-8)).clamp(-1, 1))

        # 堆叠几何特征 [B, N, N, 5]
        s_ij = torch.stack([
            torch.sin(theta_h), torch.cos(theta_h),
            torch.sin(theta_v), torch.cos(theta_v),
            d_ij
        ], dim=-1) 

        # 2. 聚合相对信息 (Aggregation)
        # 使用 mean 聚合，得到每个物体相对于全局的空间上下文
        s_agg = s_ij.mean(dim=2) # [B, N, 5]

        # 3. 拼接绝对坐标 + 相对聚合 -> [B, N, 8]
        spatial_raw = torch.cat([positions, s_agg], dim=-1)
        
        # 4. 编码得到空间 Embedding -> [B, N, 256]
        spatial_feat = self.spatial_encoder(spatial_raw)

        # ---- Part B: 非对称注入 (在 768 维进行) ----
        
        # 1. 特征拼接: [B, N, 768] cat [B, N, 256] -> [B, N, 1024]
        concat_feat = torch.cat([feature_3d, spatial_feat], dim=-1)
        
        # 2. 计算注入增量 delta -> [B, N, 768]
        injection_delta = self.injection_mlp(concat_feat)
        
        # 3. 门控残差连接
        # 使用 tanh 控制门控范围在 (-1, 1)，初始为 0
        gate = torch.tanh(self.gate_alpha)
        
        # 结果依然是 768 维
        feature_3d_new = feature_3d + gate * injection_delta

        return feature_3d_new


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
        
        # self.object_proj = nn.Sequential(
        #     nn.Linear(self.input_dim, self.llama_dim),
        #     nn.GELU(),
        #     nn.Linear(self.llama_dim, self.llama_dim)
        # )
        # self.object_img_proj = nn.Sequential(
        #     nn.Linear(self.img_input_dim, self.llama_dim),
        #     nn.GELU(),
        #     nn.Linear(self.llama_dim, self.llama_dim)
        # )

        

        tt_hidden_dim = getattr(config.model, "tt_hidden_dim", 768)
        tt_layers = getattr(config.model, "tt_layers", 2)
        tt_heads = getattr(config.model, "tt_heads", 12)
        bert_weights_path = "/home/lcx/chat-scene/Chat-Scene/pretrained_models/bert-base-uncased-pytorch_model.bin"

        self.object_proj = nn.Linear(tt_hidden_dim, self.llama_dim)
        self.object_img_proj = nn.Linear(tt_hidden_dim, self.llama_dim)
        self.scale_factor = 30.0

        self.twin_transformer = TwinTransformer(
            input_2d_dim = self.img_input_dim,         # 2D 特征: 原始2D特征维度
            input_3d_dim = self.input_dim,             # 3D 特征: 原始3D特征维度
            hidden_size = tt_hidden_dim,               # 内部和输出维度
            num_hidden_layers = tt_layers,             # Number of layers
            num_attention_heads = tt_heads,            # Number of attention heads
            hidden_dropout_prob = 0.1,                 # Hidden dropout probability
            bert_weights_path = bert_weights_path
        )

        if not self.train_img_proj:
            for p in self.object_img_proj.parameters():
                p.requires_grad = False
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=self.pos_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(self.pos_dim, self.llama_dim)
        )

        # 实例化空间注入模块
        self.spatial_injector = AsymmetricSpatialInjector(feature_dim=tt_hidden_dim)
  
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
        p_0, p_1 = prompt.split("<REPLACE>")
        # 对第一部分进行tokenize（添加特殊token）
        p_0_token = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=True)
        # 对第二部分进行tokenize（不添加特殊token）
        p_1_token = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False)
        # 获取第一部分的embedding表示并去除batch维度
        p_0_embed = self.llama_embed_tokens(p_0_token.input_ids).squeeze(0).detach()
        # 获取第二部分的embedding表示并去除batch维度
        p_1_embed = self.llama_embed_tokens(p_1_token.input_ids).squeeze(0).detach()
        return p_0_embed, p_1_embed

    # def get_text_emb(self, text, device="cpu"):
    #     text_tokens = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    #     embeds = self.llama_embed_tokens(text_tokens.input_ids)
    #     if self.train_emb:
    #         indices = text_tokens.input_ids >= self.ori_vocab_size
    #         indices = (indices * 1).unsqueeze(-1)
    #         embeds = (1 - indices) * embeds.detach() + indices * embeds
    #     else:
    #         embeds = embeds.detach()
    #     return embeds

    def get_text_emb(self, text, device="cpu"):
        """
        获取文本 Embedding，支持单条或 Batch 处理。
        Args:
            text: str 或 List[str]
        Returns:
            embeds: [B, L, D] (如果输入是str，则 B=1)
            attention_mask: [B, L] (用于指示 padding 位置)
        """
        # 1. 智能 Tokenize
        if isinstance(text, list):
            # Batch 处理：必须开启 padding
            text_tokens = self.llama_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512, # 建议加上防止OOM
                add_special_tokens=False
            ).to(device)
        else:
            # 单条处理：保持原样 (不强制 padding)
            text_tokens = self.llama_tokenizer(
                text, 
                return_tensors="pt", 
                add_special_tokens=False
            ).to(device)

        # 2. 获取原始 Embeddings
        embeds = self.llama_embed_tokens(text_tokens.input_ids)

        # 3. 梯度控制 (这部分逻辑天生支持 Batch Tensor，无需修改)
        if self.train_emb:
            # 判断哪些是新词 (token_id >= ori_vocab_size)
            indices = text_tokens.input_ids >= self.ori_vocab_size
            indices = (indices * 1).unsqueeze(-1) # [B, L, 1]
            
            # 旧词 detach (无梯度)，新词保留梯度
            embeds = (1 - indices) * embeds.detach() + indices * embeds
        else:
            embeds = embeds.detach()

        # ★ 修改返回值：同时返回 mask，这对 Batch 处理至关重要
        return embeds, text_tokens.attention_mask

    # def encode_object_feat(self, feat, img_feat, locs):
    #     """
    #     feat : 3D物体的原始特征
    #     img_feat : 对应的图像特征
    #     locs : 物体位置信息
    #     """
    #     # 特征归一化处理
    #     feat = torch.nn.functional.normalize(feat, dim=-1)
    #     img_feat = torch.nn.functional.normalize(img_feat, dim=-1)

    #     # 截取多层级信息
    #     feat_level_1, feat_level_2, feat_level_3 = torch.split(feat, 1024, dim=-1)

    #     # 创建多层级特征表示
    #     multi_scale_feats = []        
    #     # 第一个层级
    #     multi_scale_feats.append(feat_level_1)
    #     # 第二个层级
    #     multi_scale_feats.append(feat_level_2)
    #     # 第三个层级
    #     multi_scale_feats.append(feat_level_3)

    #     # 确保权重数量与特征数量匹配
    #     weights = self.multi_scale_weights[:len(multi_scale_feats)]
    #     # 归一化权重
    #     norm_weights = F.softmax(weights, dim=0)
    #     # 融合多层级特征
    #     fused_feat = torch.zeros_like(multi_scale_feats[0])
    #     for i, scale_feat in enumerate(multi_scale_feats):
    #         fused_feat += norm_weights[i] * scale_feat
    #     fused_feat = torch.nn.functional.normalize(fused_feat, dim=-1)
    #     return fused_feat, img_feat

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
        
        # 预处理所有文本提示
        prompts = [f"{q} {self.role[1]}: " for q in questions]
        # 获取文本 Embedding
        # features_text_batch: [B, L_max, D]
        # mask_text_batch:     [B, L_max]
        features_text_batch, mask_text_batch = self.get_text_emb(prompts, device=device)

        # 3. 运行 TwinTransformer
        processed_2d, processed_3d = self.twin_transformer(
            features_2d = object_img_embed,
            features_3d = object_embed,
            attention_mask_2d = scene_mask,
            attention_mask_3d = scene_mask
        )

        # 注入空间信息特征 
        processed_3d = self.spatial_injector(processed_3d, scene_locs[:, :, :3])

        # 投影
        proj_object_embed = self.object_proj(processed_3d)
        proj_object_img_embed = self.object_img_proj(processed_2d)

        proj_scene_embed = None

        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0
        p_0_embed = self.p_0_embed.to(device)
        p_1_embed = self.p_1_embed.to(device)

        for i, question in enumerate(questions):
            # 构建文本提示
            # prompt = f"{question} {self.role[1]}: "
            # prompt_embed = self.get_text_emb(prompt, device=device).squeeze(0)
            
            # 获取当前样本的 Prompt Embedding (去除 padding 部分)
            # 通过 mask 找到真实的长度
            valid_len = mask_text_batch[i].sum().item()
            prompt_embed = features_text_batch[i, :valid_len] # [L_real, D]

            # 获取对象特征列表
            object_list_embed = self.get_object_list_embed(
                proj_object_embed[i], 
                proj_object_img_embed[i] if self.add_img_token else None,
                proj_scene_embed[i] if self.add_scene_token else None,
                scene_mask[i],
                obj_ids[i],
                assigned_ids[i]
            )
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
            to_regress_embed, _ = self.get_text_emb(answer, device=device)
            to_regress_embed = to_regress_embed.squeeze(0)

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

        # 预处理所有文本提示
        # update_caption 是 CPU 字符串操作
        prompts = []
        for i in range(batch_size):
            tmp_prompt = f" {custom_prompt[i]} {self.role[1]}: "
            tmp_prompt = update_caption(tmp_prompt, assigned_ids[i])
            prompts.append(tmp_prompt)
        # 获取文本 Embedding
        # features_text_batch: [B, L_max, D]
        # mask_text_batch:     [B, L_max]
        features_text_batch, mask_text_batch = self.get_text_emb(prompts, device=device)
        
        # 运行 TwinTransformer
        processed_2d, processed_3d = self.twin_transformer(
            features_2d=object_img_embed,
            features_3d=object_embed,
            attention_mask_2d=scene_mask,
            attention_mask_3d=scene_mask
        )
         
         # 注入空间信息特征 
        processed_3d = self.spatial_injector(processed_3d, scene_locs[:, :, :3])

        # 投影
        proj_object_embed = self.object_proj(processed_3d)
        proj_object_img_embed = self.object_img_proj(processed_2d)

        proj_scene_embed = None
        
        output_texts = []
        p_0_embed = self.p_0_embed.to(device).unsqueeze(0)
        p_1_embed = self.p_1_embed.to(device).unsqueeze(0)

        for i in range(batch_size):
            # 2. 取出当前样本的 Prompt Embedding (去除 Padding)
            # 生成时不能带 padding token
            valid_len = mask_text_batch[i].sum().item()
            prompt_embed = features_text_batch[i, :valid_len].unsqueeze(0) # [1, L_real, D]

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
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device
