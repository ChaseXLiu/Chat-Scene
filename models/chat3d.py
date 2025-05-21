import random
import logging
from abc import ABC

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

        # 空间多尺度特征分组配置
        self.use_multi_scale = config.model.use_multi_scale
        self.num_scales = config.model.num_scales
        self.multi_scale_levels = ['semantic', 'geometry', 'texture']
        initial_weights = torch.tensor([0.5, 0.3, 0.2])  # 语义、几何、纹理的初始权重
        # initial_weights = torch.ones(self.num_scales) / self.num_scales  # 均匀分配权重
        self.multi_scale_weights = nn.Parameter(initial_weights)

        # 文本多尺度处理配置
        self.use_text_multi_scale = config.model.use_text_multi_scale
        self.text_scale_levels = ['coarse', 'fine', 'detailed']
        # self.text_scale_weights = nn.Parameter(torch.ones(len(self.text_scale_levels)) / len(self.text_scale_levels))
        initial_weights = torch.tensor([0.5, 0.3, 0.2])  # 粗粒度、细粒度、详细描述的初始权重
        # initial_weights = torch.ones(len(self.text_scale_levels)) / len(self.text_scale_levels)  # 均匀分配权重
        self.text_scale_weights = nn.Parameter(initial_weights)

        # # 特征一致性损失配置
        # self.use_feature_consistency = config.model.use_feature_consistency
        # self.feature_consistency_weight = config.model.feature_consistency_weight
        
        # # 空间关系注意力配置
        # self.use_spatial_attention = config.model.use_spatial_attention
        # self.spatial_attention_weight = nn.Parameter(torch.tensor(1.0))  # 可学习的权重参数

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

        # 为多尺度表示创建投影层
        # 添加可学习的多尺度权重
        # self.multiscale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        # 添加多尺度融合层
        # self.multi_scale_fusion = nn.Sequential(
        #     nn.Linear(self.llama_dim, self.llama_dim),
        #     nn.GELU(),
        #     nn.Linear(self.llama_dim, self.llama_dim),
        # )
        # # 语义投影层 (PV-T): 将全局语义特征映射到语言空间，用于与LLM的文本交互
        # self.semantic_proj = nn.Sequential(
        #     nn.Linear(self.input_dim, self.llama_dim),
        #     nn.GELU(),
        #     nn.Linear(self.llama_dim, self.llama_dim)
        # )            
        # # 几何投影层 (PV-D): 将局部几何特征映射到细粒度的空间，用于高精度任务
        # self.geometry_proj = nn.Sequential(
        #     nn.Linear(self.input_dim, self.llama_dim),
        #     nn.GELU(),
        #     nn.Linear(self.llama_dim, self.llama_dim)
        # )            
        # # 纹理投影层 (PV-T): 将纹理特征映射到语言空间，用于描述视觉细节
        # self.texture_proj = nn.Sequential(
        #     nn.Linear(self.input_dim, self.llama_dim),
        #     nn.GELU(),
        #     nn.Linear(self.llama_dim, self.llama_dim)
        # )            
        # 添加注意力机制，使投影层能够动态选择重要特征
        # self.feature_attention = nn.Sequential(
        #     nn.Linear(self.input_dim, 128),  # 降维
        #     nn.GELU(),
        #     nn.Linear(128, self.num_scales),
        #     nn.Softmax(dim=-1)
        # )
        # self.feature_attention = None
        # # 多尺度交叉注意力层
        # self.cross_attentions = nn.ModuleList([
        #     nn.MultiheadAttention(embed_dim=self.llama_dim, 
        #                         num_heads=4,
        #                         dropout=0.1,
        #                         batch_first=True)
        #     for _ in range(self.num_scales)
        # ])            
        # 特征融合层
        # self.fusion_proj = nn.ModuleList([
        #     nn.Linear(self.llama_dim * 2, self.llama_dim)
        #     for _ in range(self.num_scales)
        # ])

        if not self.train_img_proj:
            for p in self.object_img_proj.parameters():
                p.requires_grad = False
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=self.pos_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(self.pos_dim, self.llama_dim)
        )
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

    def get_text_emb(self, text, device="cpu"):
        """为文本提供embedding表示
        同时支持对新增token的embedding进行选择性训练
        这对于扩展LLaMA的词表(如添加物体ID token)非常重要        
        参数:
            text: 输入文本
            device: 计算设备
            multi_scale: 是否返回多尺度表示
        """
        text_tokens = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)  # 将文本转换为token ID序列
        embeds = self.llama_embed_tokens(text_tokens.input_ids)  # 获取token对应的embedding
        if self.train_emb:
            # 只训练新增token的embedding，保持原始token的embedding冻结
            indices = text_tokens.input_ids >= self.ori_vocab_size
            indices = (indices * 1).unsqueeze(-1)
            embeds = (1 - indices) * embeds.detach() + indices * embeds
        else:
            embeds = embeds.detach()
        # return embeds
        # if not self.use_text_multi_scale:
        #     return embeds
        
        # # 多尺度文本处理
        base_embed = embeds.squeeze(0)
        multi_scale_embeds = []
        # 第一个尺度：全局语义 - 捕捉整体语义信息
        # 使用平均池化获取全局表示，并增强句子开头和结尾的权重
        with torch.no_grad():
            seq_len = base_embed.shape[0]
            # 创建位置权重，句子开头和结尾通常包含更多全局信息
            pos_weights = torch.ones(seq_len, device=device)
            pos_weights[:min(5, seq_len)] = 1.5  # 增强开头权重
            pos_weights[max(0, seq_len-5):] = 1.5  # 增强结尾权重
            # 应用位置权重
            coarse_embed = base_embed * pos_weights.unsqueeze(-1)
        multi_scale_embeds.append(coarse_embed)

        # 第二个尺度：局部几何 - 关注物体之间的相对位置关系
        # 使用自注意力增强局部关系表示
        with torch.no_grad():
            # 计算token间的注意力权重
            attn_weights = torch.matmul(base_embed, base_embed.transpose(0, 1)) / (self.llama_dim**0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)
            # 应用注意力权重获取上下文增强的表示
            fine_embed = torch.matmul(attn_weights, base_embed)
            
            # 增强关系词的权重
            relation_keywords = ["in", "on", "at", "near", "between", "beside", "under", "above", "left", "right", "front", "back"]
            relation_mask = torch.zeros(seq_len, device=device)
            
            # 解码每个token并检查是否包含关系词
            for i, token_id in enumerate(text_tokens.input_ids[0]):
                token = self.llama_tokenizer.decode(token_id)
                if any(keyword in token.lower() for keyword in relation_keywords):
                    relation_mask[i] = 1.0
            
            # 应用关系词增强
            fine_embed = fine_embed + fine_embed * relation_mask.unsqueeze(-1) * 0.5
        multi_scale_embeds.append(fine_embed)

        # 第三个尺度：细粒度纹理 - 关注物体的细节描述
        # 增强描述性词汇的权重
        with torch.no_grad():
            # 基础表示
            detailed_embed = base_embed.clone()
            
            # 定义描述性词汇关键词
            detail_keywords = ["color", "texture", "material", "size", "shape", "white", "brown", "small", "large", "wooden", "metal", "glass"]
            detail_mask = torch.zeros(seq_len, device=device)
            
            # 解码每个token并检查是否包含描述性词汇
            for i, token_id in enumerate(text_tokens.input_ids[0]):
                token = self.llama_tokenizer.decode(token_id)
                if any(keyword in token.lower() for keyword in detail_keywords):
                    detail_mask[i] = 1.0
            
            # 应用描述性词汇增强
            detailed_embed = detailed_embed + detailed_embed * detail_mask.unsqueeze(-1) * 0.5
        multi_scale_embeds.append(detailed_embed)

        # 自适应权重
        # 计算每个尺度的特征统计信息
        scale_stats = []
        for embed in multi_scale_embeds:
            # 计算每个尺度的统计特征（均值和方差）
            mean_feat = torch.mean(embed, dim=0, keepdim=True)
            var_feat = torch.var(embed, dim=0, keepdim=True)
            scale_stats.append(torch.cat([mean_feat, var_feat], dim=-1))        
        scale_stats = torch.cat(scale_stats, dim=0)  # [num_scales, 2*llama_dim]        
        # 使用softmax计算自适应权重
        scale_importance = torch.sum(scale_stats, dim=-1)  # [num_scales]
        adaptive_weights = F.softmax(scale_importance * self.text_scale_weights, dim=0)        
        # 应用自适应权重进行特征融合
        weighted_embeds = [w * embed for w, embed in zip(adaptive_weights, multi_scale_embeds)]
        fused_embed = torch.stack(weighted_embeds).sum(dim=0)        
        # 保持原始形状
        fused_embed = fused_embed.unsqueeze(0)
        
        return fused_embed
        # return multi_scale_embeds

    def encode_object_feat(self, feat, img_feat, locs):
        """
        feat : 3D物体的原始特征
        img_feat : 对应的图像特征
        locs : 物体位置信息
        """
        # 特征归一化处理
        feat = torch.nn.functional.normalize(feat, dim=-1)
        img_feat = torch.nn.functional.normalize(img_feat, dim=-1)

        # 截取多尺度信息
        # global_feat = feat[:feat_dim]
        # local_feat = feat[feat_dim:2*feat_dim]
        # texture_feat = feat[2*feat_dim:3*feat_dim]
        global_feat, local_feat, texture_feat = torch.split(feat, 1024, dim=-1)

        # 创建多尺度特征表示
        multi_scale_feats = []
        
        # 第一个尺度：全局语义
        multi_scale_feats.append(global_feat)
        
        # 第二个尺度：局部几何
        multi_scale_feats.append(local_feat)
        
        # 第三个尺度：细粒度纹理
        multi_scale_feats.append(texture_feat)

        # 确保权重数量与特征数量匹配
        weights = self.multi_scale_weights[:len(multi_scale_feats)]
        # 归一化权重
        # norm_weights = F.softmax(weights, dim=0)
        # 融合多尺度特征
        fused_feat = torch.zeros_like(multi_scale_feats[0])
        for i, scale_feat in enumerate(multi_scale_feats):
            fused_feat += weights[i] * scale_feat
        # 返回融合后的特征和原始多尺度特征列表
        # proj_object_embed = fused_feat
        
        # return proj_object_embed, feat, img_feat
        return fused_feat, img_feat

        # 原始单尺度处理
        # return feat, img_feat


    # # 特征一致性损失计算函数
    # def compute_feature_consistency_loss(self, obj_feats, img_feat, scene_mask):
    #     """计算3D物体特征和2D图像特征之间的一致性损失
    #     obj_feats : 3D物体特征(单尺度张量或多尺度列表)
    #     img_feat : 对应的2D图像特征
    #     scene_mask : 标识有效物体的掩码
    #     """
    #     if not self.use_feature_consistency:
    #         device = img_feat.device
    #         return torch.tensor(0.0, device=device)
        
    #     # 如果是多尺度特征，计算每个尺度的一致性损失，并赋予不同权重
    #     if isinstance(obj_feats, list):
    #         loss = 0.0
    #         # 为不同尺度设置不同权重：语义>几何>纹理
    #         scale_weights = [0.5, 0.3, 0.2][:len(obj_feats)]
    #         scale_weights = [w/sum(scale_weights) for w in scale_weights]

    #         # 计算每个尺度的损失
    #         for i, obj_feat in enumerate(obj_feats):
    #             # 只考虑有效的物体
    #             valid_mask = scene_mask.unsqueeze(-1).expand_as(obj_feat)
    #             masked_obj_feat = obj_feat.masked_select(valid_mask).view(-1, obj_feat.size(-1))
    #             masked_img_feat = img_feat.masked_select(valid_mask).view(-1, img_feat.size(-1))
                
    #             # 计算余弦相似度损失
    #             masked_obj_feat = F.normalize(masked_obj_feat, dim=-1)
    #             masked_img_feat = F.normalize(masked_img_feat, dim=-1)
    #             cosine_sim = (masked_obj_feat * masked_img_feat).sum(dim=-1)
    #             # 目标是最大化相似度，所以取负值作为损失
    #             scale_loss = -cosine_sim.mean()
    #             loss += scale_weights[i] * scale_loss
    #         return loss
    #     else:
    #         # 单尺度特征处理
    #         valid_mask = scene_mask.unsqueeze(-1).expand_as(obj_feats)
    #         masked_obj_feat = obj_feats.masked_select(valid_mask).view(-1, obj_feats.size(-1))
    #         masked_img_feat = img_feat.masked_select(valid_mask).view(-1, img_feat.size(-1))
            
    #         # 计算余弦相似度损失
    #         masked_obj_feat = F.normalize(masked_obj_feat, dim=-1)
    #         masked_img_feat = F.normalize(masked_img_feat, dim=-1)
    #         cosine_sim = (masked_obj_feat * masked_img_feat).sum(dim=-1)
    #         return -cosine_sim.mean()

    # 添加空间关系注意力计算函数
    # def compute_spatial_attention(self, locs, scene_mask):
    #     """计算基于3D物体空间位置的注意力权重矩阵
    #     locs : 物体3D坐标
    #     scene_mask : 标识有效物体的掩码
    #     spatial_attention_weight : 控制注意力衰减速度的超参数

    #     通过物体间的3D距离计算注意力权重
    #     距离越近的物体获得越高的注意力权重
    #     形成NxN的注意力矩阵(N为物体数量)

    #     """
    #     if not self.use_spatial_attention:
    #         return None
        
    #     # 获取有效物体的位置
    #     batch_size = locs.shape[0]
    #     spatial_attentions = []

    #     for i in range(batch_size):
    #         valid_mask = scene_mask[i]
    #         valid_locs = locs[i, valid_mask, :3]  # 只取xyz坐标
    #         if valid_locs.shape[0] <= 1:
    #             # 如果只有一个物体，返回单位矩阵
    #             spatial_attn = torch.ones(1, 1, device=locs.device)
    #         else:
    #             # 计算物体间的距离 - 使用更高效的方式
    #             num_objects = valid_locs.shape[0]
    #             if num_objects > 100:  # 如果物体数量太多，采样处理
    #                 indices = torch.randperm(num_objects, device=locs.device)[:100]
    #                 valid_locs = valid_locs[indices]   
    #             # 分块计算距离以减少内存使用
    #             chunk_size = 8  # 根据GPU内存调整
    #             spatial_attn = torch.zeros(valid_locs.shape[0], valid_locs.shape[0], device=locs.device)
    #             for j in range(0, valid_locs.shape[0], chunk_size):
    #                 end_j = min(j + chunk_size, valid_locs.shape[0])
    #                 chunk_j = valid_locs[j:end_j]
    #                 for k in range(0, valid_locs.shape[0], chunk_size):
    #                     end_k = min(k + chunk_size, valid_locs.shape[0])
    #                     chunk_k = valid_locs[k:end_k]
    #                     # 计算这两个块之间的距离
    #                     dist_chunk = torch.cdist(chunk_j, chunk_k, p=2)
    #                     # 将距离转换为注意力权重
    #                     attn_chunk = torch.exp(-dist_chunk / self.spatial_attention_weight)
    #                     spatial_attn[j:end_j, k:end_k] = attn_chunk
    #             # 归一化
    #             spatial_attn = spatial_attn / (spatial_attn.sum(dim=-1, keepdim=True) + 1e-9)
    #         spatial_attentions.append(spatial_attn)
    #     return spatial_attentions
    
    @staticmethod
    def get_dist_attention(pos, dist_exp=1):
        """基于物体3D位置计算距离注意力权重矩阵    
        参数:
            pos: 物体3D坐标张量，形状为[batch_size, num_objects, 3]
            dist_exp: 距离计算指数(默认为1即曼哈顿距离)    
        返回:
            形状为[batch_size, num_objects, num_objects]的注意力矩阵
            其中每个元素a_ij表示物体j对物体i的重要性权重
        """
        # pos (bs, obj_num, 3)
        # 计算成对距离矩阵
        dist = pos.unsqueeze(1) - pos.unsqueeze(2)
        # 计算距离度量（根据dist_exp选择范数）
        dist = torch.sum(dist.abs()**dist_exp, dim=-1)
        # 转换为注意力权重（距离越近权重越高）
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

        # 处理多尺度表示，并构建分层码书
        # if self.use_multi_scale:
        #     # 构建多尺度分层码书
        #     multi_scale_embeds = []            
        #     # 为每个尺度创建标记组
        #     for scale_idx in range(self.num_scales):
        #         # 基础层：对象ID嵌入 (基础属性：3D坐标、颜色)
        #         scale_embed = selected_objid_embeds.clone()
        #         # 添加对应尺度的特征 (多层次标记组：全局语义+局部几何+细粒度纹理)
        #         if not self.no_obj:
        #             # 确保embed_obj[scale_idx]的维度与scale_embed匹配
        #             if isinstance(embed_obj, list):
        #                 # 如果是多尺度特征列表，确保每个尺度的特征维度匹配
        #                 if embed_obj[scale_idx].size(-1) != scale_embed.size(-1):
        #                     # 使用投影层调整维度
        #                     proj = nn.Linear(embed_obj[scale_idx].size(-1), scale_embed.size(-1)).to(scale_embed.device)
        #                     adjusted_embed = proj(embed_obj[scale_idx][assigned_ids])
        #                 else:
        #                     adjusted_embed = embed_obj[scale_idx][assigned_ids]
        #             else:
        #                 # 如果是单尺度特征，直接使用
        #                 adjusted_embed = embed_obj[assigned_ids]
        #             scale_embed += adjusted_embed
        #         # 添加图像特征（根据尺度调整权重）
        #         if self.add_img_token:
        #             img_weight = 0.7 if scale_idx == 0 else 0.3  # 语义层更依赖图像
        #             scale_embed += embed_img[assigned_ids] * img_weight                    
        #         multi_scale_embeds.append(scale_embed)
        #     # 创建最终的分层码书
        #     # 每个对象在码书中占用num_scales个位置，每个位置对应一个尺度
        #     object_list_embed = torch.zeros(
        #         (selected_objid_embeds.shape[0] * self.num_scales, selected_objid_embeds.shape[1]),
        #         dtype=selected_objid_embeds.dtype,
        #         device=selected_objid_embeds.device
        #     )
        #     # 交错排列不同尺度的特征
        #     for scale_idx in range(self.num_scales):
        #         object_list_embed[scale_idx::self.num_scales, :] = multi_scale_embeds[scale_idx]
        #     # print("Multi-scale embedding created.")
        #     return object_list_embed

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
        # print("scene_feat的大小为: ", scene_feat.shape) #torch.Size([8, 100, 1024]) --> torch.Size([8, 100, 3072])
        # proj_object_embed, object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        device = object_embed.device
        batch_size = object_embed.shape[0]
        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = self.object_img_proj(object_img_embed)
        
        # 添加位置编码
        if self.add_pos_emb:
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
            proj_pos_embed = self.pos_proj(pos_embed)
            proj_object_embed = proj_object_embed + proj_pos_embed
            proj_object_img_embed = proj_object_img_embed + proj_pos_embed

        proj_scene_embed = None
        if self.add_scene_token:  # remember to change the evaluate 
            # if self.add_img_token:
            #     object_embed = object_embed + object_img_embed
            obj_embed = self.scene_init_proj(object_embed)
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed)
            scene_embed = obj_embed + pos_embed
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~scene_mask)
            proj_scene_embed = self.scene_proj(scene_embed)

        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0
        p_0_embed = self.p_0_embed.to(device)
        p_1_embed = self.p_1_embed.to(device)
        object_list_intervals = []

        for i, question in enumerate(questions):
            # 构建文本提示
            prompt = f"{question} {self.role[1]}: "
            # 使用多尺度文本处理
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
            to_regress_embed = self.get_text_emb(answer, device=device)[0].squeeze(0)

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

        # 计算空间关系注意力
        # spatial_attentions = self.compute_spatial_attention(scene_locs, scene_mask) if self.use_spatial_attention else None
        
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

                # # 如果启用了空间关系注意力，应用空间关系权重
                # if spatial_attentions is not None:
                #     valid_ids = torch.where(scene_mask[i])[0]
                #     if len(valid_ids) > 1:  # 至少需要两个物体才能应用空间关系
                #         spatial_attn = spatial_attentions[i]
                        
                #         # 计算物体在序列中的位置
                #         obj_positions = []
                #         for j in range(len(valid_ids)):
                #             if self.use_multi_scale and self.num_scales > 1:
                #                 # 多尺度模式下，每个物体占用num_scales个位置
                #                 obj_positions.append(list(range(st + j * self.num_scales, st + (j + 1) * self.num_scales)))
                #             else:
                #                 # 单尺度模式下，每个物体占用一个位置
                #                 obj_positions.append([st + j])
                        
                #         # 应用空间关系注意力
                #         for j in range(len(valid_ids)):
                #             for k in range(len(valid_ids)):
                #                 attn_weight = spatial_attn[j, k]
                #                 for pos_j in obj_positions[j]:
                #                     for pos_k in obj_positions[k]:
                #                         causal_mask[i, :, pos_j, pos_k] = attn_weight

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

        # 清理不需要的中间变量
        del input_embeds, targets, attention_mask
        if 'causal_mask' in locals():
            del causal_mask
        torch.cuda.empty_cache()

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
        # proj_object_embed, object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        device = object_embed.device
        batch_size, obj_num = object_embed.shape[:2]
        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = self.object_img_proj(object_img_embed)
        
        # 添加位置编码
        if self.add_pos_emb:
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
            proj_pos_embed = self.pos_proj(pos_embed)
            # 为每个尺度添加位置嵌入
            proj_object_embed = proj_object_embed + proj_pos_embed
            proj_object_img_embed = proj_object_img_embed + proj_pos_embed

        if self.add_scene_token:
            # if self.add_img_token:
            #     object_embed = object_embed + object_img_embed
            obj_embed = self.scene_init_proj(object_embed)
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed)
            scene_embed = obj_embed + pos_embed
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~scene_mask)
            proj_scene_embed = self.scene_proj(scene_embed)

        output_texts = []
        p_0_embed = self.p_0_embed.to(device).unsqueeze(0)
        p_1_embed = self.p_1_embed.to(device).unsqueeze(0)

        # 计算空间关系注意力
        # spatial_attentions = self.compute_spatial_attention(scene_locs, scene_mask) if self.use_spatial_attention else None
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

                # 如果启用了空间关系注意力，应用空间关系权重
                # if spatial_attentions is not None and self.use_spatial_attention:
                #     valid_ids = torch.where(scene_mask[i])[0]
                #     if len(valid_ids) > 1:  # 至少需要两个物体才能应用空间关系
                #         spatial_attn = spatial_attentions[i]
                        
                #         # 计算物体在序列中的位置
                #         obj_positions = []
                #         for j in range(len(valid_ids)):
                #             if self.use_multi_scale and self.num_scales > 1:
                #                 # 多尺度模式下，每个物体占用num_scales个位置
                #                 obj_positions.append(list(range(st + j * self.num_scales, st + (j + 1) * self.num_scales)))
                #             else:
                #                 # 单尺度模式下，每个物体占用一个位置
                #                 obj_positions.append([st + j])
                        
                #         # 应用空间关系注意力
                #         for j in range(len(valid_ids)):
                #             for k in range(len(valid_ids)):
                #                 attn_weight = spatial_attn[j, k]
                #                 for pos_j in obj_positions[j]:
                #                     for pos_k in obj_positions[k]:
                #                         attention_mask[0, 0, pos_j, pos_k] = attn_weight
            
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
            if torch.cuda.is_bf16_supported() and dtype == torch.bfloat16:
                return torch.cuda.amp.autocast(dtype=torch.bfloat16)
            else:
                return torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device
