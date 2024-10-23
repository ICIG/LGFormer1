import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from einops import rearrange
import time
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import random
from fairseq.modules.multihead_attention import MultiheadAttention
from .modules import FeedForwardNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        初始化高效注意力层 (Efficient Attention) 适用于语音数据

        参数:
        dim: 输入特征的维度 (C)，即每个位置的特征向量维度。
        num_heads: 注意力机制中的多头数量，默认是8。
        qkv_bias: 是否在查询 (query), 键 (key), 值 (value) 计算中使用偏置，默认是False。
        qk_scale: 查询和键的缩放因子，若未指定，则使用每个头的特征维度的倒数平方根作为缩放因子。
        attn_drop: 注意力得分的dropout概率，防止过拟合，默认是0。
        proj_drop: 输出的dropout概率，默认是0。
        """
        super(EfficientAttention, self).__init__()
        assert dim % num_heads == 0, f"输入维度 {dim} 应该能被多头数量 {num_heads} 整除。"

        self.dim = dim  # 输入特征维度
        self.num_heads = num_heads  # 多头数量
        head_dim = dim // num_heads  # 每个头的特征维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子

        # 查询 (query) 的线性映射层，将输入x映射到查询空间
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # 键 (key) 和 值 (value) 的线性映射层，将输入x映射到键和值空间，输出维度为2倍的输入维度
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)  # 注意力得分的Dropout，用于在注意力计算过程中随机丢弃部分权重，防止过拟合
        self.proj = nn.Linear(dim, dim)  # 输出的线性映射层，将多头注意力的结果映射回输入维度
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的Dropout，用于防止过拟合

    def forward(self, x):
        """
        前向传播函数

        参数:
        x: 输入特征，形状为 (batch_size, seq_len, dim)，即 (B, N, C)。

        返回:
        x: 输出特征，形状为 (batch_size, seq_len, dim)。
        attn: 注意力矩阵，形状为 (batch_size, seq_len, seq_len)。
        """
        B, N, C = x.shape  # 获取输入的批量大小、序列长度和特征维度

        # 计算查询 (q)
        # q的形状为 (batch_size, num_heads, seq_len, head_dim)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 计算键 (k) 和 值 (v)
        # kv的形状为 (2, batch_size, num_heads, seq_len, head_dim)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # k 和 v 的形状均为 (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力得分，公式为: attn = (q @ k^T) * scale
        # q @ k.transpose(-2, -1) 的形状为 (batch_size, num_heads, seq_len, seq_len)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 通过softmax对注意力得分进行归一化，得到注意力权重
        attn = torch.softmax(attn, dim=-1)

        # 对注意力得分进行Dropout处理，防止过拟合
        attn = self.attn_drop(attn)

        # 计算注意力加权后的输出，公式为: output = attn @ v
        # attn @ v 的形状为 (batch_size, num_heads, seq_len, head_dim)
        x = torch.matmul(attn, v)

        # 调整输出张量的维度，恢复为原始输入的形状 (batch_size, seq_len, dim)
        x = x.transpose(1, 2).reshape(B, N, C)

        # 通过输出投影层进行线性变换，得到最终输出
        x = self.proj(x)

        # 最终输出的Dropout处理，防止过拟合
        x = self.proj_drop(x)

        # 为了返回 attn，先将其从 (batch_size, num_heads, seq_len, seq_len) 形状转换为 (batch_size, seq_len, seq_len)
        # 这里通过对所有头的注意力矩阵取平均来实现
        attn = attn.mean(dim=1)  # 形状为 (batch_size, seq_len, seq_len)

        return x, attn  # 返回输出和注意力矩阵



class LocalMHA(nn.Module):
    def __init__(self, dim, head):
        super(LocalMHA, self).__init__()
        assert dim % head == 0
        self.softmax = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.head = head
        self.mha = EfficientAttention(dim=dim, num_heads=head)  # 替换为 EfficientAttention
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask, mappingmask):
        '''
        x: 输入数据，形状 (t, batch, c)
        mask: 注意力mask, 形状 (batch, t, t)
        mappingmask: 用于计算重要性分数的mask, 形状 (batch, t)
        '''
        # 重新调整 mask 的形状
        mask1 = mask.unsqueeze(1)
        mask1 = mask1.expand(-1, self.head, -1, -1)
        mask1 = rearrange(mask1, 'b h t1 t2 -> (b h) t1 t2')
        mask2 = torch.ones_like(mappingmask)
        mask3 = (mask2 - mappingmask) * (-1e10)

        # 使用 EfficientAttention 计算注意力
        attn_output, attn_output_weights = self.mha(x)

        # 计算重要性分数
        haltingscore = torch.sum(attn_output_weights, dim=1).unsqueeze(1)
        #haltingscore shape: torch.Size([324, 1, 32])
        #mappingmask shape: torch.Size([32, 58, 324])    
        haltingscore = haltingscore.permute(2, 1, 0)
        
        haltingscore = haltingscore * mappingmask
        haltingscore = self.softmax2(haltingscore + mask3) * mappingmask
        haltingscore = torch.where(torch.isnan(haltingscore), torch.zeros_like(haltingscore), haltingscore)
        haltingscore = torch.sum(haltingscore, dim=1)

        return attn_output, haltingscore

class DynamicLocalWindowTransformer(nn.Module):
    def __init__(self, dim, head, FFNdim) -> None:
        super(DynamicLocalWindowTransformer, self).__init__()
        self.MHSA = LocalMHA(dim, head)  # 替换后的 LocalMHA
        self.FFN = FeedForwardNetwork(dim, FFNdim)
        self.ln1 = nn.LayerNorm(dim, eps=1e-5)
        self.ln2 = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x, mask, mappingmask):
        '''
        Dynamic Local Window Transformer 模块
        x: 输入数据，形状 (batch, t, c)
        mask: 注意力mask, 形状 (batch, t, t)
        padding mask: mask 用于遮蔽 padding 部分, 形状 (batch, t)
        '''
        x = x.permute([1, 0, 2])
        residual = x
        x1, attn = self.MHSA(x, mask, mappingmask)
        x = residual + x1
        x = self.ln1(x)
        residual = x
        x2 = self.FFN(x)
        x = self.ln2(residual + x2)
        x = x.permute([1, 0, 2])
        return x, attn

# 假设 FeedForwardNetwork 的定义如下
class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, ffn_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_dim, dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


