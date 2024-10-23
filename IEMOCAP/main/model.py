import torch
import torch.nn as nn
import random
from utils.vanillatransformer1 import vanilla_transformer_block
import torch.nn.functional as F
from utils.modules import PositionalEncoding
from utils.DWFormerBlock import DWFormerBlock

# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         # 修改卷积层的输入通道为 in_planes=1，适应形状 (32, 1, 324, 1024)
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
#                               stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         # 批量归一化用于卷积后的特征
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
#                                  momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None

#     def forward(self, x):
#         # 将输入 (32, 324, 1024) 转换为 (32, 1, 324, 1024)
       
#         # 应用卷积操作
#         x = self.conv(x)
#         # 如果使用批归一化则进行处理
#         if self.bn is not None:
#             x = self.bn(x)
#         # 应用 ReLU 激活函数
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

class DynamicConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1):
        super(DynamicConv, self).__init__()
        self.dynamic_kernel = nn.Parameter(torch.randn(out_dim, in_dim, kernel_size))
        self.stride = stride
        self.padding = (kernel_size - 1) // 2  # 确保输出长度不变
 
    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        # 添加 padding 参数来保持长度
        conv_out = F.conv1d(x.permute(0, 2, 1), self.dynamic_kernel, stride=self.stride, padding=self.padding)
        return conv_out.permute(0, 2, 1)  # 恢复原始形状

class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape#[32, 324, 1024]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)#([32, 8, 324, 324])
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x 


class classifier(nn.Module):
    def __init__(self, feadim, classnum):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(feadim, feadim // 2)
        self.fc2 = nn.Linear(feadim // 2, feadim // 4)
        self.fc3 = nn.Linear(feadim // 4, classnum)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = x.permute([0,2,1])
        x = self.avgpool(x).squeeze(-1)
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        x = self.dropout(self.relu(x))
        out = self.fc3(x)
        return out


class DWFormer(nn.Module):
    def __init__(self, feadim, n_head, FFNdim, classnum):#1024,8,512,4
        super(DWFormer, self).__init__() 
        '''
        feadim:input dimension of the feature
        n_head:numbers of the attention head
        FFNdim:dimension of FeedForward Network
        classnum: numbers of emotion
        '''
        self.or1 = vanilla_transformer_block(feadim, n_head, FFNdim)
        self.dt1 = DWFormerBlock(feadim, n_head, FFNdim)
        self.dt2 = DWFormerBlock(feadim, n_head, FFNdim)
        self.dt3 = DWFormerBlock(feadim, n_head, FFNdim)
        self.classifier = classifier(feadim, classnum)
        self.PE = PositionalEncoding(feadim)
        self.ln1 = nn.LayerNorm(feadim,eps = 1e-5)
        self.ln2 = nn.LayerNorm(feadim,eps = 1e-5)
        self.ln3 = nn.LayerNorm(feadim,eps = 1e-5)
        self.ln4 = nn.LayerNorm(feadim,eps = 1e-5)
        # 添加 EfficientAttention 实例
        self.efficient_attention = EfficientAttention(dim=1024)
        # 添加 DynamicConv 模块
        self.dynamic_conv = DynamicConv(in_dim=feadim, out_dim=feadim, kernel_size=3)
        #  # 添加 BasicConv 模块
        # self.basic_conv = BasicConv(in_planes=1, out_planes=feadim, kernel_size=(3, 3), stride=1, padding=1)
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, x_mask):
        '''
        x:input data, (b, t, c)
        x_mask:
        '''
        
        batch, times, _ = x.shape
        haltingscore = torch.zeros((batch, times), device=x.device)

        # propose random attention scores instead of vanilla transformer
        # randomdata = self.getNumList(0,323,120)
        # haltingscore[:,randomdata] += 1e-10
        # Vanilla Transformer Block
        
        x = self.ln1(x)
        # 对输入数据进行动态卷积处理
        x = self.dynamic_conv(x)
        # # 使用 BasicConv 进行卷积特征提取
        # x = x.unsqueeze(1)  # 改变形状为 (batch_size, 1, time_steps, feature_dim)
        # x = self.basic_conv(x)  # 卷积操作
        # x = x.squeeze(1)  # 恢复形状为 (batch_size, time_steps, feature_dim)

        x1,_,attn = self.or1(x, haltingscore)

        # DWFormer Block

        x2,thresholds1,attn11 = self.dt1(x1, x_mask, attn)
        x3 = self.ln2(x2) #[32, 324, 1024]
        x4,thresholds2,attn12 = self.dt2(x3, x_mask, attn11)
        x5 = self.ln3(x4)
        x6, thresholds3, attn13 = self.dt3(x5, x_mask, attn12)


        # 使用 EfficientAttention
        x6 = self.efficient_attention(x6, x_mask.shape[1], x_mask.shape[1])
        # Classifier
        out = self.classifier(x6)#32,4
        
        

        return out,attn,attn11,attn12,attn13
    
        # attn4 = torch.cat([attn.unsqueeze(0).data,attn11.unsqueeze(0).data,attn12.unsqueeze(0).data,attn13.unsqueeze(0)],dim = 0)#ori结果
        # thresholds = torch.cat([thresholds1.unsqueeze(0).data,thresholds2.unsqueeze(0).data,thresholds3.unsqueeze(0)],dim = 0)#分窗
        # return out,attn4,thresholds


        
    
    def getNuthresholdsist(self,start, end, n):
        '''
        generate random init
        '''
        numsArray = set()
        while len(numsArray) < n:
            numsArray.add(random.randint(start, end))

        return list(numsArray)
