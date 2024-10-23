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

class LocalMHA(nn.Module):
    def __init__(self, dim, head):
        super(LocalMHA, self).__init__()
        assert dim % head == 0
        self.softmax = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.head = head
        self.mha = MultiheadAttention(embed_dim=dim, num_heads=head)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, mask,mappingmask):
        '''
        x:input data, (t, batch, c)

        attn mask: attention mask, (batch,t, t)

        padding mask: mask the padding portion, (batch t)
        '''
        #calculate multi-head attention
        mask1 = mask.unsqueeze(1)
        mask1 = mask1.expand(-1,self.head,-1,-1)
        mask1 = rearrange(mask1,'b h t1 t2 -> (b h) t1 t2')
        mask2 = torch.ones_like(mappingmask)
        mask3 = (mask2- mappingmask) * (-1e10)
        attn_output, attn_output_weights = self.mha(query=x, key=x, value=x,attn_mask=mask1.bool())#attn_output_weights(32,324,324)
        
        #calculate importance scores
        haltingscore = torch.sum(attn_output_weights, dim=1).unsqueeze(1)#(b*n,t)
        #haltingscore shape: torch.Size([32, 1, 324])
        #mappingmask shape: torch.Size([32, 63, 324])
        haltingscore = haltingscore * mappingmask#mappingmask(b,t)
        haltingscore = self.softmax2(haltingscore + mask3) * mappingmask
        haltingscore = torch.where(torch.isnan(haltingscore),torch.zeros_like(haltingscore),haltingscore)
        haltingscore = torch.sum(haltingscore,dim = 1)
        return attn_output, haltingscore#attn_output 的形状为 (t, b, c)。altingscore 的形状为 (batch_size * num_heads, 1, t)

class DynamicLocalWindowTransformer(nn.Module):
    def __init__(self, dim, head, FFNdim) -> None:
        super(DynamicLocalWindowTransformer, self).__init__()
        self.MHSA = LocalMHA(dim, head)
        self.FFN = FeedForwardNetwork(dim, FFNdim)
        self.ln1 = nn.LayerNorm(dim, eps=1e-5)
        self.ln2 = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x, mask,mappingmask):
        '''
        Dyanmic Local Window Transformer (DLWT) modules

        x:input data, (batch, t, c)

        attn mask: attention mask, (batch, t, t)

        padding mask: mask the padding portion, (batch, t)
        '''
        x = x.permute([1,0,2])
        residual = x
        x1, attn = self.MHSA(x, mask,mappingmask)
        x = residual + x1
        x = self.ln1(x)
        residual = x
        x2 = self.FFN(x)
        x = self.ln2(residual + x2)
        x = x.permute([1,0,2])
        return x, attn