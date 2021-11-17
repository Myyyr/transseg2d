# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
import math


from .mix_transformer import mit_b4



class Attention(nn.Module):
    def __init__(self, attn):
        super().__init__()

        self.dim = attn.dim
        self.num_heads = attn.num_heads
        self.scale = attn.scale

        self.q = attn.q
        self.kv = attn.kv
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

        self.sr_ratio = attn.sr_ratio
        if self.sr_ratio > 1:
            self.sr = attn.sr
            self.norm = attn.norm

    def forward(self, x, H, W):
        B, N, C = x.shape
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
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, block):
        super().__init__()
        self.norm1 = block.norm1
        self.attn = Attention(block.attn)
        self.drop_path = block.drop_path
        self.norm2 = block.norm2
        self.mlp = block.mlp

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x






@BACKBONES.register_module()
class SegFormerGTGamma(nn.Module):
    """docstring for SegFormerGTGamma"""
    def __init__(self, gt_num = 1, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1):
        super(SegFormerGTGamma, self).__init__()
        self.mix = mit_b4(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)



    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.mix.init_weights(pretrained)

        depths=[3, 8, 27, 3]

        self.patch_embed1 = self.mix.patch_embed1
        self.patch_embed2 = self.mix.patch_embed2
        self.patch_embed3 = self.mix.patch_embed3 
        self.patch_embed4 = self.mix.patch_embed4

        # transformer encoder
        self.block1 = nn.ModuleList([Block(self.mix.block1[i])
            for i in range(depths[0])])
        self.norm1 = self.mix.norm1

        self.block2 = nn.ModuleList([Block(self.mix.block2[i])
            for i in range(depths[1])])
        self.norm2 = self.mix.norm2

        self.block3 = nn.ModuleList([Block(self.mix.block3[i])
            for i in range(depths[2])])
        self.norm3 = self.mix.norm3

        self.block4 = nn.ModuleList([Block(self.mix.block4[i])
            for i in range(depths[3])])
        self.norm4 = self.mix.norm4

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x