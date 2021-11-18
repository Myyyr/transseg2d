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
from einops import repeat, rearrange


class Attention(nn.Module):
    def __init__(self, attn, gt_num=1):
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

        self.gt_num = gt_num

    def forward(self, x, H, W, gt):
        B, N_, C = x.shape
        gt_num = self.gt_num
        # if self.gt_num != 0:
        if len(gt.shape) != 3:
            gt = repeat(gt, "g c -> b g c", b=B)# shape of (num_windows*B, G, C)
        x = torch.cat([gt, x], dim=1)

        print("x", x.shape)
        print("B, N_, C", B,N_,C)
        print("gt", gt.shape)

        exit(0)        
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            if gt_num != 0:
                # print("x", x.shape)
                # print("x[:,gt_num:,:]", x[:,gt_num:,:].shape)
                # exit(0)
                x_ = x[:,gt_num:,:].permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                x_ = torch.cat([gt, x_], dim=1)
            else:
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

        return x[:,gt_num:,:], x[:,:gt_num,:]


class Block(nn.Module):

    def __init__(self, block, gt_num=1):
        super().__init__()
        self.norm1 = block.norm1
        self.attn = Attention(block.attn)
        self.drop_path = block.drop_path
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.gt_num = gt_num

    def forward(self, x, H, W, gt):
        B, N, C = x.shape
        
        skip = x
        skip_gt = gt
        x = self.norm1(x)
        x, gt = self.attn(x, H, W, gt)
        x =self.attn(x, H, W)
        x = skip + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        # if self.gt_num != 0:
        #     if len(skip_gt.shape) != 3:
        #         skip_gt = repeat(gt, "g c -> b g c", b=B)
        # gt = skip_gt + self.drop_path(gt)

        return x, gt






@BACKBONES.register_module()
class SegFormerGTGamma(nn.Module):
    """docstring for SegFormerGTGamma"""
    def __init__(self, gt_num = 10):
        super(SegFormerGTGamma, self).__init__()
        self.gt_num = gt_num
        embed_dims=[64, 128, 320, 512]


        self.global_token1 = torch.nn.Parameter(torch.randn(gt_num,embed_dims[0]))
        # ws_pe = (45*gt_num//2**0, 45*gt_num//2**0)
        # self.pe1 = nn.Parameter(torch.zeros(gt_num, embed_dims[0]))
        # trunc_normal_(self.pe1, std=.02)

        self.global_token2 = torch.nn.Parameter(torch.randn(gt_num,embed_dims[1]))
        # ws_pe = (45*gt_num//2**1, 45*gt_num//2**1)
        # self.pe2 = nn.Parameter(torch.zeros(gt_num, embed_dims[1]))
        # trunc_normal_(self.pe2, std=.02)

        self.global_token3 = torch.nn.Parameter(torch.randn(gt_num,embed_dims[2]))
        # ws_pe = (45*gt_num//2**2, 45*gt_num//2**2)
        # self.pe3 = nn.Parameter(torch.zeros(gt_num, embed_dims[2]))
        # trunc_normal_(self.pe3, std=.02)

        self.global_token4 = torch.nn.Parameter(torch.randn(gt_num,embed_dims[3]))
        # ws_pe = (45*gt_num//2**3, 45*gt_num//2**3)
        # self.pe4 = nn.Parameter(torch.zeros(gt_num, embed_dims[3]))
        # trunc_normal_(self.pe4, std=.02)



    def init_weights(self, pretrained=None):
        mix = mit_b4(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

        if isinstance(pretrained, str):
            mix.init_weights(pretrained)

        depths=[3, 8, 27, 3]

        self.patch_embed1 = mix.patch_embed1
        self.patch_embed2 = mix.patch_embed2
        self.patch_embed3 = mix.patch_embed3 
        self.patch_embed4 = mix.patch_embed4

        # transformer encoder
        self.block1 = nn.ModuleList([Block(mix.block1[i], self.gt_num)
            for i in range(depths[0])])
        self.norm1 = mix.norm1

        self.block2 = nn.ModuleList([Block(mix.block2[i], self.gt_num)
            for i in range(depths[1])])
        self.norm2 = mix.norm2

        self.block3 = nn.ModuleList([Block(mix.block3[i], self.gt_num)
            for i in range(depths[2])])
        self.norm3 = mix.norm3

        self.block4 = nn.ModuleList([Block(mix.block4[i], self.gt_num)
            for i in range(depths[3])])
        self.norm4 = mix.norm4

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        gt = self.global_token1
        for i, blk in enumerate(self.block1):
            x, gt = blk(x, H, W, gt)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        gt = self.global_token2
        for i, blk in enumerate(self.block2):
            x, gt = blk(x, H, W, gt)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        gt = self.global_token3
        for i, blk in enumerate(self.block3):
            x, gt = blk(x, H, W, gt)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        gt = self.global_token4
        for i, blk in enumerate(self.block4):
            x, gt = blk(x, H, W, gt)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x