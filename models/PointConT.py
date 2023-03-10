'''
Date: 2022-03-12 11:47:58
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-07-13 14:05:49
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .PointConT_util import PatchAbstraction, ConT
from .ResMLP import MLPBlock1D, MLPBlockFC


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nblocks = len(cfg.patch_dim) - 1
        self.patch_abstraction = nn.ModuleList()
        self.patch_transformer = nn.ModuleList()
        self.patch_embedding = nn.ModuleList()
        for i in range(self.nblocks):
            self.patch_abstraction.append(PatchAbstraction(int(cfg.num_points/cfg.down_ratio[i]), 
                                                           cfg.patch_size[i], 
                                                           2*cfg.patch_dim[i], 
                                                           [cfg.patch_dim[i+1], cfg.patch_dim[i+1]]))
            self.patch_transformer.append(ConT(cfg.patch_dim[i+1], cfg.local_size[i], cfg.num_heads))
            self.patch_embedding.append(MLPBlock1D(cfg.patch_dim[i+1]*2, cfg.patch_dim[i+1]))

    def forward(self, x):
        if x.shape[-1] == 3:
            pos = x
        else:
            pos = x[:, :, :3].contiguous()
        features = x
        pos_and_feats = []
        pos_and_feats.append([pos, features])

        for i in range(self.nblocks):
            pos, max_features, avg_features = self.patch_abstraction[i](pos, features)
            avg_features = self.patch_transformer[i](avg_features)
            features = torch.cat([max_features, avg_features], dim=-1)
            features = self.patch_embedding[i](features.transpose(1, 2)).transpose(1, 2)
            pos_and_feats.append([pos, features])

        return features, pos_and_feats



class PointConT_cls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        self.mlp1 = MLPBlockFC(cfg.patch_dim[-1], 512, cfg.dropout)
        self.mlp2 = MLPBlockFC(512, 256, cfg.dropout)
        self.output_layer = nn.Linear(256, cfg.num_classes)
        
    def forward(self, x):
        patches, _ = self.backbone(x)  # [B, num_patches[-1], patch_dim[-1]]
        res = torch.max(patches, dim=1)[0]  # [B, patch_dim[-1]]
        res = self.mlp2(self.mlp1(res))
        res = self.output_layer(res) 

        return res


