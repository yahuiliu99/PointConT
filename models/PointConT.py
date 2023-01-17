'''
Date: 2022-03-12 11:47:58
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-07-13 14:05:49
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .PointConT_util import PatchAbstraction, PatchFeaturePropagation, FSLA, NAT
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
            self.patch_transformer.append(FSLA(cfg.patch_dim[i+1], cfg.local_size[i], cfg.num_heads))
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



class PointConT_partseg(nn.Module):
    def __init__(self, cfg, seg_num_all):
        super().__init__()
        self.backbone = Backbone(cfg)
        self.embedding = PatchAbstraction(cfg.num_points, cfg.patch_size[0], 
                                          2*cfg.input_dim, [cfg.en_dim[0], cfg.en_dim[0]])
                                                           
        # feature propagation
        self.en_blocks = len(cfg.en_dim)
        self.fp_blocks = len(cfg.de_dim)
        self.patch_feat_prop = nn.ModuleList()
        self.patch_transformer = nn.ModuleList()

        for i in range(-1, -self.fp_blocks-1, -1):
            self.patch_feat_prop.append(PatchFeaturePropagation(cfg.en_dim[i]+cfg.en_dim[i-1], [cfg.de_dim[i], cfg.de_dim[i]]))
            self.patch_transformer.append(FSLA(cfg.de_dim[i], cfg.local_size[i], cfg.num_heads))
        

        # class label mapping
        self.cls_map = nn.Sequential(
            MLPBlock1D(16, cfg.cls_token_dim),
            MLPBlock1D(cfg.cls_token_dim, cfg.cls_token_dim)
        )

        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for i in range(self.en_blocks):
            self.gmp_map_list.append(MLPBlock1D(cfg.en_dim[i], cfg.gmp_dim))
        self.gmp_map_end = MLPBlock1D(cfg.gmp_dim*self.en_blocks, cfg.gmp_dim)
        
        # classifier
        self.mlp1 = MLPBlock1D(cfg.de_dim[0]+cfg.gmp_dim+cfg.cls_token_dim, 128)
        self.mlp2 = nn.Conv1d(128, seg_num_all, kernel_size=1)
        
        
    def forward(self, x, cls_label):
        B, N, _ = x.shape
        _, feat0, _ = self.embedding(x, x)

        _, pos_and_feats = self.backbone(x) 
        pos_and_feats[0][1] = feat0

        fp_feat = pos_and_feats[-1][1]
        for i in range(self.fp_blocks):
            fp_feat = self.patch_feat_prop[i](pos_and_feats[-i-2][0], pos_and_feats[-i-1][0], 
                                              pos_and_feats[-i-2][1], fp_feat)
            fp_feat = self.patch_transformer[i](fp_feat)
        
        gmp_list = []
        for i in range(self.en_blocks):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](pos_and_feats[i][1].permute(0,2,1)), 1))

        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)).repeat(1,1,N) # [B, gmp_dim, N]       
        cls_token = self.cls_map(cls_label.view(B,16,1)).repeat(1,1,N) # [B, cls_token_dim, N]

        res = torch.cat([fp_feat.transpose(1, 2), global_context, cls_token], dim=1)
        res = self.mlp2(self.mlp1(res)) # [B, seg_num_all, N]

        return res



class PointConT_semseg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        self.embedding = MLPBlock1D(cfg.input_dim, cfg.en_dim[0])

        # feature propagation
        self.en_blocks = len(cfg.en_dim)
        self.fp_blocks = len(cfg.de_dim)
        self.patch_feat_prop = nn.ModuleList()
        self.patch_transformer = nn.ModuleList()

        for i in range(-1, -self.fp_blocks-1, -1):
            self.patch_feat_prop.append(PatchFeaturePropagation(cfg.en_dim[i]+cfg.en_dim[i-1], [cfg.de_dim[i], cfg.de_dim[i]]))
            self.patch_transformer.append(FSLA(cfg.de_dim[i], cfg.local_size[i], cfg.over_size[i], cfg.num_heads))
        
        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for i in range(self.en_blocks):
            self.gmp_map_list.append(MLPBlock1D(cfg.en_dim[i], cfg.gmp_dim))
        self.gmp_map_end = MLPBlock1D(cfg.gmp_dim*self.en_blocks, cfg.gmp_dim)
        
        # classifier
        self.mlp1 = MLPBlock1D(cfg.de_dim[0]+cfg.gmp_dim, 128)
        self.mlp2 = nn.Conv1d(128, cfg.num_classes, kernel_size=1)
        
        
    def forward(self, x):
        B, N, _ = x.shape
        feat0 = self.embedding(x.permute(0,2,1)).permute(0,2,1)

        _, pos_and_feats = self.backbone(x) 
        pos_and_feats[0][1] = feat0

        fp_feat = pos_and_feats[-1][1]
        for i in range(self.fp_blocks):
            fp_feat = self.patch_feat_prop[i](pos_and_feats[-i-2][0], pos_and_feats[-i-1][0], 
                                              pos_and_feats[-i-2][1], fp_feat)
            fp_feat = self.patch_transformer[i](fp_feat)
        
        gmp_list = []
        for i in range(self.en_blocks):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](pos_and_feats[i][1].permute(0,2,1)), 1))

        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)).repeat(1,1,N) # [B, gmp_dim, N]       
        
        res = torch.cat([fp_feat.transpose(1, 2), global_context], dim=1)
        res = self.mlp2(self.mlp1(res)) # [B, seg_num, N]

        return res