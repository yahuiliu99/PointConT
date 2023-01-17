'''
Date: 2022-03-11 11:01:07
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-07-13 10:01:25
'''

import math
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util import critical_feature_sample
from pointnet2_ops import pointnet2_utils

from pointnet_util import farthest_point_sample, index_points, square_distance
from .ResMLP import ResMLPBlock1D
from .transformer import SinusoidalPositionalEmbedding


def Point2Patch(num_patches, patch_size, xyz):
    """
    Patch Partition in 3D Space
    Input:
        num_patches: number of patches, S
        patch_size: number of points per patch, k
        xyz: input points position data, [B, N, 3]
    Return:
        centroid: patch centroid, [B, S, 3]
        knn_idx: [B, S, k]
    """
    # FPS the patch centroid out
    fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_patches).long()  # [B, S]
    centroid_xyz = index_points(xyz, fps_idx)    # [B, S, 3]
    # knn to group per patch
    dists = square_distance(centroid_xyz, xyz)  # [B, S, N]
    knn_idx = dists.argsort()[:, :, :patch_size]  # [B, S, k]
    
    return centroid_xyz, fps_idx, knn_idx


def PatchMerger(num_patches, patch_size, xyz, feature):
    # CFS the patch centroid out
    cfs_idx = critical_feature_sample(feature, num_patches) # [B, S]
    centroid_xyz = index_points(xyz, cfs_idx)    # [B, S, 3]
    # knn to group per patch
    dists = square_distance(centroid_xyz, xyz)  # [B, S, N]
    knn_idx = dists.argsort()[:, :, :patch_size]  # [B, S, k]
    
    return centroid_xyz, cfs_idx, knn_idx



class PatchAbstraction(nn.Module):
    def __init__(self, num_patches, patch_size, in_channel, mlp):
        super(PatchAbstraction, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp_act = nn.ModuleList()
        self.mlp_res = ResMLPBlock1D(mlp[-1], mlp[-1])

        last_channel = in_channel 
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            self.mlp_act.append(nn.ReLU(inplace=True))
            last_channel = out_channel

    def forward(self, xyz, feature):
        """
        Input: xyz [B, S_, 3]
               features [B, S_, C]
        Return: [B, S, 3+D]
        """
        B, _, C = feature.shape
        centroid_xyz, centroid_idx, knn_idx = Point2Patch(self.num_patches, self.patch_size, xyz)
        
        centroid_feature = index_points(feature, centroid_idx) # [B, S, C]
        grouped_feature = index_points(feature, knn_idx)    # [B, S, k, C]

        k = grouped_feature.shape[2]

        # Normalize                                                                                                                                                                                                                                                            
        grouped_norm = grouped_feature - centroid_feature.view(B, self.num_patches, 1, C) # [B, S, k, C]
        groups = torch.cat((centroid_feature.unsqueeze(2).expand(B, self.num_patches, k, C), grouped_norm), dim=-1) # [B, S, k, 2C]
        
        groups = groups.permute(0, 3, 2, 1) # [B, Channel, k, S]
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            act = self.mlp_act[i]
            groups =  act(bn(conv(groups))) # [B, D, k, S]

        max_patches = torch.max(groups, 2)[0] # [B, D, S]
        max_patches = self.mlp_res(max_patches).transpose(1, 2) # [B, S, D]

        avg_patches = torch.mean(groups, 2).transpose(1, 2) # [B, S, D]
        
        return centroid_xyz, max_patches, avg_patches


class PatchFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PatchFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp_act = nn.ModuleList()
        self.mlp_res = ResMLPBlock1D(mlp[-1], mlp[-1])

        last_channel = in_channel 
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            self.mlp_act.append(nn.ReLU(inplace=True))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, C]
            xyz2: sampled input points position data, [B, S, C]
            points1: input points data, [B, N, D1]
            points2: sampled input points data, [B, S, D2]
        Return:
            new_points: upsampled points data, [B, N, D']
        """

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)  # [B, N, D1+D2]
        else:
            new_points = interpolated_points

        new_points = new_points.transpose(1, 2)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            act = self.mlp_act[i]
            new_points =  act(bn(conv(new_points))) 

        new_points = self.mlp_res(new_points).transpose(1, 2) # [B, S, D]
        
        return new_points



class FSLA(nn.Module):
    '''
    Feature Space Local Attention
    Args:
        dim (int): Number of input channels.
        local_size (int): The size of the local feature space.
        over_size (int): The size of the local overlapping.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    '''

    def __init__(self, dim, local_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., kmeans = False):

        super().__init__()
        self.dim = dim
        self.ls = local_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kmeans = kmeans

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        '''
        Input: [B, S, D]
        Return: [B, S, D]
        '''

        B, S, D = x.shape
        nl = S // self.ls
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4) # [3, B, h, S, d]

        q_pre = qkv[0].reshape(B*self.num_heads, S, D // self.num_heads).permute(0,2,1) # [B*h, d, S]
        ntimes = int(math.log(nl, 2))
        q_idx_last = torch.arange(S).cuda().unsqueeze(0).expand(B*self.num_heads, S)
        
        # balanced binary clustering
        for _ in range(ntimes):
            bh,d,n = q_pre.shape # [B*h*2^n, d, S/2^n]
            q_pre_new = q_pre.reshape(bh, d, 2, n//2) # [B*h*2^n, d, 2, S/2^n]
            q_avg = q_pre_new.mean(dim=-1) # [B*h*2^n, d, 2]

            q_avg = torch.nn.functional.normalize(q_avg.permute(0,2,1), dim=-1) 
            q_norm = torch.nn.functional.normalize(q_pre.permute(0,2,1), dim=-1)
  
            q_scores = square_distance(q_norm, q_avg) # [B*h*2^n, S/2^n, 2]
            q_ratio = (q_scores[:,:,0]+1) / (q_scores[:,:,1]+1) # [B*h*2^n, S/2^n]
            q_idx = q_ratio.argsort()
            
            q_idx_last = q_idx_last.gather(dim=-1, index=q_idx).reshape(bh*2, n//2) # [B*h*2^n, S/2^n]
            q_idx_new = q_idx.unsqueeze(1).expand(q_pre.size()) # [B*h*2^n, d, S/2^n]
            q_pre_new = q_pre.gather(dim=-1, index=q_idx_new).reshape(bh, d, 2, n//2) # [B*h*2^n, d, 2, S/(2^(n+1))]
            q_pre = rearrange(q_pre_new, 'b d c n -> (b c) d n')   # [B*h*2^(n+1), d, S/(2^(n+1))]

        # clustering is performed independently in each head
        q_idx = q_idx_last.view(B,self.num_heads, S) # [B, h, S]
        q_idx_rev = q_idx.argsort() # [B, h, S]

        # cluster query, key, value 
        q_idx = q_idx.unsqueeze(0).unsqueeze(4).expand(qkv.size()) # [3, B, h, S, d]
        qkv_pre = qkv.gather(dim=-2, index=q_idx) # [3, B, h, S, d]
        q, k, v  = rearrange(qkv_pre, 'qkv b h (nl ls) d -> qkv (b nl) h ls d', ls=self.ls)

        # MSA
        attn = (q - k)*self.scale
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        out =  torch.einsum('bhld, bhld->bhld', attn, v) # [B*(nl), h, ls, d]

        # merge and reverse
        out = rearrange(out, '(b nl) h ls d -> b h d (nl ls)', h=self.num_heads, b=B) # [B, h, d, S]
        q_idx_rev = q_idx_rev.unsqueeze(2).expand(out.size())
        res = out.gather(dim=-1,index=q_idx_rev).reshape(B,D,S).permute(0,2,1) # [B, S, D]

        res = self.proj(res) # [B, S, D]
        res = self.proj_drop(res)

        res = x + res # [B, S, D]
        
        return res



class NAT(nn.Module):
    def __init__(self, dim, local_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., with_pos=True):
        super().__init__()
        self.dim = dim
        self.ls = local_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_pos = with_pos

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        '''
        Input: [B, S, D]
        Return: [B, S, D]
        '''
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, D) # [B, S, 3, D]

        q, k, v = rearrange(qkv, 'b s qkv (h d) -> qkv (b h) s d', h=self.num_heads) # [B*h, S, d]
        
        dists = square_distance(q, q)
        q_idx = dists.argsort()[:, :, :self.ls]  # [B*h, S, ls]

        k, v = index_points(k, q_idx), index_points(v, q_idx) # [B*h, S, ls, d]

        # neighboring is performed independently in each head
        q = rearrange(q, '(b h) s d -> (b s) h d', h=self.num_heads).unsqueeze(2) # [B*S, h, 1, d]
        k = rearrange(k, '(b h) s ls d -> (b s) h ls d', h=self.num_heads) # [B*S, h, ls, d]
        v = rearrange(v, '(b h) s ls d -> (b s) h ls d', h=self.num_heads) # [B*S, h, ls, d]
        
        # MSA
        attn = (q @ k.transpose(-2, -1))*self.scale
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        out = attn @ v  # [B*S, h, 1, d]

        res = rearrange(out, '(b s) h l d -> b (s l) (h d)', s=S) # [B, S, D]
        
        res = self.proj(res) # [B, S, D]
        res = self.proj_drop(res)

        res = x + res # [B, S, D]

        return res



class PatchTransformer(nn.Module):
    def __init__(self, patch_dim, trans_dim, point2patch=False):
        super().__init__()
        self.scale = trans_dim ** -0.5
        self.point2patch = point2patch
        self.input_layer = nn.Linear(patch_dim, trans_dim)
        self.w_qs = nn.Linear(trans_dim, trans_dim, bias=False)
        self.w_ks = nn.Linear(trans_dim, trans_dim, bias=False)
        self.w_vs = nn.Linear(trans_dim, trans_dim, bias=False)
        self.fc_gamma = nn.Sequential(
                nn.Linear(trans_dim, trans_dim),
                nn.GELU(),
                nn.Linear(trans_dim, trans_dim)
            )
        self.output_layer = nn.Linear(trans_dim, patch_dim)
        if self.point2patch:
            self.pos_embed = nn.Sequential(
                nn.Linear(10, 128),
                nn.GELU(),
                nn.Linear(128, trans_dim)
            )
            
    def forward(self, x):
        '''
        Input:  
            if point2patch: [B, S, 3+D]
            else: [B, S, D]
        Return: [B, S, D]
        '''
        if self.point2patch:
            centroid = x[:, :, :3]
            pre = x[:, :, 3:]
            
            B, S, D = centroid.shape
            k_patch = min(S, 8)
            patch_dist = square_distance(centroid, centroid)  # [B, S, S]
            patch_idx = patch_dist.argsort()[:, :, :k_patch]  # [B, S, kp]
            knn_patch = index_points(centroid, patch_idx) # [B, S, kp, 3]
            ref_vectors = knn_patch - centroid.unsqueeze(2)  # [B, S, kp, 3]
            pos_struct = torch.cat([centroid.unsqueeze(2).expand(B, S, k_patch, D), 
                                    knn_patch, ref_vectors,
                                    torch.linalg.norm(ref_vectors, dim=-1, keepdim=True)], dim=-1)  # [B, S, kp, 10]
            pos_emb = torch.max(self.pos_embed(pos_struct), dim=2)[0] # [B, S, T]

            x = self.input_layer(pre) + pos_emb  # [B, S, T]

        else:
            pre = x
            x = self.input_layer(pre) # [B, S, T]

        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
            
        attn = self.fc_gamma(q - k) * self.scale    # [B, S, T]
        attn = F.softmax(attn, dim=-1)
        res = torch.multiply(attn, v)
        res = self.output_layer(res) + pre  # [B, S, D]

        return res



class TransformerBlock(nn.Module):
    def __init__(self, patch_dim, trans_dim, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(patch_dim, trans_dim)
        self.fc2 = nn.Linear(trans_dim, patch_dim)
        self.fc_delta = nn.Sequential(
            nn.Linear(10, trans_dim),
            nn.ReLU(),
            nn.Linear(trans_dim, trans_dim)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(trans_dim, trans_dim),
            nn.ReLU(),
            nn.Linear(trans_dim, trans_dim)
        )
        self.w_qs = nn.Linear(trans_dim, trans_dim, bias=False)
        self.w_ks = nn.Linear(trans_dim, trans_dim, bias=False)
        self.w_vs = nn.Linear(trans_dim, trans_dim, bias=False)
        self.k = k
          
    def forward(self, x):
        '''
        Patch Transformer in Feature Space
        Input: [B, S, 3+D]
        Return: [B, S, D]
        '''
        pos = x[:, :, :3]
        features = x[:, :, 3:]

        dists = square_distance(features, features)
        idx = dists.argsort()[:, :, :self.k]  # b x n x k
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), idx), index_points(self.w_vs(x), idx)

        relation = q[:, :, None] - k 
        attn = self.fc_gamma(relation) # b x n x k x f
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v) # b x n x f
        res = self.fc2(res) + pre   # b x n x f
        
        return pos, res



class PointACMix(nn.Module):
    def __init__(self, num_patches, patch_size, in_channel, out_channel):
        super(PointACMix, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.to_qkv = nn.Conv2d(in_channel, out_channel * 3, 1)

        self.scale = out_channel ** -0.5
        self.fc_gamma = nn.Sequential(
                nn.Linear(out_channel, out_channel),
                nn.GELU(),
                nn.Linear(out_channel, out_channel)
            )
        self.output_layer = nn.Linear(out_channel, out_channel)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, out_channel)
        )
        
    def forward(self, x):
        centroid, groups = Point2Patch(self.num_patches, self.patch_size, x)
            
        pos_emb = self.pos_embed(centroid)

        groups = groups.permute(0, 3, 2, 1) # [B, Channel, k, S]
        q, k, v = self.to_qkv(groups).chunk(3, dim=1)  # [B, D, k, S]
        q = torch.max(q, 2)[0].transpose(-1, -2) # [B, S, D]
        k = torch.max(k, 2)[0].transpose(-1, -2) # [B, S, D]
        v = torch.max(v, 2)[0].transpose(-1, -2) # [B, S, D]

        attn = self.fc_gamma(q - k) * self.scale    # [B, S, D]
        attn = F.softmax(attn, dim=-1)
        res = torch.multiply(attn, v)
        res = self.output_layer(res) + pos_emb  # [B, S, D]
    
        return res  



class PatchTransformerMSG(nn.Module):
    def __init__(self, patch_num, patch_dim, trans_dim):
        super(PatchTransformerMSG, self).__init__()
        self.scale = trans_dim ** -0.5
        self.input_layer = nn.Linear(patch_dim, trans_dim)
        self.pm = PatchMerger(trans_dim, patch_num)
        self.w_qs = nn.Linear(trans_dim, trans_dim, bias=False)
        self.w_ks = nn.Linear(trans_dim, trans_dim, bias=False)
        self.w_vs = nn.Linear(trans_dim, trans_dim, bias=False)
        self.fc_gamma = nn.Sequential(
                nn.Linear(trans_dim, trans_dim),
                nn.GELU(),
                nn.Linear(trans_dim, trans_dim)
            )
        self.output_layer = nn.Linear(trans_dim, patch_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim)
        )

    def forward(self, idx, q, x):
        '''
        Input:  
            q: [B, S_, D_]
            x: if point2patch: [B, S, 3+D]
               else: [B, S, D]
        Return: [B, S, D]
        '''

        centroid = x[:, :, :3]
        pre = x[:, :, 3:]

        pos_emb = self.pos_embed(centroid) # [B, S, T]
        x = self.input_layer(pre) + pos_emb # [B, S, T]

        if not idx:
            q = self.w_qs(x)
        else:
            q = self.pm(q)

        k, v = self.w_ks(x), self.w_vs(x)
               
        attn = self.fc_gamma(q - k) * self.scale    # [B, S, T]
        attn = F.softmax(attn, dim=-1)
        res = torch.multiply(attn, v)
        res = self.output_layer(res) + pre  # [B, S, D]

        return q, res 



class PointTransformer(nn.Module):
    def __init__(self, num_patches, patch_size, in_channel, mlp):
        super(PointTransformer, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.mlp_alpha = nn.Sequential(
                nn.Conv1d(in_channel, mlp[0], 1),
                nn.BatchNorm1d(mlp[0]),
                nn.GELU()
            )
        self.scale = mlp[1] ** -0.5
        self.w_qs = nn.Conv1d(mlp[0], mlp[1], 1)
        self.w_ks = nn.Conv1d(mlp[0], mlp[1], 1)
        self.w_vs = nn.Conv1d(mlp[0], mlp[1], 1)
        self.mlp_gamma = nn.Sequential(
                nn.Conv1d(mlp[1], mlp[1], 1),
                nn.GELU(),
                nn.Conv1d(mlp[1], mlp[1], 1)
            )

    def forward(self, x):
        """
        Input: [B, N, 3]
        Return: [B, S, 3+D]

        """
        centroid, groups = Point2Patch(self.num_patches, self.patch_size, x)
        groups = rearrange(groups, 'b s k c -> (b s) c k')
        groups = self.mlp_alpha(groups) # [B*S, D, k]
        
        q, k, v = self.w_qs(groups), self.w_ks(groups), self.w_vs(groups)
        sim = torch.einsum('b d i, b d j -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        res = torch.einsum('b i j, b d j -> b d i', attn, v)
        res = self.mlp_gamma(res + groups) + res
        res = rearrange(res, '(b s) d k -> b s d k', s=self.num_patches)    # [B, S, D, k]
        patches = torch.max(res, dim=-1)[0]    # [B, S, D]
        patches = torch.cat([centroid, patches], dim=-1) # [B, S, 3+D]
        
        return patches



class GeometricEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super(GeometricEmbedding, self).__init__()
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.mlp_d = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_a = nn.Linear(hidden_dim, hidden_dim)

    def geometric_struct(self, centroid):
        B, S, C = centroid.shape
        k_patch = min(S, 8)
        patch_dist = square_distance(centroid, centroid)  # [B, S, S]
        d_indices = patch_dist 

        patch_idx = patch_dist.argsort()[:, :, :k_patch]  # [B, S, kp]
        knn_patch = index_points(centroid, patch_idx) # [B, S, kp, 3]
        ref_vectors = knn_patch - centroid.unsqueeze(2)  # (B, S, kp, 3)
        anc_vectors = centroid.unsqueeze(1) - centroid.unsqueeze(2)  # (B, S, S, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(B, S, S, k_patch, C)  # (B, S, S, kp, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(B, S, S, k_patch, C)  # (B, S, S, kp, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, S, S, kp)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, S, S, kp)
        patch_angle = torch.atan2(sin_values, cos_values)  # (B, S, S, kp)
        a_indices = patch_angle 

        return d_indices, a_indices

    def forward(self, centroid):
        d_indices, a_indices = self.geometric_struct(centroid)
        d_emb = self.embedding(d_indices)
        d_emb = self.mlp_d(d_emb)

        a_emb = self.embedding(a_indices)
        a_emb = self.mlp_a(a_emb) 
        a_emb = torch.max(a_emb, dim=3)[0]

        emb = d_emb + a_emb

        return emb



def Point2PatchMSG(num_patches, patch_size_list, xyz):
    """
    Input:
        num_patches: number of patches, S
        patch_size_list: number of points per patch, [k1, k2]
        xyz: input points position data, [B, N, 3]
    Return:
        centroid: patch centroid, [B, S, 3]
        grouped_xyz_list: [[B, S, k1, 3], [B, S, k2, 3]]
    """
    B, N, C = xyz.shape
    S = num_patches
    # FPS the patch centroid out
    fps_idx = farthest_point_sample(xyz, num_patches) # [B, S]
    centroid = index_points(xyz, fps_idx)    # [B, S, 3]
    # kNN to group per patch
    dists = square_distance(centroid, xyz)  # [B, S, N]
    grouped_xyz_list = []
    for i in range(len(patch_size_list)):
        idx = dists.argsort()[:, :, :patch_size_list[i]]  # [B, S, k]
        grouped_xyz = index_points(xyz, idx) # [B, S, k, 3]
        # Normalize                                                                                                                                                                                                                                                            
        grouped_xyz_norm = grouped_xyz - centroid.view(B, S, 1, C) # [B, S, k, 3]
        grouped_xyz_list.append(grouped_xyz_norm)
    
    return centroid, grouped_xyz_list



class PatchAbstractionMSG(nn.Module):
    def __init__(self, num_patches, patch_size_list, in_channel, mlp):
        super(PatchAbstractionMSG, self).__init__()
        self.num_patches = num_patches
        self.patch_size_list = patch_size_list
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp_res = ResMLPBlock1D(mlp[-1], mlp[-1])

        last_channel = in_channel 
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        """
        Input: [B, N, 3]
        Return: [B, S, 3+ns*D], where ns is the length of patch_size_list
        """
        centroid, groups = Point2PatchMSG(self.num_patches, self.patch_size_list, x)
        patches_list = []
        for group  in groups:   
            group = group.permute(0, 3, 2, 1) # [B, Channel, k/G, S]
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                group =  F.relu(bn(conv(group))) # [B, D, k/G, S]

            patches = torch.max(group, 2)[0] # [B, D, S]
            patches = self.mlp_res(patches).transpose(1, 2) # [B, S, D]
            patches_list.append(patches)

        new_patches = torch.cat(patches_list, dim=-1) # [B, S, ns*D]
        new_patches = torch.cat((centroid, new_patches), dim=-1) # [B, S, 3+ns*D]
        
        return new_patches
