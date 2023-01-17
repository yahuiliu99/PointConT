'''
Date: 2022-02-20 07:55:10
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-02-21 02:48:19
'''

import torch
import torch.nn as nn

class MLPBlockFC(nn.Module):
    def __init__(self, d_points, d_model, p_dropout):
        super(MLPBlockFC, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(d_points, d_model, bias=False),
                                 nn.BatchNorm1d(d_model),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Dropout(p=p_dropout))

    def forward(self, x):
        return self.mlp(x)


class MLPBlock2D(nn.Module):
    def __init__(self, d_points, d_model):
        super(MLPBlock2D, self).__init__()
        self.mlp = nn.Sequential(nn.Conv2d(d_points, d_model, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(d_model),
                                 nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        return self.mlp(x)


class MLPBlock1D(nn.Module):
    def __init__(self, d_points, d_model):
        super(MLPBlock1D, self).__init__()
        self.mlp = nn.Sequential(nn.Conv1d(d_points, d_model, kernel_size=1, bias=False),
                                 nn.BatchNorm1d(d_model),
                                 nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        return self.mlp(x)
        

class ResMLPBlock1D(nn.Module):
    def __init__(self, d_points, d_model):
        super(ResMLPBlock1D, self).__init__()
        self.mlp1 = nn.Sequential(nn.Conv1d(d_points, d_model, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(d_model),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.mlp2 = nn.Sequential(nn.Conv1d(d_model, d_points, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(d_points))
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.mlp2(self.mlp1(x)) + x)
