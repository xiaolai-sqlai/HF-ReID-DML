from __future__ import absolute_import

import math
import random
import copy
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange
from functools import reduce

from network.layer import BatchDrop, BatchErasing
from network.crcnn import CRCNN


class GlobalAvgPool2d(nn.Module):
    def __init__(self, p=1):
        super(GlobalAvgPool2d, self).__init__()
        self.p = p
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = x.pow(self.p)
        out = self.gap(out)
        return out.pow(1/self.p)
    
    def __repr__(self):
        s = f"{self.__class__.__name__}(p={self.p})"
        return s


class GlobalMaxPool2d(nn.Module):
    def __init__(self, p=1):
        super(GlobalMaxPool2d, self).__init__()
        self.p = p
        self.gap = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        out = x.pow(self.p)
        out = self.gap(out)
        return out.pow(1/self.p)

    def __repr__(self):
        s = f"{self.__class__.__name__}(p={self.p})"
        return s


class HFNet(nn.Module):
    def __init__(self, num_classes=751, std=0.1, net="small_v1", decoder="max-1_avg-1", erasing=0.0):
        super(HFNet, self).__init__()
        if self.training:
            self.erasing = nn.Identity()
            if erasing > 0:
                self.erasing = BatchErasing(smax=erasing)

        embed_dim, mlp_ratio, layers, group_width = net.split("_")
        embed_dim = int(embed_dim.split("-")[1])
        mlp_ratio = float(mlp_ratio.split("-")[1])
        layers = [int(l) for l in layers.split("-")[1:]]
        group_width = int(group_width.split("-")[1])

        model = CRCNN(embed_dim=embed_dim, mlp_ratio=mlp_ratio, layers=layers, group_width=group_width)
        self.feat_num = embed_dim * 4
        path = "pretrain/{}.pth".format(net)
        old_checkpoint = torch.load(path)["state_dict"]
        new_checkpoint = dict()
        for key in old_checkpoint.keys():
            if key.startswith("module."):
                new_checkpoint[key[7:]] = old_checkpoint[key]
            else:
                new_checkpoint[key] = old_checkpoint[key]
        model.load_state_dict(new_checkpoint, strict=False)


        self.stem = model.stem
        self.layer1 = model.stage1
        self.layer2 = model.stage2
        self.layer3 = model.stage3

        self.layer3.layers[0].main_local.conv.stride = (1, 1)
        self.layer3.layers[0].skip[0].conv.stride = (1, 1)

        self.pool_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.class_list = nn.ModuleList()

        self.pool_types = decoder.split("_")
        for d in self.pool_types:
            pool, p = d.split("-")
            if pool == "avg":
                self.pool_list.append(GlobalAvgPool2d(p=int(p)))
            elif pool == "max":
                self.pool_list.append(GlobalMaxPool2d(p=int(p)))

            bn = nn.BatchNorm1d(self.feat_num)
            if pool == "max":
                bn.bias.requires_grad = False
            init.normal_(bn.weight, mean=1.0, std=std)
            init.normal_(bn.bias, mean=0.0, std=std)
            self.bn_list.append(bn)

            linear = nn.Linear(self.feat_num, num_classes, bias=False)
            init.normal_(linear.weight, std=0.001)
            self.class_list.append(linear)

    def forward(self, x):
        if self.training:
            x = self.erasing(x)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        pool_list = []
        bn_list = []
        class_list = []

        for i in range(len(self.pool_types)):
            pool = self.pool_list[i](x).flatten(1)
            pool_list.append(pool)
            bn = self.bn_list[i](pool)
            bn_list.append(bn)
            feat_class = self.class_list[i](bn)
            class_list.append(feat_class)

        if self.training:
            return class_list, bn_list
        return bn_list

