from copy import copy
from torchvision import models
import torch
import torch.nn as nn

class TopKLayer(nn.Module):
    def __init__(self, topk=0.1, revert=False):
        super(TopKLayer, self).__init__()
        self.revert=revert
        self.topk=topk

    def sparse_hw(self, x, topk):
        n, c, h, w = x.shape
        if topk == 1:
            return x
        x_reshape = x.view(n, c, h * w)
        topk_keep_num = int(max(1, topk * h * w))
        _, index = torch.topk(x_reshape.abs(), topk_keep_num, dim=2)
        if self.revert:
            # Useless
            mask = (torch.ones_like(x_reshape) - torch.zeros_like(x_reshape).scatter_(2, index, 1))
        else:
            mask = torch.zeros_like(x_reshape).scatter_(2, index, 1)
        # print("mask percent: ", mask.mean().item())
        sparse_x = mask * x_reshape

        # print("sum of x used", tau_x.sum())
        return sparse_x.view(n, c, h, w)

    def forward(self, x):
        if not self.training:
            self.original_x = copy(x)
        sparse_x = self.sparse_hw(x, self.topk)
        if not self.training:
            self.sparse_x = sparse_x
        return sparse_x


def alexnet_5layer(topk, pretrained=True):
    alexnet = models.alexnet(pretrained=pretrained)
    new_features = nn.Sequential(
        # layers up to the point of insertion
        *(list(alexnet.features.children())[:3]),
        TopKLayer(topk),
        *(list(alexnet.features.children())[3:6]),
        TopKLayer(topk),
        *(list(alexnet.features.children())[6:8]),
        TopKLayer(topk),
        *(list(alexnet.features.children())[8:10]),
        TopKLayer(topk),
        *(list(alexnet.features.children())[10:]),
        TopKLayer(topk),
    )
    alexnet.features = new_features
    model = alexnet
    return model

def alexnet_2layer(topk, pretrained=True):
    alexnet = models.alexnet(pretrained=pretrained)
    new_features = nn.Sequential(
        # layers up to the point of insertion
        *(list(alexnet.features.children())[:3]),
        *(list(alexnet.features.children())[3:6]),
        TopKLayer(topk),
        *(list(alexnet.features.children())[6:8]),
        TopKLayer(topk),
        *(list(alexnet.features.children())[8:10]),
        *(list(alexnet.features.children())[10:]),
    )
    alexnet.features = new_features
    model = alexnet
    return model


def alexnet_1layer(topk, pretrained=True):
    alexnet = models.alexnet(pretrained=pretrained)
    new_features = nn.Sequential(
        # layers up to the point of insertion
        *(list(alexnet.features.children())[:3]),
        *(list(alexnet.features.children())[3:6]),
        *(list(alexnet.features.children())[6:8]),
        TopKLayer(topk),
        *(list(alexnet.features.children())[8:10]),
        *(list(alexnet.features.children())[10:]),
    )
    alexnet.features = new_features
    model = alexnet
    return model
