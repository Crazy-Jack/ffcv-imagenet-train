from copy import copy
from torchvision import models
import torch
import torch.nn as nn 
import numpy as np 
from torch.nn.utils import spectral_norm
import math 

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def sparse_ch(x, topk):
    
    n, c, _, _ = x.shape
    x = x.reshape(n, c)
    
    topk_keep_num = int(max(1, topk * c))
    _, index = torch.topk(x, topk_keep_num, dim=1)
    mask = torch.zeros_like(x).scatter_(1, index, 1)
    x = x * mask
    x = x.unsqueeze(-1).unsqueeze(-1)
    return x

class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)
    


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.ModuleList([ nn.AdaptiveAvgPool2d(4), nn.Sigmoid(),
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), nn.ReLU(), #Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid()])
        
    def forward(self, x, v=False):
        # if v:
        #     print(f"check 0 {(torch.isnan(x) * 1.).mean()}")
        # weight = self.main[0](x)
        # if v: print(f"check 1 {(torch.isnan(weight) * 1.).mean()}")
        # weight = self.main[1](weight)
        # if v: print(f"check 2 {(torch.isnan(weight) * 1.).mean()}")
        # weight = self.main[2](weight)
        # if v: print(f"check 3 {(torch.isnan(weight) * 1.).mean()}")
        # weight = self.main[3](weight)
        # if v: print(f"check 4 {(torch.isnan(weight) * 1.).mean()}")
        weight = x
        for module in self.main:
            weight = module(weight)

        weight = sparse_ch(weight, 0.2)
        return x * weight

class TopKLayer(nn.Module):
    def __init__(self, topk=0.1, revert=False, topk_tau=0., topk_decay_method='', topk_decay_ramp=1e+5, permutate=0, take_se_channel=0, ch_in=-1, activation=""):
        super(TopKLayer, self).__init__()
        self.revert=revert
        self.topk=1.
        self.target_topk = topk
        self.topk_decay_method = topk_decay_method 
        self.topk_decay_ramp = topk_decay_ramp
        self.activation = activation
        # print(f"topk_decay_ramp {topk_decay_ramp}")
        if self.topk_decay_method != "":
            self.register_buffer('topk_decay_clock', torch.Tensor([0.]))

        self.topk_tau = topk_tau
        self.permutate = permutate
        self.take_se_channel = take_se_channel and (ch_in > 0)
        if self.take_se_channel:
            self.senet = SEBlock(ch_in, ch_in)
        
        if self.topk_tau > 1: self.topk_tau = 1.
        if self.topk_tau < 0: self.topk_tau = 0.
        """
        the BigGAN teaser is achieved via the following file: 
        /lab_data/leelab/tianqinl/BigGAN-PyTorch/scripts/1percent/launch_BigGAN_bs64_ch64_mirrorE_sparse_spread.sh
        --sparsity_resolution 8_16_32_64 --sparsity_ratio 1_1_1_1 \
        tau = min(iter_num * self.sparse_decay_rate, 1) -> basically 
            sparse_x = mask * x_reshape
            sparsity_x = 1.0 - torch.where(sparse_x == 0.0)[0].shape[0] / (n * c * h * w)
            print("sparsity -- ({}): {}".format((n, c, h, w), sparsity_x)) ## around 9% decrease to 4% fired eventually this way
            if tau == 1.0:
                return sparse_x.view(n, c, h, w)
            
            # print("--- tau", tau)
            tau_x = x * torch.FloatTensor([1. - tau]).to(device)
            # print("sum of x used", tau_x.sum())
            return sparse_x.view(n, c, h, w) * torch.FloatTensor([tau]).to(device) + tau_x
        So basically 95% original activation and 5% topk activation, so you emphasis the topk
        this translate to topk_tau -> 0.95
        """
        
    def topk_decay_step(self):
        if self.training and self.topk_decay_method:
            self.topk_decay_clock += 1

        if not self.topk_decay_method:
            self.topk = max(0, min(1., self.target_topk))
        
        elif self.topk_decay_method == 'exp':
            self.topk = max(self.target_topk, self.topk * 0.99)
        
        elif self.topk_decay_method == 'cosine':
            eta_min = max(0, min(1., self.target_topk))
            self.topk = eta_min + (1. - eta_min) * (
                1 + math.cos(math.pi * min(1, self.topk_decay_clock / self.topk_decay_ramp))) / 2
            # print(f"current self.topk {self.topk}")
    

    def get_current_tau(self, tau):
        if tau is None:
            tau = self.topk_tau
        else:
            # control tau 
            if tau < 0: 
                tau = 0.
            if tau > 1.:
                tau = 1.
        
        return tau
    

    def sparse_hw(self, x, topk, tau=None):
        tau = self.get_current_tau(tau)
        # self.prev_x = x.abs().mean()

        n, c, h, w = x.shape
        topk_keep_num = int(max(1, topk * h * w))
        if topk == 1 or tau == 1. or (topk_keep_num == h * w):
            return x
        x_reshape = x.view(n, c, h * w)
        
        _, index = torch.topk(x_reshape.abs(), topk_keep_num, dim=2)
        if self.revert:
            assert self.training
            mask = (torch.ones_like(x_reshape) - torch.zeros_like(x_reshape).scatter_(2, index, 1))
        else:
            mask = torch.zeros_like(x_reshape).scatter_(2, index, 1)
        # print("mask percent: ", mask.mean().item())
        sparse_x = mask * x_reshape

        # print("sum of x used", tau_x.sum())
        sparse_x = sparse_x.view(n, c, h, w)
        self.sparse_x = sparse_x
        # combine with tau - linear interpolation
        if tau == 0:
            return sparse_x
        elif tau > 0:
            
            return torch.ones_like(x) * tau * x + torch.ones_like(sparse_x) * (1 - tau) * sparse_x

    def permutate_non_topk(self, x, topk):
        n, c, h, w = x.shape
        topk_keep_num = int(max(1, topk * h * w))
        if topk == 1 or (topk_keep_num == h * w):
            return x
        
        x_reshape = x.view(n, c, h * w)
        # permutate the non-zero entry of the activation
        non_topk_value, non_topk_index = torch.topk(-x_reshape.abs(), x_reshape.shape[2] - topk_keep_num, dim=2)
        non_topk_value_shape = non_topk_value.shape
        if self.training: # permutate if training
            non_topk_value_ = non_topk_value[:, :, torch.randperm(non_topk_value_shape[2])]
            non_topk_value = non_topk_value_ * 0.5 + non_topk_value * 0.5
        # reconstruct the non topk back
        non_topk_permutate_reshape = torch.zeros_like(x_reshape).scatter_(2, non_topk_index, non_topk_value)       
        non_topk_permutate = non_topk_permutate_reshape.reshape(n, c, h, w)
        
        return non_topk_permutate

    def take_se_channel_func(self, x):
        n, c, h, w = x.shape
        # x = x.mean(1).unsqueeze(1).repeat(1, c, 1, 1)
        # print(f"senet")
        x = self.senet(x)
        return x
    
    def forward(self, x):
        if not self.training:
            self.original_x = copy(x)
        sparse_x = self.sparse_hw(x, self.topk)
        if not self.training:
            self.sparse_x = sparse_x
        return sparse_x

        self.topk_decay_step()
        # print(f"self.topk {self.topk}")
        if self.activation == "x3":
            n, c, h, w = x.shape
            temp = 10.
            # max_x = x.max()
            # min_x = x.min()
            x = torch.softmax(x.reshape(n, c, h * w) / temp, -1).reshape(n, c, h, w)
            # # normalize it?
            # x = torch.clamp(x, min=min_x.item(), max=max_x.item())
            # pass
            
        topk_x = self.sparse_hw(x, self.topk, tau=None)

        if self.take_se_channel:
            topk_x = self.take_se_channel_func(topk_x)


        if self.permutate > 0:
            assert self.revert == False
            # with torch.no_grad():
            perm_non_topk_x = self.permutate_non_topk(x, self.topk)
            return topk_x + perm_non_topk_x # currently using random as inference 
        else:
            return topk_x
    


def alexnet_5layer(topk, pretrained=True, topk_tau=0., permutate=0, **kwags): #take_mean_channel
    alexnet = models.alexnet(pretrained=pretrained)
    # resnet50 = torch.hub.load("pytorch/vision", "alexnet", weights="IMAGENET1K_V2")
    new_features = nn.Sequential(
        # layers up to the point of insertion
        *(list(alexnet.features.children())[:3]), 
        TopKLayer(topk, topk_tau=topk_tau, permutate=permutate, ch_in=64, **kwags),
        *(list(alexnet.features.children())[3:6]),
        TopKLayer(topk, topk_tau=topk_tau, permutate=permutate, ch_in=192, **kwags),
        *(list(alexnet.features.children())[6:8]),
        TopKLayer(topk, topk_tau=topk_tau, permutate=permutate, ch_in=384, **kwags),
        *(list(alexnet.features.children())[8:10]),
        TopKLayer(topk, topk_tau=topk_tau, permutate=permutate, ch_in=256, **kwags),
        *(list(alexnet.features.children())[10:]),
        TopKLayer(topk, topk_tau=topk_tau, permutate=permutate, ch_in=256, **kwags),
    )
    alexnet.features = new_features
    model = alexnet
    return model

def alexnet_2layer(topk, pretrained=True, topk_tau=0., permutate=0):
    alexnet = models.alexnet(pretrained=pretrained)
    new_features = nn.Sequential(
        # layers up to the point of insertion
        *(list(alexnet.features.children())[:3]),
        *(list(alexnet.features.children())[3:6]),
        TopKLayer(topk, topk_tau=topk_tau, permutate=permutate),
        *(list(alexnet.features.children())[6:8]),
        TopKLayer(topk, topk_tau=topk_tau, permutate=permutate),
        *(list(alexnet.features.children())[8:10]),
        *(list(alexnet.features.children())[10:]),
    )
    alexnet.features = new_features
    model = alexnet
    return model


def alexnet_1layer(topk, pretrained=True, topk_tau=0., permutate=0):
    alexnet = models.alexnet(pretrained=pretrained)
    new_features = nn.Sequential(
        # layers up to the point of insertion
        *(list(alexnet.features.children())[:3]),
        *(list(alexnet.features.children())[3:6]),
        *(list(alexnet.features.children())[6:8]),
        TopKLayer(topk, topk_tau=topk_tau, permutate=permutate),
        *(list(alexnet.features.children())[8:10]),
        *(list(alexnet.features.children())[10:]),
    )
    alexnet.features = new_features
    model = alexnet
    return model 


# ensemble
class EnsembleAlexNet5layerTopK(nn.Module):
    def __init__(self, topk, feature_dim=6*6*256, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super(EnsembleAlexNet5layerTopK, self).__init__()
    
        self.topk = topk
        alexnet_topk = models.alexnet(pretrained=True)
        self.feature_topk = torch.nn.Sequential(
            # layers up to the point of insertion
            *(list(alexnet_topk.features.children())[:3]), 
            TopKLayer(topk),
            *(list(alexnet_topk.features.children())[3:6]),
            TopKLayer(topk),
            *(list(alexnet_topk.features.children())[6:8]),
            TopKLayer(topk),
            *(list(alexnet_topk.features.children())[8:10]),
            TopKLayer(topk),
            *(list(alexnet_topk.features.children())[10:]),
            TopKLayer(topk),
            alexnet_topk.avgpool,
        )
        alexnet_normal = models.alexnet(pretrained=True)
        self.feature_normal = torch.nn.Sequential(
            alexnet_normal.features,
            alexnet_normal.avgpool
        )
        self.head = torch.nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        bz, c, h, w = x.shape
        
        x1 = self.feature_topk(x)
        x1 = torch.flatten(x1, 1)
        x2 = self.feature_normal(x)
        x2 = torch.flatten(x2, 1)
        x = torch.cat([x1, x2], dim=1) # bz, feature_dim * 2
        x = self.head(x)
        return x

def topK_VGG_5layers(topk, topk_tau=0., pretrained=False, permutate=0.):
    vgg16 = models.vgg16(pretrained=pretrained)
    new_features = nn.Sequential(
        # layers up to the point of insertion
        *(list(vgg16.features.children())[:5]), # 4 is MaxPool2d
        TopKLayer(topk, topk_tau=topk_tau, permutate=permutate),
        *(list(vgg16.features.children())[5:10]),
        TopKLayer(topk, topk_tau=topk_tau, permutate=permutate),
        *(list(vgg16.features.children())[10:17]),
        TopKLayer(topk, topk_tau=topk_tau, permutate=permutate),
        *(list(vgg16.features.children())[17:24]),
        TopKLayer(topk, topk_tau=topk_tau, permutate=permutate),
        *(list(vgg16.features.children())[24:]),
        TopKLayer(topk, topk_tau=topk_tau, permutate=permutate),
    )
    vgg16.features = new_features
    return vgg16



def topK_resnet50(topk, topk_tau=0., pretrained=False, permutate=0., **kwags):
    # resnet50 = models.resnet50(pretrained=pretrained)
    resnet50 = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    for name, module in resnet50.named_children():
        if name in ['layer1', 'layer2', 'layer3','layer4']:
            new_module = nn.Sequential(
                module,
                TopKLayer(topk, **kwags),
            )
            setattr(resnet50, name, new_module)
    print("Using resnet50 4topk layers")
    model = resnet50
    return model 

def topK_resnet50_1layer(topk, topk_tau=0., pretrained=False, permutate=0., **kwags):
    resnet50 = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    for name, module in resnet50.named_children():
        if name in ['layer1']:
            new_module = nn.Sequential(
                module,
                TopKLayer(topk, **kwags),
            )
            setattr(resnet50, name, new_module)
    print("Using resnet50 1topk layers")
    model = resnet50
    return model 

def topK_resnet50_2layers(topk, topk_tau=0., pretrained=False, permutate=0., **kwags):
    resnet50 = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    for name, module in resnet50.named_children():
        if name in ['layer1', 'layer2']:
            new_module = nn.Sequential(
                module,
                TopKLayer(topk, **kwags),
            )
            setattr(resnet50, name, new_module)
    print("Using resnet50 1topk layers")
    model = resnet50
    return model 

def topK_resnet18(topk, topk_tau=0., pretrained=False, permutate=0., **kwags):
    resnet18 = models.resnet18(pretrained=False)
    for name, module in resnet18.named_children():
        if name in ['layer1', 'layer2', 'layer3','layer4']:
            new_module = nn.Sequential(
                module,
                TopKLayer(topk),
            )
            setattr(resnet18, name, new_module)
    print("Using resnet18 4topk layers")
    model = resnet18
    return model 
