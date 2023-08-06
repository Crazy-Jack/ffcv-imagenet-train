
import torch.nn as nn 
import torch

import importlib


def find_topk_operation_using_name(model_name):
    """Import the module "".
    
    """
    model_filename = "pytorch_pretrained_vit.topk_layer"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, nn.Module):
            model = cls

    if model is None:
        print("In %s.py, there should be a class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

class TopkLayer(nn.Module):
    """implement topk layer for transformer"""
    def __init__(self, topk):
        super(TopkLayer, self).__init__()
        self.topk = topk 
    
    def forward(self, x, v=False):
        """
        x: [b, gh*gw+1, d]
        """
        if v:
            print(f"topk {self.topk} x.shape {x.shape}")
        n, hw1, d = x.shape
        if self.topk >= 1.:
            return x
        topk_keep_num = max(1, int(hw1 * self.topk))

        x_t = torch.transpose(x, 1, 2) # n, d, hw1
        _, index = torch.topk(x_t.abs(), topk_keep_num, dim=2)
        mask = torch.zeros_like(x_t).scatter_(2, index, 1).to(x.device)
        mask = torch.transpose(mask, 1, 2) # n, hw1, d
        # print(f"mask {mask.mean().item()}")
        sparse_x = mask * x
        
        return sparse_x




class TopkLayerNoCLStopk(nn.Module):
    """implement topk layer for transformer"""
    def __init__(self, topk):
        super(TopkLayerNoCLStopk, self).__init__()
        self.topk = topk 
    
    def forward(self, x, v=False):
        """
        x: [b, gh*gw+1, d]
        """
        if v:
            print(f"topk {self.topk} x.shape {x.shape}")
        
        if self.topk >= 1.:
            return x
    
        x_cls, x_others = x[:, :1], x[:, 1:]
        n, hw, d = x_others.shape
        
        topk_keep_num = max(1, int(hw * self.topk))

        x_t = torch.transpose(x_others, 1, 2) # n, d, hw
        _, index = torch.topk(x_t.abs(), topk_keep_num, dim=2)
        mask = torch.zeros_like(x_t).scatter_(2, index, 1).to(x_others.device)
        mask = torch.transpose(mask, 1, 2) # n, hw, d

        sparse_x = mask * x_others

        x = torch.cat([x_cls, sparse_x], dim=1)
        
        return x