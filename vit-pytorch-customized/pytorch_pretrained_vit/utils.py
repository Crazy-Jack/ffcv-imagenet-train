"""utils.py - Helper functions
"""

import numpy as np
import torch
from torch.utils import model_zoo
import torch.nn as nn 
from .configs import PRETRAINED_MODELS


def load_pretrained_weights(
    model, 
    model_name=None, 
    weights_path=None, 
    load_first_conv=True, 
    load_fc=True, 
    load_repr_layer=False,
    resize_positional_embedding=False,
    verbose=True,
    strict=True,
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_first_conv (bool): Whether to load patch embedding.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        resize_positional_embedding=False,
        verbose (bool): Whether to print on completion
    """
    assert bool(model_name) ^ bool(weights_path), 'Expected exactly one of model_name or weights_path'
    
    # Load or download weights
    if weights_path is None:
        url = PRETRAINED_MODELS[model_name]['url']
        if url:
            state_dict = model_zoo.load_url(url)
        else:
            raise ValueError(f'Pretrained model for {model_name} has not yet been released')
    else:
        state_dict = torch.load(weights_path)

    # Modifications to load partial state dict
    expected_missing_keys = []
    if not load_first_conv and 'patch_embedding.weight' in state_dict:
        expected_missing_keys += ['patch_embedding.weight', 'patch_embedding.bias']
    if not load_fc and 'fc.weight' in state_dict:
        expected_missing_keys += ['fc.weight', 'fc.bias']
    if not load_repr_layer and 'pre_logits.weight' in state_dict:
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    for key in expected_missing_keys:
        state_dict.pop(key)

    # Change size of positional embeddings
    if resize_positional_embedding: 
        posemb = state_dict['positional_embedding.pos_embedding']
        posemb_new = model.state_dict()['positional_embedding.pos_embedding']
        state_dict['positional_embedding.pos_embedding'] = \
            resize_positional_embedding_(posemb=posemb, posemb_new=posemb_new, 
                has_class_token=hasattr(model, 'class_token'))
        maybe_print('Resized positional embeddings from {} to {}'.format(
                    posemb.shape, posemb_new.shape), verbose)

    # Load state dict
    ret = model.load_state_dict(state_dict, strict=False)
    if strict:
        assert set(ret.missing_keys) == set(expected_missing_keys), \
            'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
        assert not ret.unexpected_keys, \
            'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)
        maybe_print('Loaded pretrained weights.', verbose)
    else:
        maybe_print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys), verbose)
        maybe_print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys), verbose)
        return ret


def maybe_print(s: str, flag: bool):
    if flag:
        print(s)


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def resize_positional_embedding_(posemb, posemb_new, has_class_token=True):
    """Rescale the grid of position embeddings in a sensible manner
    posemb: [1, old_seq_len, dim]
    posemb_new: [1, new_seq_len, dim]
    """
    from scipy.ndimage import zoom

    # Deal with class token
    ntok_new = posemb_new.shape[1]
    if has_class_token:  # this means classifier == 'token'
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

    # Get old and new grid sizes
    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    # Rescale grid
    zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)

    # Deal with class token and return
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def resize_positional_embedding_torch(posemb, ntok_new, has_class_token=True):
    """Rescale the grid of position embeddings in a sensible manner
    posemb: [1, seq_len, dim]
    ntok_new = posemb_new.shape[1] i.e. the sequence len of the new size
    """

    # Deal with class token
    if has_class_token:  # this means classifier == 'token'
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:] # posemb_grid is [seq_len_without_cls, dim]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0] # posemb_grid is [seq_len_without_cls, dim]

    # Get old and new grid sizes
    gs_old = int(np.sqrt(len(posemb_grid))) # if 224 gs_old is 14
    gs_new = int(np.sqrt(ntok_new))  # if ntok_new is 100, then gs_new is 10
    dim = posemb_grid.shape[-1]
    posemb_grid = torch.transpose(posemb_grid, 0, 1).reshape(1, dim, gs_old, gs_old) # [dim, gs_old, gs_old]

    # Rescale grid
    posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(gs_new, gs_new), align_corners=True, mode='bicubic')
    posemb_grid = torch.transpose(posemb_grid.reshape(1, dim, gs_new * gs_new), 1, 2) # 1, gs_new * gs_new, dim

    # Deal with class token and return
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def build_2d_sincos_position_embedding(original_seq_len, embed_dim, temperature=10000.):
    h = w = int(np.sqrt(original_seq_len))
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
    pos_embedding = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    pos_embedding.requires_grad = False

    return pos_embedding





def find_topk_operation_using_name(model_name):
    """Import the module "MoCA.TopkLoss.topk_loss_module.py".
    
    """
    model_filename = "MoCA.TopkLoss.topk_loss_module"
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




class TopkLayer_noCLStopk(nn.Module):
    """implement topk layer for transformer"""
    def __init__(self, topk):
        super(TopkLayer_noCLStopk, self).__init__()
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