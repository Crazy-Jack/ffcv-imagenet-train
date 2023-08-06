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
