"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np 

from .transformer import Transformer
from .utils import load_pretrained_weights, as_tuple, resize_positional_embedding_torch, build_2d_sincos_position_embedding
from .configs import PRETRAINED_MODELS
from .linedrawing import Generator

class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim, type="learnable"):
        super().__init__()
        self.original_seq_len = seq_len
        if type == "learnable":
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
        elif type == "sincos2d":
            self.pos_embedding = build_2d_sincos_position_embedding(self.original_seq_len, dim)

    def forward(self, x, has_class_token=True):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        ntok_new = x.shape[1]
        if ntok_new != self.original_seq_len:
            pos_embedding = resize_positional_embedding_torch(self.pos_embedding, ntok_new, has_class_token=has_class_token)
        else:
            pos_embedding = self.pos_embedding
        return x + pos_embedding

class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3, 
        topk_layer_name: str = 'TopkLayer',
        posemb_type: str = 'learnable',
        pool_type: str = 'tok',
        image_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        topk_info: Optional[str] = None,
    ):
        super().__init__()

        # Configuration
        if name is None:
            # check_msg = 'must specify name of pretrained model'
            # assert not pretrained, check_msg
            # assert not resize_positional_embedding, check_msg
            # assert not check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size                

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        # Patch embedding
        # self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1
        
        # Positional embedding
        self.posemb_type = posemb_type
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim, type=posemb_type)
        else:
            raise NotImplementedError()
        
        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, topk_info=topk_info, topk_layer_name=topk_layer_name)

        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)
    
        # Initialize weights
        self.init_weights()
        
        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            load_pretrained_weights(
                self, name, 
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(image_size != pretrained_image_size),
            )
        self.pool_type = pool_type


    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        if self.posemb_type == 'learnable':
            nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x, has_class_token=hasattr(self, 'class_token'))  # b,gh*gw+1,d 
        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            if self.pool_type == 'tok':
                x = self.norm(x)[:, 0]  # b,d
            elif self.pool_type == 'gap':
                x = self.norm(x)[:, 1:].mean(dim=1)
            x = self.fc(x) # b,num_classes

        return x




class ViT_sparse(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3, 
        topk_layer_name: str = 'TopkLayer',
        posemb_type: str = 'learnable',
        pool_type: str = 'tok',
        image_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        topk_info: Optional[str] = None,
        line_drawing_ckpt: Optional[str] = None,
        patch_selection_topk: Optional[float] = None,
    ):
        super().__init__()

        # Configuration
        if name is None:
            # check_msg = 'must specify name of pretrained model'
            # assert not pretrained, check_msg
            # assert not resize_positional_embedding, check_msg
            # assert not check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size                

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        self.fh = fh
        self.fw = fw
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        # Patch embedding
        # self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1
        
        # Positional embedding
        self.posemb_type = posemb_type
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim, type=posemb_type)
        else:
            raise NotImplementedError()
        
        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, topk_info=topk_info, topk_layer_name=topk_layer_name)

        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)
    
        # Initialize weights
        self.init_weights()
        
        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            load_pretrained_weights(
                self, name, 
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(image_size != pretrained_image_size),
            )
        self.pool_type = pool_type

        # initialize line drawing
        self.net_G = Generator(3, 1, n_residual_blocks=3)
        # Load state dicts
        self.net_G.load_state_dict(torch.load(line_drawing_ckpt))
        print('==> loaded linedrawing network', line_drawing_ckpt)
        self.patch_selection_topk = float(patch_selection_topk) * 0.01 
        for param in self.net_G.parameters():
            param.requires_grad = False



    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        if self.posemb_type == 'learnable':
            nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    
    def patch_selection(self, x, x_patch_embedding, visualize):
        # x: [b, c, fh, fw]
        # x_patch_embedding: [b, gh*gw+1,d]
        
        
        b, c, _, _ = x.shape
        b, L, d = x_patch_embedding.shape
        
        topk = self.patch_selection_topk
        if topk >= 1.:
            return x_patch_embedding

        
        with torch.no_grad():
            out = self.net_G(x)
            linedrawing_pooled = F.avg_pool2d(x[:, :1, :, :], kernel_size=self.fh, stride=self.fh)  # (B, 1, Gh, Gw)
            linedrawing_pooled = linedrawing_pooled.view(b, -1)

            topk = int(topk * linedrawing_pooled.shape[1])  # Compute number of patches to keep
            _, topk_indices = torch.topk(linedrawing_pooled.abs(), topk, dim=1)  # Get indices of top-K activations
            if visualize:
                # visualize the topk patches 
                pass 

                
        # Extract spatial tokens (ignoring CLS token)
        spatial_tokens = x_patch_embedding[:, 1:, :]  # Shape: [b, gh*gw, d]
        # Retain CLS token
        cls_token = x_patch_embedding[:, 0:1, :]  # Shape: [b, 1, d]
        # select
        selected_patches = torch.gather(spatial_tokens, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, d))
        
        # Concatenate CLS token
        x_patch_embedding_selected = torch.cat([cls_token, selected_patches], dim=1)
        
        return x_patch_embedding_selected
        

    def forward(self, x, visualize=False):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        inputs = x
        x = self.patch_embedding(x)  # b,d,gh,gw
        
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x, has_class_token=hasattr(self, 'class_token'))  # b,gh*gw+1,d 
        
        x = self.patch_selection(inputs, x, visualize=visualize) # # b,gh*gw+1,d -> # b,sparse_topk+1,d 

        
        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            if self.pool_type == 'tok':
                x = self.norm(x)[:, 0]  # b,d
            elif self.pool_type == 'gap':
                x = self.norm(x)[:, 1:].mean(dim=1)
            x = self.fc(x) # b,num_classes

        return x

