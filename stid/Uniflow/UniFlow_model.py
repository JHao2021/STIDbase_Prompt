from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import numpy as np
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp

from .Embed import GraphEmbedding, DataEmbedding, TokenEmbedding, SpatialPatchEmb, get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_with_resolution, get_1d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid_with_resolution
from .mask_strategy import *
import copy

from .Prompt_network import Memory, GCN


class TransformerDecoderModel(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        self.linear = nn.Linear(d_model, d_model)  # Adjust the output dimension as needed

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        output = self.linear(output)
        return output

def model_select(args, **kwargs):
    if args.size == 'small': 
        model = UniFlow(
            embed_dim=128,
            depth=4,
            decoder_embed_dim = 128,
            decoder_depth=4,
            num_heads=4,
            decoder_num_heads=4,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model

    elif args.size == 'middle': 
        model = UniFlow(
            embed_dim=256,
            depth=4,
            decoder_embed_dim = 256,
            decoder_depth=4,
            num_heads=4,
            decoder_num_heads=4,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model
 
    elif args.size == 'large': 
        model = UniFlow(
            embed_dim=256,
            depth=6,
            decoder_embed_dim = 256,
            decoder_depth=6,
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim, bias= qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x, attn_bias = {}):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_bias!={}:
            
            if 'bias_t' in attn_bias:
                T = attn.shape[-1] // attn_bias['bias_t'].shape[-1]
            elif 'bias_f' in attn_bias:
                T = attn.shape[-1] // attn_bias['bias_f'].shape[-1]
            elif 'topo' in attn_bias:
                T = attn.shape[-1] // attn_bias['topo'].shape[-1]

            if 'bias_t' in attn_bias:
                attn_bias_t = attn_bias['bias_t'].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=4)
                attn_bias_t = attn_bias_t.repeat(1, self.num_heads, T, 1, T, 1)
                attn_bias_t = attn_bias_t.reshape(attn_bias_t.shape[0], self.num_heads, attn.shape[-2], attn.shape[-1])

                assert attn.shape == attn_bias_t.shape

                attn += attn_bias_t

            if 'bias_f' in attn_bias:

                attn_bias_f = attn_bias['bias_f'].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=4)
                attn_bias_f = attn_bias_f.repeat(1, self.num_heads, T, 1, T, 1)
                attn_bias_f = attn_bias_f.reshape(attn_bias_f.shape[0], self.num_heads, attn.shape[-2], attn.shape[-1])

                assert attn.shape == attn_bias_f.shape
                
                attn += attn_bias_f

            if 'topo' in attn_bias:

                attn_bias_topo = attn_bias['topo'].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=4)
                attn_bias_topo = attn_bias_topo.repeat(1, self.num_heads, T, 1, T, 1)
                attn_bias_topo = attn_bias_topo.reshape(attn_bias_topo.shape[0], self.num_heads, attn.shape[-2], attn.shape[-1])

                assert attn.shape == attn_bias_topo.shape
                
                attn += attn_bias_topo

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, attn_bias = {}):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_bias = attn_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x




class UniFlow(nn.Module):
    def __init__(self,  in_chans=1,
                 embed_dim=1024, decoder_embed_dim=512, depth=24, decoder_depth=4, num_heads=16,  decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, t_patch_size=1,
                 no_qkv_bias=False, pos_emb = 'trivial', args=None, ):
        super().__init__()

        self.args = args

        self.pos_emb = pos_emb

        self.Embedding_patch = DataEmbedding(1, embed_dim, args=self.args)
        self.Embedding_patch_graph = GraphEmbedding(1, embed_dim, GridEmb = self.Embedding_patch, args=self.args)

        # mask
        self.t_patch_size = t_patch_size
        self.decoder_embed_dim = decoder_embed_dim
        self.in_chans = in_chans
        

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, embed_dim)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 50, embed_dim)
        )

        self.decoder_pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, decoder_embed_dim)
        )
        self.decoder_pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 50,  decoder_embed_dim)
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.no_qkv_bias = no_qkv_bias
        self.norm_layer = norm_layer

        
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias= not self.args.no_qkv_bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        encdoer_layer2 = nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=2, dim_feedforward=decoder_embed_dim//2, batch_first = True)
        self.spatial_attn_spec_tmp = nn.TransformerEncoder(encoder_layer=encdoer_layer2, num_layers=1)

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_model = TransformerDecoderModel(d_model=decoder_embed_dim, dim_feedforward = decoder_embed_dim//2, nhead=2, num_decoder_layers=1)
        # self.pred_model_linear_GraphBJ = nn.Linear(decoder_embed_dim, self.t_patch_size * 105 * in_chans)
        self.pred_model_linear_GraphBJ = nn.Linear(decoder_embed_dim, self.t_patch_size * 512 * in_chans)
        self.pred_model_linear_GraphNJ = nn.Linear(decoder_embed_dim, self.t_patch_size * 105 * in_chans)
        self.pred_model_linear_GraphSH  = nn.Linear(decoder_embed_dim, self.t_patch_size * 210 * in_chans)

        self.initialize_weights_trivial()

        print("model initialized")

    def init_multiple_patch(self):
        
        self.Embedding_patch.multi_patch()

        self.head_layer_1 = nn.Sequential(*[
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.t_patch_size * 1**2 * self.in_chans, bias= not self.args.no_qkv_bias)
        ])

        self.head_layer_2 = nn.Sequential(*[
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.t_patch_size * 2**2 * self.in_chans, bias= not self.args.no_qkv_bias)
        ])
        
        self.head_layer_4 = nn.Sequential(*[
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.t_patch_size * 4**2 * self.in_chans, bias= not self.args.no_qkv_bias)
        ])

        self.initialize_weights_trivial()


    def initialize_weights_trivial(self):
        torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

        torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_encoder(self, x, x_mark, mask_ratio, mask_strategy, seed=None, data=None, mode='backward',prompt = {}, patch_size = 1, split_nodes=None):
        # embed patches
        N, _, T, H, W = x.shape

        edges = prompt['topo']

        TimeEmb = None

        x, TimeEmb = self.Embedding_patch_graph(x, x_mark, edges, split_nodes, is_time = self.args.is_time_emb, patch_size=patch_size, hour_num = data)
        

        T = T // self.args.t_patch_size

        x, mask, ids_restore, ids_keep = causal_masking(x, mask_ratio, T=T, mask_strategy=mask_strategy)

        input_size = (T, len(split_nodes), 1)

     
        return x, mask, ids_restore, input_size, TimeEmb

    def forward_decoder(self, x, x_mark, mask, ids_restore, mask_strategy, TimeEmb, input_size=None,  data=None):
        N = x.shape[0]
        T, H, W = input_size

        # embed tokens
        x = self.decoder_embed(x)
            
        C = x.shape[-1]

        x = causal_restore(x, ids_restore, N, T, H,  W, C, self.mask_token)

        if self.args.is_time_emb == 1:
            x_attn = x + TimeEmb
        else:
            x_attn = x 

        return x_attn

    def forward_loss(self, imgs, pred, mask, patch_size):
        """
        imgs: [N, 1, T, H, W]
        pred: [N, t*h*w, u*p*p*1]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """

        target = self.patchify(imgs, patch_size)

        assert pred.shape == target.shape

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss1 = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss2 = (loss * (1-mask)).sum() / (1-mask).sum()
        return loss1, loss2, target

    def graph_loss(self,pred, target):
        assert pred.shape == target.shape
        assert pred.shape[1] == self.args.his_len + self.args.pred_len

        loss1 = ((pred[:,self.args.his_len:] - target[:,self.args.his_len:]) ** 2).mean()

        loss2 = ((pred[:,:self.args.his_len] - target[:,:self.args.his_len]) ** 2).mean()

        mask = torch.ones_like(target)

        mask[:,:self.args.his_len] = 0

        return loss1, loss2, target, mask


    def Output_Proj(self, pred, subgraphs, data, imgs):
        T, H, W = imgs.shape[2:]
        seq_lengths = [len(i) for i in subgraphs] 
        max_len = max(seq_lengths)

        pred = self.pred_model_linear_GraphBJ(pred).reshape(pred.shape[0],T//self.args.t_patch_size,len(subgraphs),self.args.t_patch_size, -1).permute(0,1,3,2,4)

        
        pred = pred.reshape(pred.shape[0], T, len(subgraphs), -1)

        pred = torch.cat([pred[:,:,g,:seq_lengths[g]] for g in range(pred.shape[2])],dim=2)

        target = imgs.squeeze(dim=(1,4))

        target = torch.cat([torch.gather(target, 2, group.view(1, 1, group.shape[0]).expand(target.shape[0], target.shape[1], group.shape[0]).to(target).long()) for group in subgraphs],dim=2)

        return pred, target