from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import numpy as np
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp

from .Embed import GraphEmbedding, DataEmbedding, STIDEmbeddingGraph
from .mask_strategy import *
import copy

from .Prompt_network import Memory, GCN

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

        # ==================================  STID ============================================
        self.time_in_day_emb_patch = DataEmbedding(1, embed_dim, args=self.args)
        self.time_in_day_emb_patch_graph = STIDEmbeddingGraph(1, embed_dim, GridEmb = self.time_in_day_emb_patch, args=self.args)

        self.day_in_week_emb_patch = DataEmbedding(1, embed_dim, args=self.args)
        self.day_in_week_emb_patch_graph = STIDEmbeddingGraph(1, embed_dim, GridEmb = self.day_in_week_emb_patch, args=self.args)
        
        self.node_emb_patch = DataEmbedding(1, embed_dim, args=self.args)
        self.node_emb_patch_graph = STIDEmbeddingGraph(1, embed_dim, GridEmb = self.node_emb_patch, args=self.args)
        # ==================================  STID ============================================


        # mask
        self.t_patch_size = t_patch_size
        self.decoder_embed_dim = decoder_embed_dim
        self.in_chans = in_chans
        

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias= not self.args.no_qkv_bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pred_model_linear_GraphBJ = nn.Linear(decoder_embed_dim, self.t_patch_size * 1024 * in_chans)

        self.initialize_weights_trivial()

        print("model initialized")

    def init_multiple_patch(self):
        
        self.Embedding_patch.multi_patch()

        self.time_in_day_emb_patch.multi_patch()
        self.day_in_week_emb_patch.multi_patch()
        self.node_emb_patch.multi_patch()
        
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


    def STIDEmddingGraph(self, t_i_d_data, d_i_w_data, node_id_data, data=None, mode='backward',prompt = {}, patch_size = 1, split_nodes=None):
        edges = prompt['topo']
        time_in_day_emb = self.time_in_day_emb_patch_graph(t_i_d_data, edges, split_nodes, is_time = self.args.is_time_emb, patch_size=patch_size, hour_num = data)
        day_in_week_emb = self.day_in_week_emb_patch_graph(d_i_w_data, edges, split_nodes, is_time = self.args.is_time_emb, patch_size=patch_size, hour_num = data)
        node_emb = self.node_emb_patch_graph(node_id_data, edges, split_nodes, is_time = self.args.is_time_emb, patch_size=patch_size, hour_num = data)
        return time_in_day_emb, day_in_week_emb, node_emb


    def forward_encoder(self, x, x_mark, mask_ratio, mask_strategy, seed=None, data=None, mode='backward',prompt = {}, patch_size = 1, split_nodes=None):
        # embed patches
        N, _, T, H, W = x.shape

        edges = prompt['topo']

        TimeEmb = None

        x, TimeEmb = self.Embedding_patch_graph(x, x_mark, edges, split_nodes, is_time = self.args.is_time_emb, patch_size=patch_size, hour_num = data)

        T = T // self.args.t_patch_size

        return x

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