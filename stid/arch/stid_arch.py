import torch
from torch import nn

from .mlp import MultiLayerPerceptron

from stid.Uniflow.UniFlow_model import UniFlow, model_select
from types import SimpleNamespace

class STID_Prompt(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        # self.hidden_dim = self.embed_dim+self.node_dim * \
        #     int(self.if_spatial)+self.temp_dim_tid*int(self.if_day_in_week) + \
        #     self.temp_dim_diw*int(self.if_time_in_day)
        self.hidden_dim = 256
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        # self.regression_layer = nn.Conv2d(
        #     in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=(1, 1), bias=True)

        model_args["mask_ratio"] = model_args["input_len"] / (model_args["input_len"] + model_args["output_len"])
        self.args = SimpleNamespace(**model_args)
        self.prompt = model_select(args=self.args)
        if self.args.multi_patch:
            print('multi_patch')
            self.prompt.init_multiple_patch()  
        if self.args.is_prompt == 1: 
            print('start prompt')        
            self.prompt.init_prompt()

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, 
    mask_ratio=0.5, mask_strategy='causal',seed=520, data_name='none',  mode='backward',topo = None, subgraphs = None, patch_size = 100, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        # Prompt
        combined_data = torch.cat((history_data, future_data), dim=1)
        imgs = combined_data[..., [0]].permute(0, 3, 1, 2).unsqueeze(-1) #从[B, L, N, C]改为[B, C, L, N, 1]
        imgs_mark = combined_data[:, :, 0, [1, 2]]

        #由于Prompt的tid和diw都是整数，所以重新处理时间特征
        time_of_day = imgs_mark[..., 0] 
        day_of_week = imgs_mark[..., 1] 

        steps_per_day = None
        if("PEMS" in data_name):
            steps_per_day = 288
        original_hours = torch.floor(time_of_day * steps_per_day).long()  
        original_days = torch.floor(day_of_week * 7).long()  

        imgs_mark[..., 0] = original_days
        imgs_mark[..., 1] = original_hours   
        imgs_mark = imgs_mark.to(torch.int64) 
        
        if self.args.is_prompt == 1:
            # if 'Graph' not in data_name:
            #     img_tmp, img_spec = self.prompt.adpative_graph(imgs, imgs_mark, self.prompt.Embedding_patch, data=data_name, patch_size = patch_size)
            # else:
                img_tmp, img_spec = self.prompt.adpative_graph(imgs, imgs_mark, self.prompt.Embedding_patch_graph, data=data_name, node_split = subgraphs, patch_size = patch_size)
        else:
            img_tmp = None
            img_spec = None

        T, H, W = imgs.shape[2:]
        x_attn, mask, ids_restore, input_size, TimeEmb, prompt = self.prompt.forward_encoder(imgs, imgs_mark, mask_ratio, mask_strategy, seed=seed, data=data_name, mode=mode, prompt = {'t': img_tmp, 'f':img_spec,'topo':topo}, patch_size = patch_size, split_nodes=subgraphs)
        hidden = x_attn
        # =========================================================================

        hidden = self.prompt.forward_decoder(hidden, imgs_mark, mask, ids_restore, mask_strategy, TimeEmb, input_size = input_size, data = data_name, prompt_graph = prompt)  # [N, L, p*p*1]


        prediction = hidden.squeeze(-1)

        # [prompt]Output Projection : 
        prediction, target = self.prompt.Output_Proj(prediction, subgraphs, data_name, imgs)
        assert prediction.shape == target.shape
        assert prediction.shape[1] == self.args.his_len + self.args.pred_len
        # prediction = prediction.view(B, L, N, C)

        prediction = prediction.unsqueeze(-1)
        target = target.unsqueeze(-1)

        prediction = prediction[:, self.input_len:, :, :]
        return prediction, target
