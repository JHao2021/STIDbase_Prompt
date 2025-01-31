import torch
from torch import nn

from .mlp import MultiLayerPerceptron, LastLayerMultiLayerPerceptron

from stid.Uniflow.UniFlow_model import UniFlow, model_select
from types import SimpleNamespace
from easytorch.device import to_device

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

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_day_in_week) + \
            self.temp_dim_diw*int(self.if_time_in_day)
        # self.hidden_dim = self.embed_dim
        
        # self.encoder = nn.Sequential(
        #     *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer - 1)],  # 前 num_layer-1 层
            LastLayerMultiLayerPerceptron(self.hidden_dim, self.embed_dim))  # 最后一层，输出通道数为 256)

        # regression
        # self.regression_layer = nn.Conv2d(
        #     in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        self.regression_layer = nn.Conv2d(
            in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        model_args["mask_ratio"] = model_args["input_len"] / (model_args["input_len"] + model_args["output_len"])
        self.args = SimpleNamespace(**model_args)
        self.prompt = model_select(args=self.args) # Uniflow模型的规模决定了embed_dim,STID的embed_dim在CFG里面
        if self.args.multi_patch:
            print('multi_patch')
            self.prompt.init_multiple_patch()   
        # if self.args.is_prompt == 1:     
        #     self.prompt.init_prompt()

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, 
    mask_ratio=0.5, mask_strategy='causal',seed=520, data_name='none',  mode='backward',topo = None, subgraphs = None, patch_size = 100, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        # prepare data
        B, L, N, C = history_data.shape
        # =====================Prompt=========================================
        imgs = history_data[..., [0]].permute(0, 3, 1, 2).unsqueeze(-1) #从[B, L, N, C]改为[B, C, L, N, 1]
        imgs_mark = history_data[:, :, 0, [1, 2]]

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
        

        img_tmp = None
        img_spec = None

        x_attn = self.prompt.forward_encoder(imgs, imgs_mark, mask_ratio, mask_strategy, seed=seed, data=data_name, mode=mode, prompt = {'t': img_tmp, 'f':img_spec,'topo':topo}, patch_size = patch_size, split_nodes=subgraphs)
        # x_attn : [B, L, D]

        #============================STID===================================
        input_data = history_data[..., range(self.input_dim)]
        
        t_i_d_data = history_data[..., 1].unsqueeze(1).unsqueeze(-1) # [B, C, L, N, 1]
        d_i_w_data = history_data[..., 2].unsqueeze(1).unsqueeze(-1)
        

        n_indices = torch.arange(N, dtype=torch.float32)  # node id 的值就是第3维N的索引
        node_id_data = to_device(n_indices.view(1, 1, 1, N, 1).expand(B, 1, L, N, 1))  # [B, C, L, N, 1]

        time_in_day_emb, day_in_week_emb, node_emb = self.prompt.STIDEmddingGraph(
            t_i_d_data, d_i_w_data, node_id_data, data=data_name, mode=mode, prompt = {'t': img_tmp, 'f':img_spec,'topo':topo}, patch_size = patch_size, split_nodes=subgraphs)
        

        # concate all embeddings
        # hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        fused_emb = torch.cat([x_attn, time_in_day_emb, day_in_week_emb, node_emb], dim=-1) #add stid

        #============================STID===================================
        fused_emb = fused_emb.permute(0, 2, 1).unsqueeze(-1)
        # encoding
        hidden = self.encoder(fused_emb) #[B, D, L, 1], 沿D维编码
        hidden = hidden.squeeze(-1).permute(0, 2, 1)

        # =========================================================================
        # regression
        hidden = hidden.permute(0, 2, 1).unsqueeze(-1)
        prediction = self.regression_layer(hidden)
        prediction = prediction.squeeze(-1).permute(0, 2, 1)


        # [prompt]Output Projection : 
        prediction, target = self.prompt.Output_Proj(prediction, subgraphs, data_name, imgs)
        prediction = prediction.unsqueeze(-1)
        target = target.unsqueeze(-1)

        return prediction, target
