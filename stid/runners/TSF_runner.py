from typing import Dict

import torch
from .base_tsf_runner import BaseTimeSeriesForecastingRunner

class TimeSpaceForecastingRunner(BaseTimeSeriesForecastingRunner):
    """
    A Simple Runner for Time Series Forecasting: 
    Selects forward and target features. This runner is designed to handle most cases.

    Args:
        cfg (Dict): Configuration dictionary.
    """

    def __init__(self, cfg: Dict):

        super().__init__(cfg)
        self.forward_features = cfg['MODEL'].get('FORWARD_FEATURES', None)
        self.target_features = cfg['MODEL'].get('TARGET_FEATURES', None)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Selects input features based on the forward features specified in the configuration.

        Args:
            data (torch.Tensor): Input history data with shape [B, L, N, C].

        Returns:
            torch.Tensor: Data with selected features.
        """

        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Selects target features based on the target features specified in the configuration.

        Args:
            data (torch.Tensor): Model prediction data with arbitrary shape.

        Returns:
            torch.Tensor: Data with selected target features and shape [B, L, N, C].
        """

        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: Dict, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> Dict:
        """
        Performs the forward pass for training, validation, and testing. 

        Args:
            data (Dict): A dictionary containing 'target' (future data) and 'inputs' (history data) (normalized by self.scaler).
            epoch (int, optional): Current epoch number. Defaults to None.
            iter_num (int, optional): Current iteration number. Defaults to None.
            train (bool, optional): Indicates whether the forward pass is for training. Defaults to True.

        Returns:
            Dict: A dictionary containing the keys:
                  - 'inputs': Selected input features.
                  - 'prediction': Model predictions.
                  - 'target': Selected target features.

        Raises:
            AssertionError: If the shape of the model output does not match [B, L, N].
        """

        # Preprocess input data
        future_data, history_data, topo, data_name = data['target'], data['inputs'], data['topo'], data['data_name']
        topo = topo[0, ...]
        
        # prompt的topo是边，不是邻接矩阵
        edges = []
        for i in range(topo.size(0)):
            for j in range(i + 1, topo.size(1)):  # 只考虑上三角部分以避免重复
                if topo[i, j] != 0:
                    edges.append((i, j))
        edges = torch.tensor(edges)

        data_name = data_name[0]
        history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = self.to_running_device(future_data)    # Shape: [B, L, N, C]
        edges = self.to_running_device(edges)
        

        subgraphs = []
        subgraphs.append(torch.arange(0, topo.size(0)))

        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)
        
        if not train:
            # For non-training phases, use only temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Forward pass through the model
        model_return, model_target = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train,
            topo=edges,  
            data_name=data_name,  
            mask_ratio=0.5,  
            mask_strategy='causal',  
            seed=520,  
            mode='backward',  
            patch_size=1,  
            subgraphs=subgraphs,
            **kwargs  
        )

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)
            # model_return['target'] = model_target

        # Ensure the output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        return model_return
