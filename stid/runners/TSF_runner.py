from typing import Dict

import torch
from .base_tsf_runner import BaseTimeSeriesForecastingRunner
import time
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

    def remove_duplicates_topo(self, topo):
        cleaned_topo = []
        for pair in topo:
            # 提取每个张量的第一个值（因为所有值都相同）
            first_value = pair[0][0].item()  # 转换为 Python 标量
            second_value = pair[1][0].item()  # 转换为 Python 标量
            cleaned_topo.append([first_value, second_value])
        return torch.tensor(cleaned_topo)


    def remove_duplicates_subgraphs(self, subgraphs):
        cleaned = []
        for sub in subgraphs:
            # 对每个子图中的 tensor 进行拼接并去重
            unique_values = torch.cat([tensor.unique() for tensor in sub])
            # 将去重后的结果存储为一个张量
            cleaned.append(unique_values)
        return cleaned
    
    def remove_duplicates_subgraphs_foreachnode(self, subgraphs):
        cleaned = []
        for sub in subgraphs:  # 遍历每个子图
            for tensor_list in sub:  # 遍历子图中的每个张量列表
                for tensor in tensor_list:  # 遍历张量列表中的每个张量
                    # 提取唯一值并转换为整数
                    sub_cleaned = []
                    unique_value = int(torch.unique(tensor).item())
                    sub_cleaned.append(unique_value)
                    # 将每个子图的 list 转换为 tensor
                    cleaned.append(torch.tensor(sub_cleaned))
        return cleaned
    
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
        future_data, history_data, topo, data_name, subgraphs = data['target'], data['inputs'], data['topo'], data['data_name'], data['subgraphs']
        topo = self.remove_duplicates_topo(topo)
        
        # start_time = time.time()
        subgraphs = self.remove_duplicates_subgraphs(subgraphs)
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"执行时间: {execution_time:.6f} 秒")

        data_name = data_name[0]
        history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = self.to_running_device(future_data)    # Shape: [B, L, N, C]
        topo = self.to_running_device(topo)


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
            topo=topo,  
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
