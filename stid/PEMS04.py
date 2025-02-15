import os
import sys
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from .data.TSF_dataset import TimeSpaceForecastingDataset
from .runners.TSF_runner import TimeSpaceForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from .arch import STID_Prompt

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'PEMS04'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data
# Model architecture and parameters
MODEL_ARCH = STID_Prompt
MODEL_PARAM = {
    "num_nodes": 307,
    "input_len": INPUT_LEN,
    "input_dim": 3,
    "embed_dim": 32,
    "output_len": OUTPUT_LEN,
    "num_layer": 3,
    "if_node": True,
    "node_dim": 32,
    "if_T_i_D": True,
    "if_D_i_W": True,
    "temp_dim_tid": 32,
    "temp_dim_diw": 32,
    "time_of_day_size": 288,
    "day_of_week_size": 7,


    #Prompt params
    "patch_size": 4,
    # "seq_len": 24,
    # "data_dir": "",
    # "weight_decay": 0.0,
    # "batch_size": 2,
    # "early_stop": 10,
    # "log_interval": 5,
    # "device_id": '0',
    # "machine": 'LM1',
    "mask_ratio": 0.5,
    # "random": True,
    # "eval": True,
    # "is_block": True,
    # "data_norm": 1,
    # "clip_grad": 0.02,
    "mask_strategy": 'causal',
    "mode": 'training',
    # "file_load_path": '',
    # "lr_anneal_steps": 150,
    # "total_epoches": 200,
    # "lr": 1e-3,
    "pos_emb": 'SinCos',
    "no_qkv_bias": 0,
    "is_prompt": 1,
    "is_time_emb": 1,
    # "batch_size_taxibj": 128,
    # "batch_size_nj": 256,
    # "batch_size_nyc": 256,
    # "batch_size_crowd": 256,
    # "emb_tuning": 0,
    # "mask_ablation": '',
    # "data_type": 'GridGraph',
    # "batch_size_graph_large": 32,
    # "batch_size_graph_small": 64,
    # "spec_mlp": 0,
    # "few_ratio": 1.0,
    "multi_patch": True,
    # "seed": 100,
    # "few_data": '',
    # "finetune": 0,


    "t_patch_size": 4,
    "his_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "used_data": "GridGraphall",
    "num_memory": 128,
    "prompt_content": "node_graph",
    "size": "middle",
    # "batch_ratio": 0.05,

}
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = TimeSpaceForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSpaceForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MAPE': masked_mape,
                                'RMSE': masked_rmse,
                            })
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0008, #0.002,
    "weight_decay": 0.0005,#0.0001,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 50, 80],
    "gamma": 0.5
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 1.0 #5.0
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 32

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 32

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.HORIZONS = [] # Prediction horizons for evaluation. Default: [] 3, 6, 12
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True
