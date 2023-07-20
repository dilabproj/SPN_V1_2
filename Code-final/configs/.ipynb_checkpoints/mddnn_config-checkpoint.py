import os
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Config():
    # model transfer map
    model_name: str = "MDDNN"

    gpu_device_order_type: str = "PCI_BUS_ID"
    
    gpu_devices_id: str = "2"

    num_worker: int = 4

    epoch_size: int = 100

    learning_rate: float = 0.0001

    series_ratio: float = 1.0

    batch_size: int = 32

    input_size: int = 12

    output_size: int = 9

    message: str = "None"

    seed: int = 1

    total_length: int = 30000

    ratio: int = 1

    # path
    root_dir: str = "/home/waue0920/hugo/"

    data_dir: str = "ECG"

    snippet_dir: str = "snippet"

    tmp_dir: str = "tmp"

    model_dir: str = "models"

    wandb_dir: str = "wandb"

    state_dir: str = "state"

    output_dir: str = "eTSC"

    state_name: str = "state.pkl"

    dataset_name: str = "ptbxl"

    snippet_name: str = "christov_norm_1000.pickle"
