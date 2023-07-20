from datetime import datetime
from dataclasses import dataclass


@dataclass
class Config():
    # model transfer map
    model_name: str = "EARLIEST"

    num_worker: int = 4

    epoch_size: int = 100

    learning_rate: float = 0.0001

    total_length: int = 30000
        
    series_ratio: float = 1.0

    batch_size: int = 1

    hidden_size: int = 10

    layer_size: int = 1

    input_size: int = 12

    output_size: int = 5

    lambda_val: float = 1e-06

    message: str = "None"

    seed: int = 1


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
