from datetime import datetime
from dataclasses import dataclass


@dataclass
class Config():
    # model transfer map
    model_name: str = "ELECTS"

    num_worker: int = 4

    epoch_size: int = 100

    learning_rate: float = 0.0001

    series_ratio: float = 1.0

    batch_size: int = 64

    input_size: int = 12

    hidden_size: int = 256

    layer_size: int = 2

    total_length: int = 30000

    output_size: int = 9

    dropout: float = 0.5

    message: str = "None"

    seed: int = 1

    alpha: float = 0.5
        
    pretrained: bool = False
        
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

    dataset_name: str = "ICBEB"

    snippet_name: str = "christov_norm_1000.pickle"
