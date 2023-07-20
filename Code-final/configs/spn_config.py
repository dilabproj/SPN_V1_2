import os
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Config():
    # model transfer map
    model_name: str = "Snippet_Policy_Learning"

    num_worker: int = 4

    epoch_size: int = 100

    learning_rate: float = 0.0001

    batch_size: int = 12

    input_size: int = 12

    hidden_size: int = 256

    hidden_output_size: int = 1

    output_size: int = 9

    alpha: float = 0.3

    beta: float = 0.001

    gamma: float = 0.5

    message: str = "None"
        
    offset: int = 499
    
    seed: int = 1

    # path
    #root_dir: str = "/home/timchen/SPNV1_2/"
    root_dir: str = "../"

    data_dir: str = "ECG-data"

    snippet_dir: str = "snippet"

    model_dir: str = "models"

    wandb_dir: str = "wandb"

    state_dir: str = "state"

    output_dir: str = "Production"

    state_name: str = "state.pkl"

    dataset_name: str = "ICBEB"

    snippet_name: str = "christov_1000_doubel_check.pickle"
