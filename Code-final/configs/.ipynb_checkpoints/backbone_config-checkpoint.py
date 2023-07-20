from datetime import datetime
from dataclasses import dataclass

@dataclass
class Config():
    # model transfer map
    model_name: str = "CNNLSTM"

    num_worker: int = 4

    epoch_size: int = 100

    learning_rate: float = 0.0001

    batch_size: int = 32

    input_size: int = 12

    hidden_size: int = 256

    hidden_output_size: int = 1

    output_size: int = 9

    message: str = "None"
    
    seed: int = 1

    #threshold

    training_performance_thr: float = 0.0
    
    testing_performance_thr: float = 0.0

    # path
    root_dir: str = "/home/waue0920/hugo/"

    data_dir: str = "ECG"

    snippet_dir: str = "snippet"

    model_dir: str = "models"

    wandb_dir: str = "wandb"

    state_dir: str = "state"
        
    weight_dir: str = "weights"

    output_dir: str = "eTSC"

    state_name: str = "state.pkl"
        
    weight_name: str = "weight.pkl"

    dataset_name: str = "ICBEB"

    pretrain_name: str = "pretrained.pt"

    snippet_name: str = "christov_norm_1000.pickle"
