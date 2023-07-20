from datetime import datetime
from dataclasses import dataclass


@dataclass
class Config():

    model_name: str = "KGEVO"

    # enviorment
    core_size: int = 30

    individual_size: int = 50

    population_size: int = 2000

    eval_threshold: float = 500
        
    seed: int = 0
        
    message: str = ""

    # model parameters

    input_size: int = 256

    action_size : int = 1

    output_size: int = 9

    learning_rate: float = 0.01

    sigma: float = 0.02 # mutation stregth or step size


    # path
    root_dir: str = "/home/waue0920/hugo/"

    data_dir: str = "ECG"

    snippet_dir: str = "snippet"

    model_dir: str = "models"

    wandb_dir: str = "wandb"

    state_dir: str = "state"
        
    weight_dir: str = "weights"

    output_dir: str = "eTSC"

    state_model_name: str = "CNNLSTM"

    state_name: str = "state.pkl"
        
    weight_name: str = "weight.pkl"

    dataset_name: str = "ICBEB"

    model_save_name: str = "model.pkl"
