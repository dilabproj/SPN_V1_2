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

    action_size : int = 1

    output_size: int = 9

    input_size: int = 12

    hidden_size: int = 256

    hidden_output_size: int = 1
        
    learning_rate: float = 0.01

    sigma: float = 0.02 # mutation stregth or step size


    # path
    #root_dir: str = "/home/timchen/SPNV1_2"
    root_dir: str = "../" 

    data_dir: str = "ECG-data"

    snippet_dir: str = "snippet"

    model_dir: str = "models"

    tmp_dir: str = "tmp"

    wandb_dir: str = "wandb"

    state_dir: str = "state"
        
    weight_dir: str = "weights"

    output_dir: str = "Production"

    state_model_name: str = "CNNLSTM"
        
    backbone_model_name: str = "CNNLSTM"
        
    state_name: str = "state.pkl"
        
    weight_name: str = "weight.pkl"

    dataset_name: str = "ICBEB"

    model_save_name: str = "model.pkl"

    pretrain_name: str = "pretrained.pt"

    snippet_name: str = "christov_norm_1000.pickle"
