import os
import torch
import random
import numpy as np
import torch.nn as nn
import utils.io as uio

from tqdm import tqdm
from sklearn.metrics import f1_score, fbeta_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from core.loss import FocalLoss
from configs.backbone_config import Config
from models.snippet_cnnlstm import snippet_cnnlstm
#import mkl
#mkl.set_num_threads(3)
torch.set_num_threads(3)

def execute(config, pretrained_model, train_X, train_Y, test_X, test_Y, train_I, test_I, train_L, test_L):

    print("In: ",config.model_name)
    
    model = snippet_cnnlstm(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            hidden_output_size=config.hidden_output_size,
                            output_size=config.output_size,
                            core_model = config.model_name,
                            isCuda = True)

    model.load_state_dict(torch.load(pretrained_model))

    model.cuda()

    model.eval()
    
    train_state = []
    
    train_index = []
    
    for idx in tqdm(range(train_X.shape[0])):
        X = train_X[idx:idx+1]
        state = model.predict(X)

        train_state.append(np.array(state))
        
        train_index.append(train_I[idx]+249)

    test_state = []
    
    test_index = []

    for idx in tqdm(range(test_X.shape[0])):
        X = test_X[idx:idx+1]

        state = model.predict(X)

        test_state.append(np.array(state))
        
        test_index.append(test_I[idx]+249)

    train_state = np.array(train_state)
    test_state = np.array(test_state)

    return train_state, test_state, train_index, test_index


def get_weight(config, pretrained_model):
    
    model = snippet_cnnlstm(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            hidden_output_size=config.hidden_output_size,
                            output_size=config.output_size,
                            core_model = config.model_name)

    model.load_state_dict(torch.load(pretrained_model))
    
    weight = model.Discriminator.fc.weight.cpu().detach().numpy()
        
    bias = model.Discriminator.fc.bias.cpu().detach().numpy()
    
    output = {
        "weight":weight.T,
        "bias":bias
    }
    
    return output


if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    
    
    # "D3CNN", "HeartNetIEEE", "ResCNN","CNNLSTM"
    
    for seed in range(1, 11):
        config = Config(dataset_name = "ptbxl", 
                        output_size = 5, 
                        model_name = "CNNLSTM-500",
                        snippet_name = "christov_500_checkup.pickle")
        
        print("Here:", config.model_name)
        
        input_folder = os.path.join(config.root_dir,
                                config.data_dir,
                                config.dataset_name,
                                config.snippet_dir,
                                config.snippet_name)
        # Snippet Input
        X, Y, I, L = uio.load_snippet_data_with_il(input_folder)
        
        pretrained_model = os.path.join(config.root_dir,
                                        config.output_dir,
                                        config.model_dir,
                                        config.model_name,
                                        config.dataset_name,
                                        str(seed),
                                        config.pretrain_name)

        weight_path = os.path.join(config.root_dir,
                                        config.output_dir,
                                        config.weight_dir,
                                        config.model_name,
                                        config.dataset_name,
                                        str(seed))

        state_path = os.path.join(config.root_dir,
                                        config.output_dir,
                                        config.state_dir,
                                        config.model_name,
                                        config.dataset_name,
                                        str(seed))


        print(pretrained_model)

        uio.check_folder(weight_path)
        
        uio.check_folder(state_path)

        
        if config.dataset_name == "ptbxl":
            
            X, Y, I, L, info = uio.load_snippet_data_with_il_info(input_folder)
            
            train_val_data = X[info.strat_fold!=seed]
            testing_data = X[info.strat_fold==seed]
            
            train_val_label = Y[info.strat_fold!=seed] 
            testing_label = Y[info.strat_fold==seed]
            
            training_index = I[info.strat_fold!=seed] 
            testing_index = I[info.strat_fold==seed] 
            
            training_length = L[info.strat_fold!=seed] 
            testing_length = L[info.strat_fold==seed] 
            
        else:
            X, Y, I, L = uio.load_snippet_data_with_il(input_folder)
            
            y = np.argmax(Y, axis=1)
        
            sss = StratifiedKFold(n_splits=10, random_state=0)

            sss.get_n_splits(X, y)

            for index, (train_val_index, test_index) in enumerate(sss.split(X, y)):
                if(index is (seed-1)):
                    print("Runing:",seed)
                    break

            train_val_data, testing_data = X[train_val_index], X[test_index]

            train_val_label, testing_label = Y[train_val_index], Y[test_index]
        
            training_index, testing_index = I[train_index], I[test_index]

            training_length, testing_length = L[train_index], L[test_index]

        
        print("processing: ", seed)
        
        output_weight = get_weight(config, pretrained_model)
        
        uio.save_pkfile(weight_path+"/"+config.weight_name, output_weight)
        
        train_state, test_state, train_index, test_index = execute(
            config, pretrained_model, 
            train_val_data, train_val_label, 
            testing_data, testing_label,
            training_index, testing_index,
            training_length, testing_length
        )
        
        output_data = {
            "training": train_state,
            "training_label": train_val_label,
            "training_index": train_index,
            "training_length": training_length,
            "testing": test_state,
            "testing_label": testing_label,
            "testing_index": test_index,
            "testing_length": testing_length
            
        }

        uio.save_pkfile(state_path+"/"+config.state_name, output_data)
        




