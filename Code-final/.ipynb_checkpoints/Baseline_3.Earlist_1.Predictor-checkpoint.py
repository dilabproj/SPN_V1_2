import os
import wandb
import torch
import random
import scipy.io
import numpy as np
import torch.nn as nn
import utils.io as uio
import core.utils as cuil

from tqdm import tqdm
from core.loss import FocalLoss
from core.model import EARLIEST
from configs.earlist_config import Config

from sklearn.metrics import f1_score, fbeta_score
from core.preprocessor import load_data, input_resizeing
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import mkl
mkl.set_num_threads(3)
torch.set_num_threads(3)


def execute(config, training_data, training_label, training_length, testing_data, testing_label, testing_length, wandb=None):

    loss_func = nn.CrossEntropyLoss()

    exponentials = cuil.exponentialDecay(config.epoch_size)

    model = EARLIEST(config.input_size, config.output_size, config.hidden_size,
                     N_LAYERS=config.layer_size, LAMBDA=config.lambda_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    batch_size = config.batch_size

    best_score = 0

    result_model = model

    for epoch in range(config.epoch_size):
        correct = 0
        earliness = 0.0
        time_rate = 0
        time_point = 0

        model._REWARDS = 0
        model._epsilon = exponentials[epoch]

        model.train()

        s = np.arange(training_data.shape[0])
        np.random.shuffle(s)
        training_data = training_data[s]
        training_label = training_label[s]

        for idx, row in tqdm(enumerate(training_data)):

            input_row = torch.unsqueeze(
                torch.from_numpy(row).float().squeeze(), 0)
            input_label = torch.from_numpy(training_label[idx:idx+1])

            input_length = training_length[idx *
                                           batch_size:idx*batch_size+batch_size]

            _, input_label = input_label.max(dim=1)

            predictions, t = model(input_row)

            _, predicted = torch.max(predictions.data, 1)

            correct += (predicted == input_label.squeeze()
                        ).sum().cpu().detach().numpy()

            time_rate += (t / training_length[idx])

            time_point += t

            optimizer.zero_grad()
            loss = model.applyLoss(predictions, input_label)
            loss.backward()
            optimizer.step()

            if (wandb is not None):
                wandb.log({'training/loss': loss.cpu().detach().numpy()})

        if (wandb is not None):

            wandb.log({'epoch': epoch,
                       'training/accuracy': correct/len(training_label),
                       'training/earliness': time_rate/len(training_label),
                       'training/haulting_point': time_point/len(training_label)
                       })

        model.eval()
        correctness = 0
        time_rate = 0
        time_point = 0

        y_pred = []
        y_true = []

        for idx, row in tqdm(enumerate(testing_data)):

            input_row = torch.unsqueeze(
                torch.from_numpy(row).float().squeeze(), 0)

            input_label = torch.from_numpy(testing_label[idx:idx+1])

            _, input_label = input_label.max(dim=1)

            input_label_list = testing_label[idx *
                                             batch_size:idx*batch_size+batch_size]

            predictions, t = model(input_row)

            _, predicted = torch.max(predictions.data, 1)

            time_rate += (t / testing_length[idx])

            time_point += t

            loss = model.applyLoss(predictions, input_label)

            _, predicted = torch.max(predictions.data, 1)

            correctness, y_pred, y_true = cuil.update_performance(predicted,
                                                                  correctness,
                                                                  y_pred,
                                                                  y_true,
                                                                  input_label,
                                                                  input_label_list)

        acc = correctness/len(testing_label)

        if (wandb is not None):

            wandb.log({'epoch': epoch,
                       'testing/loss': loss.cpu().detach().numpy(),
                       'testing/accuracy': acc,
                       'testing/earliness': time_rate/len(testing_label),
                       'testing/haulting_point': time_point/len(testing_label),
                       'testing/f1': f1_score(y_true, y_pred, average='weighted'),
                       'testing/f2': fbeta_score(y_true, y_pred, average='weighted', beta=2)
                       })

        if (acc > best_score):
            best_score = acc
            acc = round(acc, 2)
            print('Best Model Acc:', best_score)
            result_model = model

    return result_model


if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    for seed in range(1, 6):

        config = Config(message="20210513-og-kfold", 
                        series_ratio=1.0, 
                        seed=seed,
                        output_size=5,
                        dataset_name = "ptbxl",
                        total_length=5000)

        input_path = os.path.join(config.root_dir,
                                  config.data_dir,
                                  config.dataset_name) + "/"

        tmp_path = os.path.join(config.root_dir,
                                config.data_dir,
                                config.dataset_name,
                                config.tmp_dir)

        model_path = os.path.join(config.root_dir,
                                  config.output_dir,
                                  config.model_dir,
                                  config.model_name,
                                  config.dataset_name,
                                  str(seed))

        log_path = os.path.join(config.root_dir,
                                      config.output_dir,
                                      config.wandb_dir,
                                      config.model_name,
                                      config.dataset_name,
                                      str(seed))
        uio.check_folder(model_path)

        uio.check_folder(log_path)

        wandb.init(project="etscmoo",
                       entity='hugo',
                       dir = log_path,
                       name = config.model_name,
                       config=config,
                       reinit=True)

        if config.dataset_name == "ptbxl":
            input_data, raw_labels, labels, classes = uio.load_formmated_raw_data(
                input_path, "superdiagnostic", tmp_path)
            
            raw_length = uio.get_length(input_data)
            
            training_data = input_data[labels.strat_fold!=seed]
            testing_data = input_data[labels.strat_fold==seed]
                
            training_label = raw_labels[labels.strat_fold!=seed]
            testing_label = raw_labels[labels.strat_fold==seed]
                
            training_length = raw_length[labels.strat_fold!=seed]
            testing_length = raw_length[labels.strat_fold==seed]
        else:        
            input_data, raw_labels, labels, classes = uio.load_formmated_raw_data(
                input_path, "all", tmp_path)
                
            y = np.argmax(raw_labels, axis=1)

            raw_length = uio.get_length(input_data)

            sss = StratifiedKFold(n_splits=10, random_state=0)

            sss.get_n_splits(input_data, y)

            for index, (train_index, test_index) in enumerate(sss.split(input_data, y)):
                if(index is (seed-1)):
                    print("Runing:",seed)
                    break

            training_data, testing_data = input_data[train_index], input_data[test_index]
            training_label, testing_label = raw_labels[train_index], raw_labels[test_index]
            training_length, testing_length = raw_length[train_index], raw_length[test_index]

        result_model = execute(config, 
                               training_data, 
                               training_label, 
                               training_length,
                               testing_data, 
                               testing_label, 
                               testing_length, 
                               wandb=wandb)

        torch.save(result_model.state_dict(), model_path+"/model.pt")

        wandb.save(model_path+"/model.pt")

