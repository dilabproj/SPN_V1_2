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
from core.model import EARLIEST_NEW
from configs.earlist_config import Config

from sklearn.metrics import f1_score, fbeta_score
from core.preprocessor import load_data, input_resizeing
from sklearn.model_selection import StratifiedShuffleSplit
import mkl
mkl.set_num_threads(3)
torch.set_num_threads(3)


def execute(config, training_data, training_label, training_length, testing_data, testing_label, testing_length, wandb=None):

    loss_func = nn.CrossEntropyLoss()

    exponentials = cuil.exponentialDecay(config.epoch_size)
    
    model = EARLIEST_NEW(ninp = config.input_size, 
                         nclasses = config.output_size, 
                         nhid = config.hidden_size,
                         nlayers = config.layer_size, 
                         lam=config.lambda_val)

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

        len_t = int(len(training_data)/config.batch_size)
        s = np.arange(training_data.shape[0])
        np.random.shuffle(s)
        training_data = training_data[s]
        training_label = training_label[s]

        for idx in tqdm(range(len_t)):

            input_row = torch.from_numpy(training_data[idx*config.batch_size:idx *
                              config.batch_size+config.batch_size]).float()
            
            input_label = torch.from_numpy(
                training_label[idx*config.batch_size:idx*config.batch_size+config.batch_size])

            input_length = training_length[idx *
                                           batch_size:idx*batch_size+batch_size]

            _, input_label = input_label.max(dim=1)

            predictions, t = model(input_row)

            _, predicted = torch.max(predictions.data, 1)

            correct += (predicted == input_label.squeeze()
                        ).sum().cpu().detach().numpy()

            time_rate += t

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

            input_row = torch.from_numpy(testing_data[idx:idx+1]).float()

            input_label = torch.from_numpy(testing_label[idx:idx+1])

            _, input_label = input_label.max(dim=1)

            input_label_list = testing_label[idx:idx+1]

            predictions, t = model(input_row)
            
            _, predicted = torch.max(predictions.data, 1)

            time_rate += t

            time_point += t
            
            correctness, y_pred, y_true = cuil.update_performance(predicted,
                                                                  correctness,
                                                                  y_pred,
                                                                  y_true,
                                                                  input_label,
                                                                  input_label_list)

        acc = correctness/len(testing_label)

        if (wandb is not None):

            wandb.log({'epoch': epoch,
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    for seed in range(4, 5):

        config = Config(message="20210130", series_ratio=1.0,
                       batch_size = 128, seed=seed)

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

        input_data, raw_labels, labels, classes = uio.load_formmated_raw_data(
            input_path, "superdiagnostic", tmp_path)
        
        y = np.argmax(raw_labels, axis=1)

        raw_length = uio.get_length(input_data)

        sss = StratifiedShuffleSplit(
            n_splits=config.seed, test_size=0.1, random_state=0)
        sss.get_n_splits(input_data, y)

        for train_index, test_index in sss.split(input_data, y):
            training_data, testing_data = input_data[train_index], input_data[test_index]
            training_label, testing_label = raw_labels[train_index], raw_labels[test_index]
            training_length, testing_length = raw_length[train_index], raw_length[test_index]

        result_model = execute(config, training_data, training_label, training_length,
                               testing_data, testing_label, testing_length, wandb=wandb)

        torch.save(result_model.state_dict(), model_path+"/model.pt")

        wandb.save(model_path+"/model.pt")

