import os
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
import utils.io as uio
import core.utils as cuil

from tqdm import tqdm
from core.model import MDDNN
from configs.mddnn_config import Config
from sklearn.metrics import f1_score, fbeta_score
from core.preprocessor import load_data, input_resizeing
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import mkl

mkl.set_num_threads(3)
torch.set_num_threads(3)


def execute(config, training_data, training_label, training_length, testing_data, testing_label, testing_length, wandb=None):

    freq_training_data = np.absolute(np.fft.fft(
        training_data, n=64, axis=2, norm="ortho"))
    freq_testing_data = np.absolute(np.fft.fft(
        testing_data, n=64, axis=2, norm="ortho"))

    loss_func = nn.CrossEntropyLoss()
    model = MDDNN(config.input_size, int(config.total_length *
                                         config.series_ratio), 16, config.output_size)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                       milestones=[20,40,60,80],
                       gamma=0.5,
                       last_epoch=-1)
    
    batch_size = config.batch_size

    best_score = 0

    result_model = model

    for epoch in range(config.epoch_size):
        correct = 0

        model.train()

        len_t = int(len(training_data)/batch_size)
        s = np.arange(training_data.shape[0])
        np.random.shuffle(s)
        training_data = training_data[s]
        freq_training_data = freq_training_data[s]
        training_label = training_label[s]

        for idx in tqdm(range(len_t)):

            input_row = torch.as_tensor(
                training_data[idx*batch_size:idx*batch_size+batch_size]).float().cuda()
            input_fft = torch.as_tensor(
                freq_training_data[idx*batch_size:idx*batch_size+batch_size]).float().cuda()
            input_label = torch.as_tensor(
                training_label[idx*batch_size:idx*batch_size+batch_size]).cuda()

            _, input_label = input_label.max(dim=1)

            predictions = model(input_row, input_fft)

            loss = loss_func(predictions, input_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(predictions.data, 1)

            correct += (predicted == input_label.squeeze()
                        ).sum().cpu().detach().numpy()

            if (idx % 10 == 0 and wandb is not None):

                wandb.log({'training/loss': loss.cpu().detach().numpy()})

        if (wandb is not None):

            wandb.log({'epoch': epoch,
                       'training/accuracy': correct/len(training_label)})

        print("epoch: ", epoch, " accuracy: ", correct / len(training_label))

        model.eval()
        correctness = 0
        test_loss_sum = 0
        y_pred = []
        y_true = []
        y_tau = []

        for idx in tqdm(range(len(testing_data))):
            input_row = torch.as_tensor(testing_data[idx:idx+1]).float().cuda()
            input_fft = torch.as_tensor(
                freq_testing_data[idx:idx+1]).float().cuda()

            input_label = torch.as_tensor(testing_label[idx:idx+1]).cuda()

            input_label_list = testing_label[idx:idx+1]

            _, input_label = input_label.max(dim=1)

            predictions = model(input_row, input_fft)

            loss = loss_func(predictions, input_label)
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
                       'testing/f1': f1_score(y_true, y_pred, average='weighted'),
                       'testing/f2': fbeta_score(y_true, y_pred, average='weighted', beta=2)
                       })

        print("Testing accuracy: ", acc, 'f1: ', f1_score(y_true, y_pred, average='weighted'),
              'f2: ', fbeta_score(y_true, y_pred, average='weighted', beta=2))

        
        if (acc > best_score):
            best_score = acc
            acc = round(acc, 2)
            print('Best Model Acc:', best_score)
        
        scheduler.step()
        result_model = model

    return result_model

if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    for seed in range(1, 4):

        for ratio in range(1, 21):

            series_ratio = ratio/20

            config = Config(message="20210516-kfold",
                            dataset_name = "ICBEB",
                            batch_size=256,
                            series_ratio=series_ratio,
                            epoch_size=100,
                            output_size=9,
                            total_length = 30000,
                            ratio=ratio,
                            seed=seed)

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
                                      str(seed),
                                      str(ratio))

            log_path = os.path.join(config.root_dir,
                                      config.output_dir,
                                      config.wandb_dir,
                                      config.model_name,
                                      config.dataset_name,
                                      str(seed),
                                      str(ratio))

            uio.check_folder(model_path)

            uio.check_folder(log_path)

            wandb.init(project="etscmoo",
                       entity='hugo',
                       dir = log_path,
                       name = config.model_name,
                       config=config,
                       reinit=True)
            #superdiagnostic
            data, raw_labels, labels, classes = uio.load_formmated_raw_data(
                input_path, "all", tmp_path)

            y = np.argmax(raw_labels, axis=1)
        
            raw_length = uio.get_length(data)

            input_data = uio.resize(
                data, config.total_length, config.series_ratio)

            if config.dataset_name == "ptbxl":
                training_data = input_data[labels.strat_fold!=seed]
                testing_data = input_data[labels.strat_fold==seed]
                
                training_label = raw_labels[labels.strat_fold!=seed]
                testing_label = raw_labels[labels.strat_fold==seed]
                
                training_length = raw_length[labels.strat_fold!=seed]
                testing_length = raw_length[labels.strat_fold==seed]
                
            else:
                '''
                sss = StratifiedShuffleSplit(
                    n_splits=config.seed, test_size=0.1, random_state=0)
                sss.get_n_splits(input_data, y)
                '''

                sss = StratifiedKFold(n_splits=10)

                sss.get_n_splits(input_data, y)

                for index, (train_index, test_index) in enumerate(sss.split(input_data, y)):
                    if(index is (seed-1)):
                        print("Runing:",seed)
                        break

                training_data, testing_data = input_data[train_index], input_data[test_index]
                training_label, testing_label = raw_labels[train_index], raw_labels[test_index]
                training_length, testing_length = raw_length[train_index], raw_length[test_index]

            result_model = execute(config, training_data, training_label, training_length,
                                   testing_data, testing_label, testing_length, wandb=wandb)

            torch.save(result_model.state_dict(), model_path+"/model.pt")

            wandb.save(model_path+"/model.pt")

