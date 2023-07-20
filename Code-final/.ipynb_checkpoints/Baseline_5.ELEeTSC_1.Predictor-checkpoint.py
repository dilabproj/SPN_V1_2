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
from core.model import DualOutputRNN
from configs.elects_config import Config

from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import recall_score, precision_score
from core.preprocessor import load_data, input_resizeing
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import mkl
mkl.set_num_threads(3)
torch.set_num_threads(3)


def execute(config, training_data, training_label, training_length, testing_data, testing_label, testing_length, 
            pretrained=None, wandb=None):

    model = DualOutputRNN(input_dim=config.input_size, nclasses=config.output_size, hidden_dims=config.hidden_size,
                          num_rnn_layers=config.layer_size, dropout=config.dropout, init_late=True)

    model.cuda()

    if (pretrained is not None):
         model.load_state_dict(torch.load(pretrained))
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    batch_size = config.batch_size

    best_score = 0

    loss_func = cuil.loss_early_reward

    result_model = model

    for epoch in range(config.epoch_size):

        correct = 0

        model.train()
        len_t = int(len(training_data)/batch_size)
        s = np.arange(training_data.shape[0])
        np.random.shuffle(s)
        training_data = training_data[s]
        training_label = training_label[s]
        earliness = 0.0
        for idx in tqdm(range(len_t)):

            input_row = torch.as_tensor(
                training_data[idx*batch_size:idx*batch_size+batch_size]).float().cuda()

            input_label = torch.as_tensor(
                training_label[idx*batch_size:idx*batch_size+batch_size]).cuda()

            input_length = training_length[idx *
                                           batch_size:idx*batch_size+batch_size]

            _, input_label = input_label.max(dim=1)

            logprobabilities, deltas, pts, budget = model.forward(input_row)

            loss, stats = loss_func(logprobabilities, pts, input_label, alpha=config.alpha)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            predicted, t_stops = model.predict(logprobabilities, deltas)

            correct += (predicted == input_label.squeeze()
                        ).sum().cpu().detach().numpy()

            earliness = cuil.update_earliness(t_stops,
                                              earliness,
                                              input_length,
                                              config.total_length,
                                              config.series_ratio)

            if (idx % 10 == 0 and wandb is not None):

                wandb.log({'training/loss': loss.cpu().detach().numpy()})
        torch.cuda.empty_cache()
        
        if (wandb is not None):

            wandb.log({'epoch': epoch,
                       'training/accuracy': correct/len(training_label),
                       'training/earliness': earliness/len(training_length)})

        print("epoch: ", epoch, " accuracy: ", correct/len(training_label))

        model.eval()
        y_pred = []
        y_true = []
        correctness = 0
        earliness = 0.0

        len_t = int(len(testing_data)/batch_size)

        for idx in tqdm(range(len_t)):
            input_row = torch.as_tensor(
                testing_data[idx*batch_size:idx*batch_size+batch_size]).float().cuda()

            input_label = torch.as_tensor(
                testing_label[idx*batch_size:idx*batch_size+batch_size]).cuda()

            input_length = testing_length[idx *
                                          batch_size:idx*batch_size+batch_size]

            _, input_label = input_label.max(dim=1)

            input_label_list = testing_label[idx *
                                             batch_size:idx*batch_size+batch_size]

            logprobabilities, deltas, pts, budget = model.forward(input_row)

            loss, stats = loss_func(logprobabilities, pts, input_label, alpha=config.alpha)

            predicted, t_stops = model.predict(logprobabilities, deltas)

            correctness, y_pred, y_true = cuil.update_performance(predicted,
                                                                  correctness,
                                                                  y_pred,
                                                                  y_true,
                                                                  input_label,
                                                                  input_label_list)

            earliness = cuil.update_earliness(t_stops,
                                              earliness,
                                              input_length,
                                              config.total_length,
                                              config.series_ratio)

        acc = correctness/len(testing_label)
        earliness = earliness/len(testing_label)
        torch.cuda.empty_cache()
        
        if (wandb is not None):

            wandb.log({'epoch': epoch,
                       'testing/loss': loss.cpu().detach().numpy(),
                       'testing/accuracy': acc,
                       'testing/earliness': earliness,
                       'testing/f1': np.mean(f1_score(y_true, y_pred, average=None)),
                       'testing/f2': fbeta_score(y_true, y_pred, average='weighted', beta=2),
                       'testing/recall': recall_score(y_true, y_pred, average='macro'),
                       'testing/precision': f1_score(y_true, y_pred, average='macro')
                       })

        print("Testing accuracy: ", acc, 'f1: ', f1_score(y_true, y_pred, average='weighted'),
              'f2: ', fbeta_score(y_true, y_pred, average='weighted', beta=2), 'earliness:', earliness)

        '''
        if (acc > best_score):
            best_score = acc
            acc = round(acc, 2)
            print('Best Model Acc:', best_score)
        '''
        
        if(epoch%10 ==0):
            wandb.save(model_path+"/"+str(config.alpha)+"_"+str(epoch)+"model-cont.pt")
            torch.save(result_model.state_dict(), model_path+"/"+str(config.alpha)+"_"+str(epoch)+"model-cont.pt")

        result_model = model

    return result_model


if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    
    for seed in range(1, 2):

        config = Config(message="20210525-og-kfold-save-cont", batch_size=49, 
                        total_length=30000, dataset_name="ICBEB",
                        series_ratio=0.5, alpha = 0.5, output_size=9,
                        hidden_size = 512, pretrained=True,
                        epoch_size=100, seed=seed)

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

        wandb.init(project="earlytsc",
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

        input_data = uio.resize(data, config.total_length, config.series_ratio)

        
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

                sss = StratifiedKFold( n_splits=10, random_state=0)

                sss.get_n_splits(input_data, y)

                for index, (train_index, test_index) in enumerate(sss.split(input_data, y)):
                    if(index is (seed-1)):
                        print("Runing:",seed)
                        break

                training_data, testing_data = input_data[train_index], input_data[test_index]
                training_label, testing_label = raw_labels[train_index], raw_labels[test_index]
                training_length, testing_length = raw_length[train_index], raw_length[test_index]
        
        #print(training_data.shape)
        
        if (config.pretrained):
            print("Continus training...")
            pretrained = model_path+"/0.5model.pt"
        else:
            print("Original training...")
            pretrained = None
            
        result_model = execute(config, training_data, training_label, training_length,
                               testing_data, testing_label, testing_length, 
                               pretrained = pretrained, wandb=wandb)

        if (config.pretrained):
            wandb.save(model_path+"/"+str(config.alpha)+"model-cont.pt")
            torch.save(result_model.state_dict(), model_path+"/"+str(config.alpha)+"model-cont.pt")
        else:
            wandb.save(model_path+"/"+str(config.alpha)+"model.pt")
            torch.save(result_model.state_dict(), model_path+"/"+str(config.alpha)+"model.pt")


