import os
import csv
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
import core.utils as cuils
import utils.io as uio

from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from core.loss import FocalLoss
from core.pytorchtools import EarlyStopping
from core.model import SPL
from configs.sel_config import Config
import mkl
mkl.set_num_threads(3)
torch.set_num_threads(3)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
        nn.init.constant_(m.bias.data, 0.0)
        print('Done:', classname)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)
        print('Done:', classname)
    elif classname.find('BatchNorm1d')!= -1:
        nn.init.constant(m.weight, 1)
        nn.init.constant(m.bias, 0)
        print('Done:', classname)


def execute(config, training_data, training_label, testing_data, testing_label, 
            training_length = None, training_index = None,
            testing_length = None, testing_index = None,
            val_X = None, val_Y=None, wandb=None):

    model = SPL(hidden_size = config.hidden_size)

    #model.apply(weights_init)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    #optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                       milestones=[20,40,60,80,120,160],
                       gamma=0.5,
                       last_epoch=-1)
    
    exponentials = cuils.exponentialDecay(config.epoch_size)

    
    early_stopping = EarlyStopping(patience=20, verbose=True)
        
    best_score = 0
    result_model = None
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
        
    for epoch in range(config.epoch_size):
        correct = 0
        model._epsilon = exponentials[epoch]
        model._REWARDS = 0
        loss_sum = 0
        loss_sum = 0
        model.train()

        len_t = int(len(training_data)/config.batch_size)
        s = np.arange(training_data.shape[0])
        np.random.shuffle(s)

        training_data = training_data[s]
        training_label = training_label[s]

        hpoint = []

        labels = []
        preds = []

        count = 0

        for idx in tqdm(range(len_t)):
            X = training_data[idx*config.batch_size:idx *
                              config.batch_size+config.batch_size]

            input_label = torch.from_numpy(
                training_label[idx*config.batch_size:idx*config.batch_size+config.batch_size]).cuda()

            _, input_label = input_label.max(dim=1)

            predictions, t = model(X)

            hpoint.append(t)

            loss, c, r, b, p = model.applyLoss(
                predictions, input_label, beta=config.beta )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            _, predicted = torch.max(predictions.data, 1)
            correct += (predicted == input_label.squeeze()
                        ).sum().cpu().detach().numpy()
            n_iter = epoch*len_t + idx
            #writer.add_scalar('train/loss', loss, n_iter)

            if (idx % 10 == 0 and wandb is not None):

                wandb.log({'training/loss': loss.cpu().detach().numpy(),
                           'training/classification_loss': c.cpu().detach().numpy(),
                           'training/regression_loss': r.cpu().detach().numpy(),
                           'training/baseline_loss': b.cpu().detach().numpy(),
                           'training/time_penalty_loss': p.cpu().detach().numpy()
                           })

        if (wandb is not None):

            wandb.log({'epoch': epoch,
                       'training/haulting_point': np.array(hpoint).sum()/len(hpoint)/config.batch_size,
                       'training/accuracy': correct/len(training_label)})

        print("epoch: ", epoch, "loss: ", loss.cpu().detach().numpy(), " accuracy: ", correct/len(training_label), "haulting point:", np.array(hpoint).sum()/len(hpoint)/config.batch_size, "classification loss: ", loss.cpu().detach().numpy(),
              c, r, b, p)

        ''' Validation '''
        if (val_X is not None):
            model.eval()
            correct = 0
            y_pred = []
            y_true = []
            y_tau = []
            rate_tau = []
            length_tau = []
            for idx, row in tqdm(enumerate(val_X)):

                X = val_X[idx:idx+1]
                input_label = torch.from_numpy(val_Y[idx:idx+1]).cuda()
                input_label_list = val_Y[idx:idx+1]
                _, input_label = input_label.max(dim=1)
                predictions, t = model(X)

                loss,c,r,b,p = model.applyLoss(predictions, input_label)

                valid_losses.append(loss.item())

                _, predicted = torch.max(predictions.data, 1)

                y_tau.append(t[0])
                length_tau.append(X[0].shape[0])
                rate_tau.append(t[0]/X[0].shape[0])

                for index, val in enumerate(predicted):
                    y_pred.append(val.cpu().detach().numpy())

                    if(input_label_list[0][val.cpu().detach().numpy()]):
                        correct += 1
                        y_true.append(val.cpu().detach().numpy())
                    else:
                        tr_lbl = input_label[index]

                        y_true.append(tr_lbl.cpu().numpy())
                        #print(val.cpu().detach().numpy(), input_label_list[index])

            y_tau = np.array(y_tau)

            val_earliness = np.mean(rate_tau)
            val_accuracy = correct/len(val_Y)
        
            if (wandb is not None):

                wandb.log({'epoch': epoch,
                           'validation/accuracy': val_accuracy,
                           'validation/earliness': val_earliness,
                           })
            print("epoch: ", epoch,"validation accuracy: ", val_accuracy, "validation earliness: ", val_earliness)

        ''' Testing '''
        model.eval()
        correct = 0
        y_pred = []
        y_true = []
        y_tau = []
        rate_tau = []
        length_tau = []
        for idx, row in tqdm(enumerate(testing_data)):

            X = testing_data[idx:idx+1]
            input_label = torch.from_numpy(testing_label[idx:idx+1]).cuda()
            input_label_list = testing_label[idx:idx+1]
            _, input_label = input_label.max(dim=1)
            predictions, t = model(X)

            loss,c,r,b,p = model.applyLoss(predictions, input_label)
            
            valid_losses.append(loss.item())
            
            _, predicted = torch.max(predictions.data, 1)

            y_tau.append(t[0])
            length_tau.append(X[0].shape[0])
            
            if(testing_length is None):
                rate_tau.append(t[0]/(X[0].shape[0]))
            else:
                tt_point = t[0]
                
                #print(tt_point)
                
                # 499 for snippet length 1000
                # 249 for snippet length 500
                
                earliness = (testing_index[idx][tt_point]+249) / testing_length[idx]

                rate_tau.append(earliness)
            
            for index, val in enumerate(predicted):
                y_pred.append(val.cpu().detach().numpy())

                if(input_label_list[0][val.cpu().detach().numpy()]):
                    correct += 1
                    y_true.append(val.cpu().detach().numpy())
                else:
                    tr_lbl = input_label[index]

                    y_true.append(tr_lbl.cpu().numpy())
                    #print(val.cpu().detach().numpy(), input_label_list[index])
        y_tau = np.array(y_tau)

        if (wandb is not None):

            wandb.log({'epoch': epoch,
                       'testing/accuracy': correct/len(testing_label),
                       'testing/haulting_point': np.mean(y_tau),
                       'testing/earliness': np.mean(rate_tau),
                       'testing/f1': np.mean(f1_score(y_true, y_pred, average=None)),
                       'testing/f2': fbeta_score(y_true, y_pred, average='weighted', beta=2),
                       'testing/recall': recall_score(y_true, y_pred, average='macro'),
                       'testing/precision': f1_score(y_true, y_pred, average='macro')
                       })

        acc = correct/len(testing_label)
        
        if (val_X is None):
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            train_losses = []
            valid_losses = []

            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")

                print("Stopping accuracy: ", acc,
                  "=", np.mean(rate_tau),'vs Best Model Acc:', best_score)

        print("Testing accuracy: ", acc,
              "hp:", np.mean(y_tau),
              "--", np.mean(length_tau),
              "=", np.mean(rate_tau))
        scheduler.step()

        '''
        if (correct/len(testing_label) > best_score):
            best_score = correct/len(testing_label)

            print('Best Model Acc:', best_score, epoch)
        '''    
        
        result_model = model
            
        model_path = os.path.join(config.root_dir,
                                  config.output_dir,
                                  config.model_dir,
                                  config.model_name,
                                  config.dataset_name,
                                  str(config.seed))
            
        torch.save(result_model.state_dict(), model_path+"/model"+str(epoch)+".pt")

        wandb.save(model_path+"/model"+str(epoch)+".pt")
            
        
    return result_model

if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    for seed in range(5,7):

        config = Config(message="schduler = [20,40,60,80,120,160]",
                        model_name="FIT-SPN-B0.0001-500SNP-FINALTEST-OGK",
                        hidden_size=256,
                        seed=seed, 
                        output_size=5,
                        beta=0.0001, 
                        batch_size=32, 
                        epoch_size=100,
                        dataset_name="ptbxl",
                       snippet_name = "christov_500_checkup.pickle")

        input_folder = os.path.join(config.root_dir,
                                    config.data_dir,
                                    config.dataset_name,
                                    config.snippet_dir,
                                    config.snippet_name)

        model_path = os.path.join(config.root_dir,
                                  config.output_dir,
                                  config.model_dir,
                                  config.model_name,
                                  config.dataset_name,
                                  str(config.seed))

        log_path = os.path.join(config.root_dir,
                                      config.output_dir,
                                      config.wandb_dir,
                                      config.model_name,
                                      config.dataset_name,
                                      str(seed))
        uio.check_folder(model_path)

        uio.check_folder(log_path)

        wandb.init(project="etscmoo", # earlytsc
                       entity='hugo',
                       dir = log_path,
                       name = config.model_name,
                       config=config,
                       reinit=True)

        
        if config.dataset_name == "ptbxl":
            
            print("Orignal K-Fold..........")
            
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

            sss = StratifiedKFold(
                n_splits=10, random_state=0)

            sss.get_n_splits(X, y)

            for index, (train_val_index, test_index) in enumerate(sss.split(X, y)):
                if(index is (seed-1)):
                    print("Runing:",seed)
                    break

            train_val_data, testing_data = X[train_val_index], X[test_index]

            train_val_label, testing_label = Y[train_val_index], Y[test_index]

            training_index, testing_index = I[train_val_index], I[test_index]

            training_length, testing_length = L[train_val_index], L[test_index]
       
        '''
        sss_2 = StratifiedShuffleSplit(
            n_splits=1, test_size=0.02, random_state=0)

        y = np.argmax(train_val_label, axis=1)
        
        print(y)
        
        sss_2.get_n_splits(train_val_data, y)
        
        for train_index, val_index in sss_2.split(train_val_data, y):

            training_data, val_data = train_val_data[train_index], train_val_data[val_index]

            training_label, val_label = train_val_label[train_index], train_val_label[val_index]         
        '''
        
        result_model = execute(config, 
                               train_val_data, 
                               train_val_label,
                               testing_data, 
                               testing_label, 
                               testing_index = testing_index,
                               testing_length = testing_length,
                               wandb=wandb)
        '''
        
        result_model = execute(config, 
                               training_data, 
                               training_label,
                               testing_data, 
                               testing_label, 
                               val_X = val_data,
                               val_Y = val_label,
                               wandb=wandb)
        '''
        
        torch.save(result_model.state_dict(), model_path+"/model.pt")

        wandb.save(model_path+"/model.pt")


