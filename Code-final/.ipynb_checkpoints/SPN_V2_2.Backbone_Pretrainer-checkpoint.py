import os
import torch
import wandb
import random
import numpy as np
import torch.nn as nn
import utils.io as uio

from tqdm import tqdm
from sklearn.metrics import f1_score, fbeta_score
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from core.loss import FocalLoss
from configs.backbone_config import Config
from models.snippet_cnnlstm import snippet_cnnlstm
import mkl
mkl.set_num_threads(3)
torch.set_num_threads(3)


def execute(config, train_X, train_Y, test_X, test_Y, val_X = None, val_Y=None, wandb = None):

    model = snippet_cnnlstm(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            hidden_output_size=config.hidden_output_size,
                            output_size=config.output_size,
                            core_model = config.model_name)
    
    model.cuda()

    criterion = FocalLoss(gamma=2)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                       milestones=[20],
                       gamma=0.1,
                       last_epoch=-1)

    min_tr_thld = config.training_performance_thr

    min_te_thld = config.testing_performance_thr

    result_model = None

    for e in range(config.epoch_size):

        model, loss, training_accuracy = fit(
            model, config.batch_size, train_X, train_Y, criterion, optimizer, scheduler, wandb)

        if (val_X is not None):
            validation_accuracy = inference(model, val_X, val_Y)
        
        testing_accuracy = inference(model, test_X, test_Y)

        scheduler.step()

        if (wandb is not None):
            
            if (val_X is None):
                wandb.log({"epoch": e,
                       "training/accuracy": training_accuracy,
                      "testing/accuracy": testing_accuracy})
            else:
                wandb.log({"epoch": e,
                       "training/accuracy": training_accuracy,
                       "validation/accuracy": validation_accuracy,
                      "testing/accuracy": testing_accuracy})
                
        if (val_X is None):
            print("Epoch: ", e, 
                  "Training Accuracy: ", training_accuracy,
                  "Testing Accuracy: ", testing_accuracy,
                  'optimizer: ',optimizer)
        else:
            print("Epoch: ", e, 
                  "Training Accuracy: ", training_accuracy,
                  "Validation Accuracy: ", validation_accuracy,
                  "Testing Accuracy: ", testing_accuracy,
                  'optimizer: ',optimizer)
            

        if (val_X is None):
            if(training_accuracy >= min_tr_thld):
                min_tr_thld = training_accuracy
                result_model = model
                print("Change Model -> Training Accuracy: ", training_accuracy, 
                     "Testing Accuracy: ", testing_accuracy)
        else:
            if(validation_accuracy >= min_tr_thld):
                min_te_thld = validation_accuracy
                result_model = model
                print("Change Model -> Training Accuracy: ", training_accuracy, 
                  "Validation Accuracy: ", validation_accuracy,
                 "Testing Accuracy: ", testing_accuracy)
            
    return result_model 

def fit(model, step, train_X, train_Y, criterion, optimizer, scheduler, wandb = None):

    model.train()

    correct = 0

    sample_size = int(len(train_X)/step)

    random_sample = np.arange(train_X.shape[0])

    np.random.shuffle(random_sample)

    training_data = train_X[random_sample]
    training_label = train_Y[random_sample]

    for idx in tqdm(range(sample_size)):

        X = training_data[idx*step:idx*step+step]

        input_X = [torch.from_numpy(x).float().cuda() for x in X]

        predictions = model(input_X)

        _, predicted = torch.max(predictions.data, 1)

        input_label = []

        for ldx, lb in enumerate(predicted.cpu().detach().numpy()):
            if(training_label[idx*step:idx*step+step][ldx][lb]):
                input_label.append(lb)
            else:
                input_label.append(np.argmax(
                    training_label[idx*step:idx*step+step][ldx], axis=None))

        input_label = torch.tensor(input_label).cuda()

        loss = criterion(predictions, input_label)

        if (wandb is not None):
            
            wandb.log({"training/loss": loss})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(predictions.data, 1)

        correct += (predicted == input_label.squeeze()
                    ).sum().cpu().detach().numpy()

    return model, loss, correct/len(training_label)


def inference(model, test_X, test_Y):

    model.eval()

    correct = 0

    y_pred = []

    y_true = []

    for idx, row in tqdm(enumerate(test_X)):

        X = test_X[idx:idx+1]

        input_X = [torch.from_numpy(x).float().cuda() for x in X]

        input_label = torch.from_numpy(test_Y[idx:idx+1]).cuda()

        input_label_list = test_Y[idx:idx+1]

        predictions = model(input_X)

        _, predicted = torch.max(predictions.data, 1)

        for index, val in enumerate(predicted):
            y_pred.append(val.cpu().detach().numpy())
            if(input_label_list[0][val.cpu().detach().numpy()]):
                correct += 1
                y_true.append(val.cpu().detach().numpy())
            else:
                y_true.append(input_label[index].cpu().numpy())

    return correct/len(test_Y)


if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    
    for seed in range(10, 11):

        config = Config(message="20210518-Original-K-fold-[step20]",epoch_size=50, model_name="CNNLSTM-500" ,
                        dataset_name = "ptbxl", output_size = 5,
                        seed=seed, snippet_name = "christov_500_checkup.pickle")

        input_folder = os.path.join(config.root_dir,
                                config.data_dir,
                                config.dataset_name,
                                config.snippet_dir,
                                config.snippet_name)

        pretrained_model_path = os.path.join(config.root_dir,
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
        
        wandb.init(project="etscmoo",
                       entity='hugo',
                       dir = log_path,
                       name = config.model_name,
                       config=config,
                       reinit=True)

        uio.check_folder(pretrained_model_path)

        uio.check_folder(log_path)

        if config.dataset_name == "ptbxl":
            
            X, Y, I, L, info = uio.load_snippet_data_with_il_info(input_folder)
            
            train_val_data = X[info.strat_fold!=seed]
            testing_data = X[info.strat_fold==seed]
            
            train_val_label = Y[info.strat_fold!=seed] 
            testing_label = Y[info.strat_fold==seed]
            
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
        
        tvs = StratifiedShuffleSplit(
            n_splits=1, test_size=0.1, random_state=0)

        y = np.argmax(train_val_label, axis=1)
                
        tvs.get_n_splits(train_val_data, y)
        
        for train_index, val_index in tvs.split(train_val_data, y):

            training_data, val_data = train_val_data[train_index], train_val_data[val_index]

            training_label, val_label = train_val_label[train_index], train_val_label[val_index]
                
                
        '''
        result_model = execute(config, 
                               training_data, 
                               training_label, 
                               testing_data, 
                               testing_label, 
                               val_X = val_data, 
                               val_Y = val_label, 
                               wandb = wandb)
        '''
        
        result_model = execute(config, 
                               train_val_data, 
                               train_val_label, 
                               testing_data, 
                               testing_label, 
                               wandb = wandb)
                
        torch.save(result_model.state_dict(), pretrained_model_path+"/pretrained.pt")
        
        wandb.save(pretrained_model_path+"/pretrained.pt")

a = [5,2,3,1,0,6,7]
sorted_stop_probs = np.sort(softmax_stop_probs[:-1])
