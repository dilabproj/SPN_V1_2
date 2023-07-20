import os
import csv
import time
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
import core.utils as cuils
import utils.io as uio
import utils.utils as uu

from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import f1_score, fbeta_score
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from core.loss import FocalLoss
from core.model import SPL
from configs.sel_config import Config
#import mkl
#mkl.set_num_threads(3)
torch.set_num_threads(3)

from sklearn.metrics import confusion_matrix


def execute(config, training_data, training_label, training_index, training_length,
            testing_data, testing_label, testing_index, testing_length, pretrained = None, wandb=None):
    
    model = SPL(hidden_size = config.hidden_size)

    if (pretrained is not None):
        model.load_state_dict(torch.load(pretrained))
    
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    exponentials = cuils.exponentialDecay(config.epoch_size)

    best_score = 0
    
    result_model = None
    
    best_ear = 1
    
    best_acc = 0
    
    performance = [1]
    
    results = [1]
    
    for epoch in range(config.epoch_size):
        correct = 0
        model._REWARDS = 0
        model._epsilon = exponentials[epoch]
        
        model.eval()
        correct = 0
        y_pred = []
        y_true = []
        y_tau = []
        rate_tau = []
        length_tau = []
        true_count = []
        run_time = 0
        for idx, row in tqdm(enumerate(testing_data)):
            
            start = time.time()

            X = testing_data[idx:idx+1]
            input_label = torch.from_numpy(testing_label[idx:idx+1]).cuda()
            input_label_list = testing_label[idx:idx+1]

            predictions, t = model(X)

            #loss,c,r,b,p = model.applyLoss(predictions, input_label)

            _, predicted = torch.max(predictions.data, 1)

            #print(t[0],len(testing_index[idx]),X[0].shape)
            
            end = time.time()
            
            run_time += (end - start)
            
            length_tau.append(X[0].shape[0])
            
            y_tau.append(t[0])
            
            if(testing_length is None):
                rate_tau.append(t[0]/(X[0].shape[0]+1))
            else:
                tt_point = t[0]
                
                #print(tt_point)
                
                earliness = (testing_index[idx][tt_point]+499) / testing_length[idx]

                rate_tau.append(earliness)
            
            for index, val in enumerate(predicted):
                y_pred.append(val.cpu().detach().numpy())
                
                if(input_label_list[0][val.cpu().detach().numpy()]):
                    correct += 1
                    true_count.append(1)
                    #print("ID: ",idx,"time:",t,"rate:",t[0]/X[0].shape[0],val.cpu().detach().numpy(),input_label_list[0])
                    
                    y_true.append(val.cpu().detach().numpy())
                else:
                    true_count.append(0)
                    _, tr_lbl = input_label[index].max(dim=0)
                    #print(val.cpu().detach().numpy(),tr_lbl,input_label[index])
                    y_true.append(tr_lbl.cpu().numpy())
                    #print(val.cpu().detach().numpy(), input_label_list[index])
            
        y_tau = np.array(y_tau)

        if (wandb is not None):

            wandb.log({'epoch': epoch,
                       'testing/accuracy': correct/len(testing_label),
                       'testing/haulting_point': np.mean(y_tau),
                       'testing/earliness': np.mean(rate_tau),
                       'testing/f1': f1_score(y_true, y_pred, average='macro'),
                       'testing/f2': fbeta_score(y_true, y_pred, average='weighted', beta=2)
                       })

        acc = correct/len(testing_label)
        
        run_time = run_time/len(testing_label)
        
        earliness = np.mean(rate_tau)
        
        f_1 = np.mean(f1_score(y_true, y_pred, average=None))
        
        f_2 = fbeta_score(y_true, y_pred, average='macro', beta=2)
        
        recall = recall_score(y_true, y_pred, average='macro')
        
        precision = precision_score(y_true, y_pred, average='macro')
        
        hm = (2*(1-earliness)*acc) / (1-earliness+acc)
        
        print(run_time)
        
        print("Iter: (",epoch,
              ") Accuracy: ", acc,
              "Earliness: ", np.mean(rate_tau),
              "F1-Weighted: ", np.mean(f1_score(y_true, y_pred, average=None)),
              "F2: ",fbeta_score(y_true, y_pred, average='macro', beta=2),
              "Recall: ",recall_score(y_true, y_pred, average='macro'),
              "Precision: ",precision_score(y_true, y_pred, average='macro'),
             )
        cm = confusion_matrix(y_true, y_pred)
        C = np.around(cm / cm.astype(np.float).sum(axis=0),2)
        
        '''
        if(best_acc < acc):
            best_ear = earliness
            best_acc = acc
            results[0] = (y_pred,rate_tau,y_tau,true_count)
            performance[0] = (acc, earliness, f_1, f_2, recall, precision, run_time, hm)
        '''
        results.append((y_pred,rate_tau,y_tau,true_count))
        performance.append((acc, earliness, f_1, f_2, recall, precision, run_time, hm))
        
    return C,cm,performance,results


seed = 1

# ### Raw Data Input

# +
from configs.data_config import DataConfig

raw_data_config = DataConfig(dataset_name = "ICBEB", segmenter = "christov")

raw_input_path = os.path.join(raw_data_config.root_dir,
                              raw_data_config.data_dir,
                              raw_data_config.dataset_name) + "/"

raw_tmp_path = os.path.join(raw_data_config.root_dir,
                            raw_data_config.data_dir,
                            raw_data_config.dataset_name,
                            raw_data_config.tmp_dir)

raw_data, raw_Y, raw_labels, classes = uio.load_formmated_raw_data(raw_input_path, "all", raw_tmp_path)

# +
one_label_y = np.argmax(raw_Y, axis=1)
        
raw_sss = StratifiedKFold(n_splits=10, random_state=0)

raw_sss.get_n_splits(raw_data, one_label_y)

raw_train_index_list = []
raw_test_index_list = []

for index, (train_index, test_index) in enumerate(raw_sss.split(raw_data, one_label_y)):
    raw_train_index_list.append(train_index)
    
    raw_test_index_list.append(test_index)

raw_test_index = raw_test_index_list[seed-1]

raw_testing_data = raw_data[raw_test_index]

raw_testing_label = raw_Y[raw_test_index]

# +
from configs.sel_config import Config

config = Config(model_name="FIT-SPN-E0.01-B0.0001-(2144)",
                beta=0.00001,
                epoch_size=1,
                snippet_name="filter_116_binorm_1000.pickle")

input_folder = os.path.join(config.root_dir,
                                    config.data_dir,
                                    config.dataset_name,
                                    config.snippet_dir,
                                    config.snippet_name)

X, Y, I, L = uio.load_snippet_data_with_il(input_folder)

# +
y = np.argmax(Y, axis=1)
        
snippet_sss = StratifiedKFold(n_splits=10, random_state=0)

snippet_sss.get_n_splits(X, y)

snippet_train_index_list = []
snippet_test_index_list = []

for index, (train_index, test_index) in enumerate(snippet_sss.split(X, y)):
        
    snippet_train_index_list.append(train_index)
    
    snippet_test_index_list.append(test_index)
    
test_index = snippet_test_index_list[seed-1]

snippet_testing_data = X[test_index]

snippet_testing_label = Y[test_index]
            
snippet_testing_index = I[test_index]
            
snippet_testing_length = L[test_index]


# +
model_path = os.path.join(config.root_dir,
                          config.output_dir,
                          config.model_dir,
                          config.model_name,
                          config.dataset_name,
                          str(seed))

model = SPL(hidden_size = config.hidden_size)

print(model_path)

model.load_state_dict(torch.load(model_path+"/model-best.pt"))
    
model.cuda()

model.eval()

# +
from tqdm import tqdm
from biosppy.signals import ecg
from biosppy.signals import tools

count = 0

run_time = 0

avg_earliness = 0

for idx, raw_testing_ecg in enumerate(raw_testing_data):
    
    start = time.time()
    
    tmp_norm = tools.normalize(raw_testing_ecg, ddof=2)['signal']
        
    peaks = ecg.christov_segmenter(signal=tmp_norm[:, 0], sampling_rate = 500)[0]
    
    if(len(peaks)<=1):
        la_peaks = ecg.christov_segmenter(signal=tmp_norm[peaks[0]+500:, 0],
                                           sampling_rate = 500)[0]
        peaks = [(x+500) for x in la_peaks]
    
    hb = ecg.extract_heartbeats(signal=raw_testing_ecg,
                                rpeaks=peaks,
                                sampling_rate=500,
                                before=1,
                                after=1)
    
    rpeak_list = hb[1]
    
    raw_testing_ecg_corresponding_label = raw_testing_label[idx]
    
    input_snippet = np.array([hb[0]])
    
    predictions, t = model(input_snippet)
        
    end = time.time()
    
    run_time += (end - start)
        
    _, predicted = torch.max(predictions.data, 1)
    
    pred_position = t[0]
    
    pred_time_point = rpeak_list[pred_position]+499  # add a offset
    
    earliness = pred_time_point/raw_testing_ecg.shape[0]
    
    avg_earliness += earliness
    
    pred_label = predicted.cpu().numpy()[0]
    
    ground_truth = np.argmax(raw_testing_ecg_corresponding_label)
    
    if (raw_testing_ecg_corresponding_label[pred_label]): count+=1
    
    
    print(" Raw testing ECG shape: ", raw_testing_ecg.shape,
          "\n Snippet shape: ", hb[0].shape, 
          "\n R-peaks", rpeak_list,
          "\n Prediction Snippet Position: ", pred_position,
          "\n Real Prediction Point: ", pred_time_point,   # add a offset
          "\n Prediction Earliness: ", earliness,
          "\n Predicted Label:", pred_label,
          "\n Ground Truth Label:", ground_truth,
          "\n Classified Result: ", pred_label == ground_truth,
          "\n Real Time Accuracy: ", count/ raw_testing_data.shape[0],
          "\n Inference Time: ",(end - start),
          "\n ###################################################")
       
    
print(count/raw_testing_data.shape[0])
    
print(avg_earliness/raw_testing_data.shape[0])
    
print(run_time/raw_testing_data.shape[0])
# -

count, raw_testing_data.shape

if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cms = []
    
    results = []
    
    for seed in range(1, 2,1):
        #FIT-SPN-MIN-TURE-Sche-TauE0.01E0.1-B0.0001
        config = Config(message="202103testing",model_name="FIT-SPN-E0.01-B0.0001-(2144)",hidden_size=256,
                        seed=seed, beta=0.00001,epoch_size=1,snippet_name="filter_116_binorm_1000.pickle")

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
                                  str(seed))

        log_path = os.path.join(config.root_dir,
                                      config.output_dir,
                                      config.wandb_dir,
                                      config.model_name,
                                      config.dataset_name,
                                      str(seed))
        uio.check_folder(model_path)

        uio.check_folder(log_path)

        X, Y, I, L = uio.load_snippet_data_with_il(input_folder)
        
        y = np.argmax(Y, axis=1)
        
        sss = StratifiedKFold(
            n_splits=10, random_state=0)

        sss.get_n_splits(X, y)
        
        for index, (train_index, test_index) in enumerate(sss.split(X, y)):
            if(index is (seed-1)):
                print("Runing:",seed)
                break

        print(I.shape, Y.shape)
        
        training_data, testing_data = X[train_index], X[test_index]

        training_label, testing_label = Y[train_index], Y[test_index]
            
        training_index, testing_index = I[train_index], I[test_index]
            
        training_length, testing_length = L[train_index], L[test_index]
        
        print(np.mean(training_length))        
        print(np.mean(testing_length))
        
        '''
        for idx, raw in enumerate(training_data):
            print(raw.shape, len(I[idx]))
        '''    
        
        pretrained = model_path+"/model.pt"
        #pretrained = "/home/waue0920/hugo/eTSC/tmp"+"/model.pt"
        
        C,cm,performance,label_time = execute(config, training_data, training_label, training_index, training_length,
                       testing_data, testing_label, testing_index, testing_length,
                       pretrained = pretrained, wandb=None)
        
        cms.append(cm)
        for perf in performance:
            results.append(perf)


def exponentialDecay(N):
    tau = 1 
    tmax = 4 
    t = np.linspace(0, tmax, N)
    y = np.exp(-t/tau)
    y = torch.FloatTensor(y)
    return y/10.


exponentialDecay(100)

import statistics
bakcup =results


# +

arr = [0,0,0,0,0,0,0,0,0,0]
tmp = []

best_acc = 0
count=-1
idx = 0
for pre in results:
    if pre is 1:
        count+=1
        best_acc = 0
        continue
    if(best_acc < pre[idx]):
        best_acc = pre[idx]
        arr[count] = pre

# +
acc_avg = 0

ear_avg = 0

f_1_avg = 0

f_2_avg = 0

recall_avg = 0

precision_avg = 0

run_time_avg = 0

hm_avg = 0

for epp in results:
    
    if epp is 1: continue
    
    acc_avg += epp[0]
    
    ear_avg += epp[1]
    
    f_1_avg += epp[2]

    f_2_avg += epp[3]

    recall_avg += epp[4]

    precision_avg += epp[5]
    
    run_time_avg += epp[6]
    
    hm_avg += epp[7]
# -

test_std = []
for re in results:
    if(re is 1):
        continue
    test_std.append(re)

test_std = np.array(test_std)
statistics.mean(test_std[:,0]),statistics.stdev(test_std[:,0])


output = np.array(arr)
