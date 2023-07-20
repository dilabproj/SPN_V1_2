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
#from tensorboardX import SummaryWriter
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


# if __name__ == "__main__":
#
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
#     cms = []
#     
#     results = []
#     
#     #save_index = uio.load_pkfile(config.root_dir+config.output_dir+"/save_index.pkl")  
#     
#     for seed in range(1,11):
#
#         config = Config(message="202103testing",model_name="FIL-SPN-FINAL-20406080",
#                         seed=seed, beta=0.00001,epoch_size=10,snippet_name="filter_116_binorm_1000.pickle")
#
#         input_folder = os.path.join(config.root_dir,
#                                     config.data_dir,
#                                     config.dataset_name,
#                                     config.snippet_dir,
#                                     config.snippet_name)
#
#         model_path = os.path.join(config.root_dir,
#                                   config.output_dir,
#                                   config.model_dir,
#                                   config.model_name,
#                                   config.dataset_name,
#                                   str(seed))
#
#         log_path = os.path.join(config.root_dir,
#                                       config.output_dir,
#                                       config.wandb_dir,
#                                       config.model_name,
#                                       config.dataset_name,
#                                       str(seed))
#         
#         idx_folder = os.path.join(config.root_dir,
#                                     config.data_dir,
#                                     config.dataset_name,
#                                     config.snippet_dir,
#                                     "filter_116_binorm_1000_backup.pickle")
#         
#         uio.check_folder(model_path)
#
#         uio.check_folder(log_path)
#
#         
#         X, Y, I, L = uio.load_snippet_data_with_il(idx_folder)
#         
#         
#         y = np.argmax(Y, axis=1)
#         
#         sss = StratifiedKFold(
#             n_splits=seed, test_size=0.10, random_state=0)
#
#         sss.get_n_splits(X, y)
#
#         print(I.shape, Y.shape)
#         X_1, Y_1, I_1, L_1 = uio.load_snippet_data_with_il(input_folder)
#         
#         print(I_1.shape, Y_1.shape)
#         for train_index, test_index in sss.split(X, y):
#             
#             all_index = set([x for x in range(6877)])
#             
#             #t_index = np.array([194,865,1263,1392,3032,4268 ,4768,6651])
#             t_index = np.array([1263])
#             
#             training_data, testing_data = X[train_index], X[test_index]
#
#             training_label, testing_label = Y[train_index], Y[test_index]
#
#             training_index, testing_index = I[train_index], I[test_index]
#
#             training_length, testing_length = L[train_index], L[test_index]
#             
#             
#             testing_data = np.concatenate((testing_data, X_1[t_index]), axis=0)
#             testing_label = np.concatenate((testing_label, Y_1[t_index]), axis=0)
#             testing_index = np.concatenate((testing_index, I_1[t_index]), axis=0)
#             testing_length = np.concatenate((testing_length, L_1[t_index]), axis=0)
#                 
#         pretrained = model_path+"/model.pt"
#         #pretrained = "/home/waue0920/hugo/eTSC/tmp"+"/model.pt"
#         ''''''
#         C,cm,performance,label_time = execute(config, training_data, training_label, training_index, training_length,
#                        testing_data, testing_label, testing_index, testing_length,
#                        pretrained = pretrained, wandb=None)
#         cms.append(cm)
#         for perf in performance:
#             results.append(perf)
#         

if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    cms = []
    
    results = []
    
    for seed in range(1, 11, 1):
        #FIT-SPN-MIN-TURE-Sche-TauE0.01E0.1-B0.0001
         
        
        config = Config(message="202103testing",model_name="FIT-SPN-E0.01-B0.0001-(2144)",hidden_size=256,
                        seed=seed, beta=0.00001,epoch_size=1,snippet_name="christov_1000_doubel_check.pickle",
                       dataset_name = "ICBEB", output_size = 9)
                
        '''
        config = Config(message="202103testing",model_name="FIT-SPN-B0.0001-500SNP-FINALTEST-OGK",hidden_size=256,
                        seed=seed, beta=0.00001,epoch_size=1,snippet_name="christov_500_checkup.pickle",
                       dataset_name = "ptbxl", output_size = 5)
        '''

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

        if config.dataset_name == "ptbxl":
            
            print("Orignal K-Fold..........")
            
            X, Y, I, L, info = uio.load_snippet_data_with_il_info(input_folder)
            
            training_data = X[info.strat_fold!=seed]
            testing_data = X[info.strat_fold==seed]
                        
            training_label = Y[info.strat_fold!=seed] 
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

            training_data, testing_data = X[train_val_index], X[test_index]

            training_label, testing_label = Y[train_val_index], Y[test_index]

            training_index, testing_index = I[train_val_index], I[test_index]

            training_length, testing_length = L[train_val_index], L[test_index]
        
        print(np.mean(training_length))        
        print(np.mean(testing_length))
        print(training_data[0].shape)
        
        '''
        for idx, raw in enumerate(training_data):
            print(raw.shape, len(I[idx]))
        '''    
        
        pretrained = model_path+"/model94.pt"
        #pretrained = "/home/waue0920/hugo/eTSC/tmp"+"/model.pt"
        #pretrained = model_path+"/model99.pt"
        C,cm,performance,label_time = execute(config, 
                                              training_data, 
                                              training_label, 
                                              training_index, 
                                              training_length,
                                              testing_data, 
                                              testing_label, 
                                              testing_index, 
                                              testing_length,
                                              pretrained = pretrained, 
                                              wandb=None)
        
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


output = np.array(arr[:-1])

output[1]

# +
import statistics

print(
statistics.mean(output[:,5]), statistics.stdev(output[:,5]),'\n',
statistics.mean(output[:,4]), statistics.stdev(output[:,4]),'\n',
statistics.mean(output[:,0]), statistics.stdev(output[:,0]),'\n',
statistics.mean(output[:,2]), statistics.stdev(output[:,2]),'\n',
statistics.mean(output[:,1]), statistics.stdev(output[:,1]),'\n',
statistics.mean(output[:,7]), statistics.stdev(output[:,7]),'\n',
statistics.mean(output[:,6]), statistics.stdev(output[:,6])
    )
# -

base = 500
precision_avg/base, recall_avg/base,  acc_avg/base,f_1_avg/base, ear_avg/base, hm_avg/base, run_time_avg/base



statistics.mean(precision_avg),statistics.stdev(arr)


from torch.distributions import Bernoulli,RelaxedBernoulli,Normal,Categorical
temperature = torch.FloatTensor([0.5])
probs = torch.FloatTensor([0.8,0.1])
m = Categorical(probs= probs)
m.sample()



sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
ss = sss.split(X, y)
for index, (train_index, test_index) in enumerate(sss.split(X, y)):
    print(index,np.sort(test_index))

sfk = StratifiedKFold(n_splits=10, random_state=1,shuffle=True)
ss = sfk.split(X, y)
for index, (train_index, test_index) in enumerate(sfk.split(X, y)):
    print(index,np.sort(test_index))

sfk = StratifiedKFold(n_splits=10, random_state=777,shuffle=True)
ss = sfk.split(X, y)
for index, (train_index, test_index) in enumerate(sfk.split(X, y)):
    print(index,np.sort(test_index))





testing_data[0].shape

remove_idx = [6651, 6604, 4792, 4768, 4268, 3032, 2130, 1425, 1392, 1263, 865, 194]


from tqdm import tqdm
from biosppy.signals import ecg
from biosppy.signals import tools
from configs.data_config import DataConfig
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
specific_array=[464]
import pandas as pd

# +
input_path = os.path.join(config.root_dir,
                              config.data_dir,
                              config.dataset_name) + "/"

tmp_path = os.path.join(config.root_dir,
                            config.data_dir,
                            config.dataset_name,
                            "tmp")

output_path = os.path.join(config.root_dir,
                               config.data_dir,
                               config.dataset_name,
                               config.snippet_dir)

print(input_path)
    #['diagnostic', 'subdiagnostic', 'superdiagnostic']
data, Y, labels, classes = uio.load_formmated_raw_data(
        input_path, "all", tmp_path)

'''
data = np.delete(data,remove_idx, axis=0)
Y = np.delete(Y,remove_idx, axis=0)
y = np.argmax(Y, axis=1)
'''

sss = StratifiedKFold(n_splits=10, random_state=0)

sss.get_n_splits(data, y)

for index, (train_index, test_index) in enumerate(sss.split(data, y)):
    if(index is (8)):
        print("Runing:",seed)
        break
        
raw_training_data = data[train_index]
raw_training_label = Y[train_index]
raw_testing_data = data[test_index]
raw_testing_label = Y[test_index]
    
# -
for i, iid in enumerate(test_index):
    if int(iid) == 6220:
        print(i)


test_index

# +
data[0].shape

data[0].T.shape
# -

# 194 -> (0,)
# 865 -> (0,)
# 1263 -> (0,)
# 1392 -> (0,)
# 3032 -> (0,)
# 4268 -> (0,)
# 4768 -> (0,)
# 6651 -> (0,)

for idx, sample in enumerate(testing_data):
    print(sample.shape, label_time[1][2][idx])



# +

draw_data = data[1914]

filter_row = tools.filter_signal(signal = draw_data, 
                                         sampling_rate=500,
                                         band='bandpass',
                                         frequency = [1,16],
                                         order = 2)

draw_data = filter_row[0]

peaks = ecg.gamboa_segmenter(signal=draw_data.T[0],sampling_rate = 500)[0]

hb = ecg.extract_heartbeats(signal=draw_data,
                                    rpeaks=peaks,
                                    sampling_rate=500,
                                    before=1,
                                    after=1)

rpeaks = hb[1]

ts = hb[0]


fig, axes = plt.subplots(nrows=12, ncols=1,figsize=(20, 10))
    #fig.suptitle(label_set[y_true[idx]], fontsize=12)
    #plt.xlabel('round',font1)
plt.xticks(fontsize=28)
for pp in peaks:
    plt.scatter(x=[pp-20, 0], y=[0, 200], c='r', s=40)
for i in range(12):
    plot_data = np.transpose(draw_data)[i]
    a_obs = pd.DataFrame(plot_data)
        #plt.plot(a_obs)
    ax = a_obs.plot(ax=axes[i])
    ax.set_yticks([])
    y_axis = ax.axes.get_yaxis()
    x_axis = ax.axes.get_xaxis()
    if (i < 11): x_axis.set_visible(False)
    ax.get_legend().remove()
# -



# +

draw_data = data[70]

peaks = ecg.christov_segmenter(signal=draw_data.T[0],sampling_rate = 500)[0]
print(peaks)



fig, axes = plt.subplots(nrows=12, ncols=1,figsize=(20, 10))
    #fig.suptitle(label_set[y_true[idx]], fontsize=12)
    #plt.xlabel('round',font1)
plt.xticks(fontsize=28)
for pp in peaks:
    plt.scatter(x=[pp, 0], y=[0, 200], c='r', s=40)
for i in range(12):
    plot_data = np.transpose(draw_data)[i]
    a_obs = pd.DataFrame(plot_data)
        #plt.plot(a_obs)
    ax = a_obs.plot(ax=axes[i])
    ax.set_yticks([])
    y_axis = ax.axes.get_yaxis()
    x_axis = ax.axes.get_xaxis()
    if (i < 11): x_axis.set_visible(False)
    ax.get_legend().remove()


# -

def get_median_filtered(signal, threshold=100):
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal



# +
peaks = ecg.christov_segmenter(signal=filter_row[:, 0],
                                           sampling_rate = 500)[0]
print(peaks,filter_row.shape)

fig, axes = plt.subplots(nrows=12, ncols=1,figsize=(20, 10))
    #fig.suptitle(label_set[y_true[idx]], fontsize=12)
    #plt.xlabel('round',font1)
plt.xticks(fontsize=28)
for i in range(12):
    plot_data = np.transpose(filter_row)[i]
    a_obs = pd.DataFrame(plot_data)
        #plt.plot(a_obs)
    ax = a_obs.plot(ax=axes[i])
    ax.set_yticks([])
    y_axis = ax.axes.get_yaxis()
    x_axis = ax.axes.get_xaxis()
    if (i < 11): x_axis.set_visible(False)
    ax.get_legend().remove()
# -

filter_row.shape







# for idx, l in enumerate(raw_testing_label):
#     print(l, testing_label[idx], (l == testing_label[idx]))

ccc = 0
rrr = 0
numccc = 0
for idx, timetime in enumerate(label_time[0][2]):
    
    if(testing_index[idx][timetime] < 1000):
        ccc+=1
        tmp = 0
        for point in testing_index[idx]:
            tmp = point
            if(tmp >1000):break
                
        rrr+= (tmp / testing_length[idx])
        numccc+=1
        #print(timetime, (testing_index[idx][timetime]-1), testing_length[idx])
    else:
        if(label_time[0][3][idx] > 2):
            print(timetime, (testing_index[idx][timetime]-1), testing_length[idx])
        rrr+= ((testing_index[idx][timetime]-1) / testing_length[idx])
        numccc+=1
print(ccc,rrr,rrr/numccc)

label_time[0][2]

idx = 7


label_time[0][1][idx], label_time[0][2][idx], testing_data[idx].shape, testing_length[idx]


haltpoint = label_time[0][2][idx]
(testing_index[idx][haltpoint]+999) / testing_length[idx]

Y

show_list = []
for idx,label in enumerate(Y):
    if(label[1] == 1 and data[idx].shape[0] <= 5500):
        show_list.append(int(idx))

data[15].shape

# +

label_set = ['1AVB', 'AFIB', 'CLBBB', 'CRBBB', 'NORM', 'PAC', 'STD_', 'STE_', 'VPC']
raw_testing_label[668]

# +


leads = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
idxxx = [668]
for idx in idxxx:
    
    fig, axes = plt.subplots(nrows=12, ncols=1,figsize=(20, 10))
    #fig.suptitle(label_set[y_true[idx]], fontsize=12)
    #plt.xlabel('round',font1)
    plt.xticks(fontsize=28)
    for i in range(12):
        plot_data = np.transpose(raw_testing_data[idx])[i]
        a_obs = pd.DataFrame(plot_data)
        #plt.plot(a_obs)
        ax = a_obs.plot(ax=axes[i])
        ax.set_yticks([])
        y_axis = ax.axes.get_yaxis()
        x_axis = ax.axes.get_xaxis()
        if (i < 11): x_axis.set_visible(False)
        y_axis.set_label_text(leads[i])
        ax.get_legend().remove()
        haltpoint = label_time[1][2][idx]
        position = testing_index[idx][haltpoint]
        rect = patches.Rectangle((0,-10),position+499, 30,linewidth=1,edgecolor='r',facecolor='g',alpha=0.2)
        #rect = patches.Rectangle((0,-10),999,30,linewidth=1,edgecolor='r',facecolor='r',alpha=0.2)

        # Add the patch to the Axes
        ax.add_patch(rect)
        #print(idx,',',plot_data.shape[0] * label_time[0][1][idx])
    
    #fig.savefig("/home/hugo/eTSC/visual/"+label_set[y_true[idx]]+"_"+str(idx)+".png")
    #plt.close(fig) 
# -

performance

results[1]

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

base = 100
precision_avg/base, recall_avg/base,  acc_avg/base,f_1_avg/base, ear_avg/base, hm_avg/base, run_time_avg/base

total = np.zeros((9,9))
nm = np.zeros((9,9))
np.set_printoptions(suppress=True)

label_set = ['1AVB', 'AFIB', 'CLBBB', 'CRBBB', 'NORM', 'PAC', 'STD_', 'STE_', 'VPC']




# import seaborn as sn
# import pandas as pd
# import matplotlib.pyplot as plt
# array = C
# df_cm = pd.DataFrame(array, index = [i for i in label_set],
#                   columns = [i for i in label_set])
# plt.figure(figsize = (10,7))
# ax = sn.heatmap(df_cm, annot=True,cmap =sn.cubehelix_palette(100, light=.95))
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom + 0.5, top - 0.5)

# ### Efficiency Calculation

# +

from thop import profile

model = SPL(hidden_size = 256)
input = testing_data[0:100]
print(input[0].shape)
macs, params = profile(model, inputs=(input, ))
# -

from thop import clever_format
macs/=100
macs, params = clever_format([macs, params], "%.3f")

macs

params













if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    for seed in range(5,6):

        config = Config(message="schduler = [20,40,60,80,120,160]",
                        model_name="FIT-SPN-B0.0005-LinePlot",
                        hidden_size=256,
                        seed=seed, 
                        beta=0.0005, 
                        batch_size=32, 
                        epoch_size=100,
                       snippet_name = "filter_116_binorm_1000.pickle")

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
       
        
        sss_2 = StratifiedShuffleSplit(
            n_splits=1, test_size=0.02, random_state=0)

        y = np.argmax(train_val_label, axis=1)
        
        '''
        result_model = execute(config, 
                               train_val_data, 
                               train_val_label,
                               testing_data, 
                               testing_label, 
                               testing_index = testing_index,
                               testing_length = testing_length,
                               wandb=None)
        '''


