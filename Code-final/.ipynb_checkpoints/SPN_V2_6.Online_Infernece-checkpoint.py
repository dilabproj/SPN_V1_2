# -*- coding: utf-8 -*-
import os
import math
import time
import wandb
import pickle
import numpy as np
import utils.io as uio
import multiprocessing as mp
import state_generator as sg
import models.kg_policynet as kgpn
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from tqdm import tqdm
from biosppy.signals import ecg
from biosppy.signals import tools

from core.loss import SGD
from core.utils import *
from configs.online_config import Config
from models.snippet_cnnlstm import snippet_cnnlstm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


def is_better(a, b):

    if(a < b):
        return True
    else:
        return False


def is_worse(a, b):

    if(a > b):
        return True
    else:
        return False


def dominates(individualA, individualB):
    '''
        Dominates
        Retuen the domination relationship
    '''
    a = False
    for i in range(0, len(individualA)):
        if is_worse(individualA[i], individualB[i]):
            return False
        if is_better(individualA[i], individualB[i]):
            a = True
    return a


def fast_non_dominated_sort(objectives):
    '''
        Fast NonDominated Sort
        Return the non-dominated sets
    '''
    f = []
    f.append([])

    dominated_index = []

    dominated_count = []

    for i in objectives:
        dominated_index.append([])
        dominated_count.append(0)

    # calculate the dominated relationship between each individual

    for idx_A, individual_A in enumerate(objectives):
        # print(individualA.name)
        for idx_B, individual_B in enumerate(objectives):

            if dominates(individual_A, individual_B):
                dominated_index[idx_A].append(idx_B)

                # print("\t",individualB.name)
            elif dominates(individual_B, individual_A):
                dominated_count[idx_A] = dominated_count[idx_A] + 1

        if dominated_count[idx_A] == 0:  # if np == 0:
            f[0].append(idx_A)

        # calculate and sort the ranks relationship
    i = 0
    while(len(f[i]) != 0):
        h = []
        for p_idx in f[i]:
            for q in dominated_index[p_idx]:  # for q in sp:
                dominated_count[q] = dominated_count[q] - 1
                if dominated_count[q] == 0:
                    h.append(q)
                    # q.rank = i + 1 #Agrega el atributo de rank en el individuo...
        i = i + 1
        f.append(h)  # f[i] = h

    return f


def knee_mdd_selection(non_dominated_objectives):

    obj_max_array = []
    obj_min_array = []
    Lm = []

    dist = []

    for i in range(non_dominated_objectives.shape[0]):
        dist.append(0)

    for m in range(non_dominated_objectives.shape[1]):
        obj_max_array.append(np.max(non_dominated_objectives[:, m]))
        obj_min_array.append(np.min(non_dominated_objectives[:, m]))

        Lm.append(obj_max_array[m]-obj_min_array[m])

        for i, obj_A in enumerate(non_dominated_objectives):
            dist[i] = dist[i] + \
                ((obj_A[m] - obj_min_array[m]) / Lm[m])  # Lm[m]

            #print('idl:',i,'obj: ',m,'dist: ',( (obj_A[m] - obj_min_array[m]) / Lm[m] ))

    kneeval = np.max(dist)
    kneeidx = np.argmax(dist)
    for idx, val in enumerate(dist):
        if(val == 1):
            continue
        if(val < kneeval):
            kneeval = val
            kneeidx = idx

    return kneeidx


def boundary_individuals_selection(non_dominated_objectives):

    obj_max_array = []
    obj_min_array = []

    obj_max_idx = []
    obj_min_idx = []

    for i in range(non_dominated_objectives.shape[1]):
        # obj_max_array.append(np.max(self.non_dominated_objectives[:,i]))
        obj_min_array.append(np.min(non_dominated_objectives[:, i]))

        # obj_max_idx.append(np.argmax(self.non_dominated_objectives[:,i]))
        obj_min_idx.append(np.argmin(non_dominated_objectives[:, i]))

    return obj_min_idx


def feature_extractor(model, test_X, hidden):
    
    state, hidden = model.inference(test_X, hidden)
        
    return state, hidden


def show_vis(vis_idx, rpeak_list, pred_position, probs, flag, true, pred,diff):

    show_position = []
    show_position.append(0)
    for position in rpeak_list[:pred_position+1]:
        show_position.append(position)
    show_position.append(position+250)

    show_probs = []
    show_probs.append(0)
    for prob in probs.reshape(1,-1)[0]:
        show_probs.append(prob)
    show_probs.append(0)
    show_probs = np.array(show_probs)


    leads = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    fig, axes = plt.subplots(nrows=12, ncols=1,figsize=(20, 15))

    plt.xticks(fontsize=28)
    for i in range(12):
        plt.margins(0,0)
        plot_data = np.transpose(raw_testing_data[vis_idx])[i]
        min_val = np.min(plot_data)
        max_val = np.max(plot_data)
        
        plot_data = (plot_data-plot_data.min())/(plot_data.max()-plot_data.min())
            
        min_val = np.min(plot_data)
        max_val = np.max(plot_data)
        
        a_obs = pd.DataFrame(plot_data)
        
        ax = a_obs.plot(ax=axes[i])

        if(math.isnan(min_val) is False and math.isnan(max_val) is False):
            plt.ylim([min_val, max_val])
        ax.set_yticks([])
        y_axis = ax.axes.get_yaxis()
        x_axis = ax.axes.get_xaxis()
        if (i < 11): x_axis.set_visible(False)
        y_axis.set_label_text(leads[i], fontsize=28)
        ax.margins(x=0)
        ax.get_legend().remove()

            #print(np.sin(data), data)
        if(flag):
            ax.fill_between(x=show_position, y1= min_val, y2=show_probs*(max_val-min_val), facecolor='green', alpha=0.5)
        else:
            ax.fill_between(x=show_position, y1= min_val, y2=show_probs*(max_val-min_val), facecolor='red', alpha=0.5)
            # Add the patch to the Axes
            #ax.add_patch(rect)
            #print(idx,',',plot_data.shape[0] * label_time[0][1][idx])
        
        
    fig.savefig("/home/waue0920/hugo/eTSC/visual/ptbxl/"+str(vis_idx)+"_G_"+true+"_P_"+pred+"_D_"+str(diff)+".png", bbox_inches='tight')

    plt.close(fig) 

if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    
    config = Config(dataset_name = "ptbxl", 
                        output_size = 5, 
                        model_name = "CNNLSTM-500")
    
    seed = 1
    
    input_path = os.path.join(config.root_dir,
                              config.data_dir,
                              config.dataset_name) + "/"

    tmp_path = os.path.join(config.root_dir,
                            config.data_dir,
                            config.dataset_name,
                            config.tmp_dir)

    output_path = os.path.join(config.root_dir,
                               config.data_dir,
                               config.dataset_name,
                               config.snippet_dir)
    
    pretrained_model = os.path.join(config.root_dir,
                                        config.output_dir,
                                        config.model_dir,
                                        "CNNLSTM-500",
                                        "ptbxl",#"ICBEB-train_test_10fold",
                                        str(seed),
                                        config.pretrain_name)

    model_path = os.path.join(config.root_dir,
                                  config.output_dir,
                                  config.model_dir,
                                  config.model_name,
                                  config.dataset_name,
                                  str(seed))

    weight_path = os.path.join(config.root_dir,
                                        config.output_dir,
                                        config.weight_dir,
                                        config.model_name,
                                        config.dataset_name,
                                        str(seed))

    uio.check_folder(weight_path)

    sampling_rate = 500
    before_and_after = 1
    
    if(config.dataset_name == "ptbxl"):
        before_and_after = 0.5
    
        data, Y, labels, classes = uio.load_formmated_raw_data(input_path, "superdiagnostic", tmp_path)
        '''
        label_list = labels['superdiagnostic'].tolist()
        Y = []
        for idx_y, label in enumerate(label_list):
            first_label_idx = np.where(classes == label[0])[0][0]
            y = [0,0,0,0,0]
            y[first_label_idx] = 1
            Y.append(y)
        Y = np.array(Y)
        '''
        raw_testing_data = data[labels.strat_fold==seed]
        
        raw_testing_label = Y[labels.strat_fold==seed]        
    else:
        
        data, Y, labels, classes = uio.load_formmated_raw_data(input_path, "all", tmp_path)
        '''
        label_list = labels['all'].tolist()
        
        Y = []
        for idx_y, label in enumerate(label_list):
            first_label_idx = np.where(classes == label[0])[0][0]
            y = [0,0,0,0,0]
            y[first_label_idx] = 1
            Y.append(y)
            
        Y = np.array(Y)
        '''
        y = np.argmax(Y, axis=1)
        
        sss = StratifiedKFold(n_splits=10)

        sss.get_n_splits(data, y)

        for index, (train_val_index, test_index) in enumerate(sss.split(data, y)):
            if(index is (seed-1)):
                print("Runing:",seed)
                break

        raw_testing_data = data[test_index]

        raw_testing_label = Y[test_index]

    
    
    model = snippet_cnnlstm(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            hidden_output_size=config.hidden_output_size,
                            output_size=config.output_size,
                            core_model = config.model_name,
                            isCuda = True)

    model.load_state_dict(torch.load(pretrained_model))

    model.cuda()

    model.eval()
        
    agent = uio.load_pkfile(model_path+"/"+config.model_save_name)
      
    agent_net_shapes = agent["shapes"]
        
    agent_knee = agent["knee"]

    run_time = 0
    count = 0
    
    for idx, raw_testing_ecg in tqdm(enumerate(raw_testing_data)):

        #if(idx<387): continue
                
        #peak detection
        
        peaks = ecg.christov_segmenter(signal=raw_testing_ecg[:, 0], sampling_rate = 500)[0]
        
        ''' extract the snippet by filtering the anomorly data points'''
        
        check = False
        for peak in peaks:
            if(peak > 250 and peak<4750): 
                check =True
                break
            
        if((len(peaks)<=1) or (check is False)):
            la_peaks = ecg.christov_segmenter(signal=raw_testing_ecg[peaks[0]+250:, 0],
                                       sampling_rate = 500)[0]
            peaks = [(x+250) for x in la_peaks]

        hb = ecg.extract_heartbeats(signal=raw_testing_ecg,
                                    rpeaks=peaks,
                                    sampling_rate=500,
                                    before = before_and_after,
                                    after = before_and_after)

        rpeak_list = hb[1]
        
        raw_testing_ecg_corresponding_label = raw_testing_label[idx]

        #snippet generation
        
        input_snippet = np.array([hb[0]])
        
        #check the correctness
        '''
        for check_idx in range(rpeak_list.shape[0]):
            print(check_idx, raw_testing_ecg[rpeak_list[check_idx]-250:rpeak_list[check_idx]+250] == input_snippet[0][check_idx])
        
        print(rpeak_list)
                
        break
        '''
        #snippet classification
        start = time.time()
        
        #shape[1] is the snippet_size
        hidden = (torch.zeros(1, 1, 256).cuda(), torch.zeros(1, 1, 256).cuda())
        
        spatial_temporal_features = []
        
        for snp_idx in range(input_snippet.shape[1]):
            
            #print(input_snippet[:,:idx+1,:,:].shape)
            
            #feature extraction
            #batch, snippet_size, length, features
            spatial_temporal_state, hidden = feature_extractor(model, input_snippet[:,snp_idx,:,:], hidden)
            #print(spatial_temporal_features.shape)
            spatial_temporal_features.append(spatial_temporal_state)
            #early classification
            pred_label, pred_position, flag, probs = kgpn.inference(agent_net_shapes, agent_knee, spatial_temporal_features)
                    

        end = time.time()

        run_time += (end - start)
        
        pred_time_point = rpeak_list[pred_position]+250  # add a offset

        ground_truth = np.argmax(raw_testing_ecg_corresponding_label)

        earliness = pred_time_point/raw_testing_ecg.shape[0]
        
        if (raw_testing_ecg_corresponding_label[pred_label]): count+=1      
        
        print(" Raw testing ECG shape: ", raw_testing_ecg.shape,
              "\n Snippet shape: ", hb[0].shape, 
              "\n R-peaks", rpeak_list,
              "\n Prediction Snippet Position: ", pred_position,
              "\n Real Prediction Point: ", pred_time_point,   # add a offset
              "\n Prediction Earliness: ", earliness,
              "\n Predicted Label: ", pred_label,
              "\n Predicted Probs: ",probs,
              "\n Ground Truth Label:", ground_truth, raw_testing_ecg_corresponding_label,
              "\n Classified Result: ", raw_testing_ecg_corresponding_label[pred_label],
              "\n Real Time Accuracy: ", count/ (idx+1),
              "\n Inference Time: ",(end - start),
              "\n ###################################################")
        show_vis(idx, rpeak_list, pred_position, probs,raw_testing_ecg_corresponding_label[pred_label],classes[ground_truth],classes[pred_label],rpeak_list.shape[0]-pred_position)



run_time/2141

show_vis(1, rpeak_list, pred_position, probs)


def execute(config, models, train_X, train_Y, train_I, train_L, test_X, test_Y, test_I, test_L):

    # build Network
    print(models)
    
    net_shapes = models["shapes"]
    knee = models["knee"]
    
    boundary_1 = models["boundary_1"]
    boundary_2 = models["boundary_2"]
    boundary_3 = models["boundary_3"]
    
    #train_L = np.array([len(x)+1 for x in train_X])

    #test_L = np.array([len(x)+1 for x in test_X])

    knee_result = kgpn.get_reward(net_shapes, knee, test_X, test_Y, test_I, test_L, None, seed_and_id=None,real_value = True)[0]
    
    boundary_1_result = kgpn.get_reward(net_shapes, boundary_1, test_X, test_Y, test_I, test_L, None, seed_and_id=None, real_value = True)[0]
    
    boundary_2_result = kgpn.get_reward(net_shapes, boundary_2, test_X, test_Y, test_I, test_L, None, seed_and_id=None, real_value = True)[0]
    
    boundary_3_result = kgpn.get_reward(net_shapes, boundary_3, test_X, test_Y, test_I, test_L, None, seed_and_id=None, real_value = True)[0]
    
    print(
            'Knee: ',
            '| Accuracy: %.4f' % float(1-knee_result[0]),
            '| Earliness: %.4f' % float(knee_result[1]),
            '| Reward: %.4f' % float(knee_result[2]),
    )
    print(
            'Boundary_1: ',
            '| Accuracy: %.4f' % float(1-boundary_1_result[0]),
            '| Earliness: %.4f' % float(boundary_1_result[1]),
            '| Reward: %.4f' % float(boundary_1_result[2]),
    )
    print(
            'Boundary_2: ',
            '| Accuracy: %.4f' % float(1-boundary_2_result[0]),
            '| Earliness: %.4f' % float(boundary_2_result[1]),
            '| Reward: %.4f' % float(boundary_2_result[2]),
    )
    print(
            'Boundary_3: ',
            '| Accuracy: %.4f' % float(1-boundary_3_result[0]),
            '| Earliness: %.4f' % float(boundary_3_result[1]),
            '| Reward: %.4f' % float(boundary_3_result[2]),
    )
    
    
    acc=[round(float(1-knee_result[0]),3), 
         round(float(1-boundary_1_result[0]),3), 
         round(float(1-boundary_2_result[0]),3), 
         round(float(1-boundary_3_result[0]),3)]
    
    ear=[round(float(knee_result[1]),3), 
         round(float(boundary_1_result[1]),3), 
         round(float(boundary_2_result[1]),3), 
         round(float(boundary_3_result[1]),3)]
    
    return acc, ear



if __name__ == "__main__":

    all_acc = []
    
    all_ear = []
    
    for seed in range(1,2):

        config = Config(seed = seed,message ="20210125",
                        output_size=5, dataset_name="ptbxl",state_model_name="CNNLSTM-500")

        state_path = os.path.join(config.root_dir,
                                  config.output_dir,
                                  config.state_dir,
                                  config.state_model_name,
                                  config.dataset_name,
                                  str(seed))

        model_path = os.path.join(config.root_dir,
                                  config.output_dir,
                                  config.model_dir,
                                  "CNNLSTM-500",
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
        
        data_dict = uio.load_state_data(
            state_path+"/"+config.state_name)

        trainX = data_dict['training'] 
        trainY = data_dict['training_label'] 
        testX = data_dict['testing'] 
        testY = data_dict['testing_label']
        trainI = data_dict["training_index"]
        testI = data_dict["testing_index"]
        trainL = data_dict['training_length']
        testL = data_dict['testing_length']
                            
        models = uio.load_pkfile(model_path+"/"+config.model_save_name)
        
        acc, ear = execute(
            config, models, trainX, trainY, trainI, trainL, testX, testY, testI, testL)
        
        all_acc.append(acc)
        
        all_ear.append(ear)
        
    
    all_acc = np.array(all_acc)
    
    all_ear = np.array(all_ear)
    
    np.savetxt('ptb_acc.out', all_acc, fmt='%f')
    
    np.savetxt('ptb_ear.out', all_ear, fmt='%f')

# +

from thop import profile

model = snippet_cnnlstm(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            hidden_output_size=config.hidden_output_size,
                            output_size=config.output_size,
                            core_model = config.model_name,
                            isCuda = True)

model.cuda()
input = [torch.from_numpy(x).float().cuda() for x in input_snippet[:,0:1]]
macs, params = profile(model, inputs=(input, ))
macs
# -

from thop import clever_format
macs
macs, params = clever_format([macs, params], "%.3f")

macs

params


