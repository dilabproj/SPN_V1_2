import os
import time
import wandb
import pickle
import numpy as np
import utils.io as uio
import multiprocessing as mp
import state_generator as sg
import models.kg_policynet as kgpn
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import f1_score, fbeta_score

from tqdm import tqdm
from core.loss import SGD
from core.utils import *
from configs.kgea_config import Config


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


def train(X, Y, l, net_shapes, net_params, optimizer, utility, pool, individual_size, sigma):

    noise_seed = np.random.randint(
        0, 2 ** 32 - 1, size=individual_size, dtype=np.uint32).repeat(2)    # mirrored sampling

    pick_seed = np.random.randint(
        0, 3, size=individual_size, dtype=np.uint32).repeat(2)    # mirrored sampling

    # distribute training in parallel
    jobs = [pool.apply_async(kgpn.get_reward, (net_shapes, net_params[pick_seed[k_id]], X, Y, l, sigma,
                                               [noise_seed[k_id], k_id], )) for k_id in range(individual_size*2)]

    rewards = np.array([j.get()[0] for j in jobs])

    new_params = ([j.get()[1] for j in jobs])

    sort_rewards = rewards[:, 0]

    # rank kid id by reward
    kids_rank = np.argsort(sort_rewards)[::-1]

    non_domin_rank = fast_non_dominated_sort(rewards)

    non_dominted_set_idxs = non_domin_rank[0]

    if(len(non_dominted_set_idxs) < 3):
        non_dominted_set_idxs.append(non_domin_rank[1][0])

    non_dominated_objectives = np.array(
        [rewards[idx] for idx in non_dominted_set_idxs])

    knee_idx = knee_mdd_selection(non_dominated_objectives)

    # get the best individual, aka the knee individual

    best_params = new_params[non_dominted_set_idxs[knee_idx]]

    best_rewards = rewards[non_dominted_set_idxs[knee_idx]]

    boundary_idxs = boundary_individuals_selection(non_dominated_objectives)

    knee_boundary_idxs = [non_dominted_set_idxs[x] for x in boundary_idxs]

    knee_boundary_idxs.insert(0, non_dominted_set_idxs[knee_idx])

    cumulative_update = np.zeros_like(
        net_params[0])       # initialize update values

    knee_boundary_set = [new_params[x] for x in knee_boundary_idxs]

    for ui, k_id in enumerate(boundary_idxs):
        # reconstruct noise using seed
        np.random.seed(noise_seed[k_id])
        cumulative_update += utility[ui] * np.random.randn(net_params[0].size)

    gradients = optimizer.get_gradients(cumulative_update/((len(boundary_idxs))*sigma))

    for p in knee_boundary_set:
        p = p + gradients

    #best_params = net_params + gradients

    #best_params = best_params + LR/(3*SIGMA) * cumulative_update
    #best_params = net_params + LR/(2*N_KID*SIGMA) * cumulative_update

    return knee_boundary_set, best_rewards


def measurement(result):
    y_pred = result[0]
    y_list = result[1]
    predict_length = result[2]
    true_length = result[3]
    
    correct = 0
    y_true = []
    earliness = 0
    
    for idx, val in enumerate(y_pred):
        if(y_list[idx][val]):
            correct+=1
            y_true.append(val)
        else:
            y_true.append(np.argmax(y_list[idx], axis=0))
        
        earliness+= predict_length[idx]/true_length[idx]
        
    
    y_pred_oh = np.eye(9)[y_pred]
    
    recall_per_class = []
    for num_label in range(y_pred_oh.shape[1]):
        recall_per_class.append(recall_score(y_list[:,num_label], y_pred_oh[:,num_label], average=None)[1])
    
    recall_per_class = np.array(recall_per_class)
    
    y_true = np.array(y_true)
    
    acc = correct/len(y_pred)
    
    earliness = earliness/len(y_pred)
        
    f_1 = np.mean(f1_score(y_true, y_pred, average=None))
        
    f_2 = fbeta_score(y_true, y_pred, average='macro', beta=2)
        
    recall = recall_score(y_true, y_pred, average='macro')
            
    precision = precision_score(y_true, y_pred, average='macro')
        
    hm = (2*(1-earliness)*acc) / (1-earliness+acc)
    
    return (acc, earliness, f_1, f_2, recall, precision, hm, np.mean(recall_per_class))


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

    knee_result = kgpn.get_results(net_shapes, knee, test_X, test_Y, test_I, test_L, None, seed_and_id=None,real_value = True)
    
    boundary_1_result = kgpn.get_results(net_shapes, boundary_1, test_X, test_Y, test_I, test_L, None, seed_and_id=None, real_value = True)
    
    boundary_2_result = kgpn.get_results(net_shapes, boundary_2, test_X, test_Y, test_I, test_L, None, seed_and_id=None, real_value = True)
    
    boundary_3_result = kgpn.get_results(net_shapes, boundary_3, test_X, test_Y, test_I, test_L, None, seed_and_id=None, real_value = True)
    
    
    kn = measurement(knee_result)
    b1 = measurement(boundary_1_result)
    b2 = measurement(boundary_2_result)
    b3 = measurement(boundary_3_result)
    
    
    return (kn, b1,b2,b3)

if __name__ == "__main__":

    kn_array = []
    
    b1_array = []
    
    b2_array = []
    
    b3_array = []
    
    for seed in range(1, 11):

        config = Config(seed = seed,message ="20210125",
                        output_size=9,dataset_name="ICBEB",state_model_name="CNNLSTM", 
                        model_name = "KGEVO")

        state_path = os.path.join(config.root_dir,
                                  config.output_dir,
                                  config.state_dir,
                                  config.state_model_name,
                                  config.dataset_name,
                                  str(seed))

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
        
        kn, b1, b2, b3 = execute(
            config, models, trainX, trainY, trainI, trainL, testX, testY, testI, testL)
        
        kn_array.append(kn)
        
        b1_array.append(b1)

        b2_array.append(b2)

        b3_array.append(b3)
    
    #np.savetxt('ptb_acc.out', all_acc, fmt='%f')
    
    #np.savetxt('ptb_ear.out', all_ear, fmt='%f')

import statistics
kn_array = np.array(kn_array)
b1_array = np.array(b1_array)
b2_array = np.array(b2_array)
b3_array = np.array(b3_array)


def show_sta(result_array):
    print(
    "Precision:",statistics.mean(result_array[:,5]), statistics.stdev(result_array[:,5]),'\n',
    "Recall:",statistics.mean(result_array[:,4]), statistics.stdev(result_array[:,4]),'\n',
    "Accuracy:",statistics.mean(result_array[:,0]), statistics.stdev(result_array[:,0]),'\n',
    "F1:",statistics.mean(result_array[:,2]), statistics.stdev(result_array[:,2]),'\n',
    "Earliness:",statistics.mean(result_array[:,1]), statistics.stdev(result_array[:,1]),'\n',
    "HM:",statistics.mean(result_array[:,6]), statistics.stdev(result_array[:,6]),'\n',
    "RecallPerCLass:",statistics.mean(result_array[:,7]), statistics.stdev(result_array[:,7])
    )


show_sta(kn_array)

show_sta(b1_array)

show_sta(b2_array)

show_sta(b3_array)


