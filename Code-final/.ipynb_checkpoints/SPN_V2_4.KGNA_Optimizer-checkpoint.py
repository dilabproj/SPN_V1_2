import os
import time
import wandb
import pickle
import numpy as np
import utils.io as uio
import multiprocessing as mp
import state_generator as sg
import models.kg_policynet as kgpn

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


def train(X, Y, I, L, net_shapes, net_params, optimizer, utility, pool, individual_size, sigma, boundary_size=4):

    noise_seed = np.random.randint(
        0, 2 ** 32 - 1, size=individual_size, dtype=np.uint32).repeat(2)    # mirrored sampling

    pick_seed = np.random.randint(
        0, boundary_size, size=individual_size, dtype=np.uint32).repeat(2)    # mirrored sampling

    # distribute training in parallel
    jobs = [pool.apply_async(kgpn.get_reward, (net_shapes, net_params[pick_seed[k_id]], X, Y, I, L, sigma,
                                               [noise_seed[k_id], k_id], )) for k_id in range(individual_size*2)]

    rewards = np.array([j.get()[0] for j in jobs])

    new_params = ([j.get()[1] for j in jobs])

    sort_rewards = rewards[:, 0]

    # rank kid id by reward
    kids_rank = np.argsort(sort_rewards)[::-1]

    non_domin_rank = fast_non_dominated_sort(rewards)

    non_dominted_set_idxs = non_domin_rank[0]

    if(len(non_dominted_set_idxs) < boundary_size):
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

    #variations
    for ui, k_id in enumerate(knee_boundary_idxs):
        # reconstruct noise using seed
        np.random.seed(noise_seed[k_id])
        cumulative_update += utility[ui] * np.random.randn(net_params[0].size)

    gradients = optimizer.get_gradients(cumulative_update/((len(knee_boundary_idxs))*sigma))

    for p in knee_boundary_set:
        p = p + gradients 

    #best_params = net_params + gradients

    #best_params = best_params + LR/(3*SIGMA) * cumulative_update
    #best_params = net_params + LR/(2*N_KID*SIGMA) * cumulative_update

    print(len(knee_boundary_set),rewards[knee_boundary_idxs[0]])
    print(len(knee_boundary_set),rewards[knee_boundary_idxs[1]])
    print(len(knee_boundary_set),rewards[knee_boundary_idxs[2]])
    print(len(knee_boundary_set),rewards[knee_boundary_idxs[3]])
    return knee_boundary_set, best_rewards


def execute(config, trainX, trainY, trainI, trainL, testX, testY, testI, testL, pretrained=None, wandb=None):

    # utility instead reward for update parameters (rank transformation)
    base = config.individual_size * 2    # *2 for mirrored sampling
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base #

    # build Network
    net_shapes, net_params = kgpn.build_net(
        config.input_size, config.action_size, config.output_size)

    if(pretrained is not None):
        print("load pretrained")
        net_params[(config.input_size+1):] = pretrained
    
    optimizer = SGD(net_params, config.learning_rate)
    pool = mp.Pool(processes=config.core_size)
    mar = None      # moving average reward

    params_set = [net_params, net_params, net_params, net_params]

    for g in range(config.population_size):
        t0 = time.time()
        params_set, kid_rewards = train(trainX, trainY, trainI, trainL,
                                        net_shapes, params_set, optimizer, utility, pool, config.individual_size, config.sigma)

        # test trained net without noise
        net_r = kgpn.get_reward(
            net_shapes, params_set[0], trainX, trainY, trainI, trainL, None)[0]

        # mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward

        testing_r = kgpn.get_reward(
            net_shapes, params_set[0], testX, testY, testI, testL, None, real_value = True)[0]

        if (wandb is not None):

            wandb.log({'epoch': g,
                       "training/accuracy": 1-net_r[0],
                       "training/earliness": net_r[1],
                       "training/reward": net_r[2],
                       "testing/accuracy": 1-testing_r[0],
                       "testing/earliness": testing_r[1],
                       "testing/reward": testing_r[2]
                       })

        print(
            'Gen: ', g,
            '| Accuracy: %.4f' % float(1-net_r[0]),
            '| Earliness: %.4f' % float(net_r[1]),
            '| Reward: %.4f' % float(-1*net_r[2]),
            '| Gen_T: %.2f' % (time.time() - t0),)

        print(
            'Testing: ', g,
            '| Accuracy: %.4f' % float(1-testing_r[0]),
            '| Earliness: %.4f' % float(testing_r[1]),
            '| Reward: %.4f' % float(-1*testing_r[2]),

        )

        # if mar >= CONFIG['eval_threshold']: break

    return net_shapes, params_set


if __name__ == "__main__":

    for seed in range(7, 11):

        #config = Config(seed = seed,message ="20210216-log+-reward-2000")
                
        config = Config(seed = seed,message ="20210522-ogk-cursoft-punish1-trueidx-trainfake-testtrue",
                        population_size=500,output_size=9,learning_rate=0.01,
                        model_name="KGNA-NO-Pre",state_model_name="CNNLSTM",dataset_name="ICBEB")

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

        wandb.init(project="etscmoo",
                           entity='hugo',
                           dir = log_path,
                           name = config.model_name,
                           config=config,
                           reinit=True)

        if (config.dataset_name in ["ICBEB","ptbxl"]):
            weight_path = os.path.join(config.root_dir,
                                            config.output_dir,
                                            config.weight_dir,
                                            config.state_model_name,
                                            config.dataset_name,
                                            str(seed))
            uio.check_folder(weight_path)
            
            weight_dict = uio.load_pkfile(weight_path+"/"+config.weight_name)

            pretrained = None#np.concatenate((weight_dict['weight'].flatten(),weight_dict['bias']))
            
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
        
        net_shapes, params_set = execute(
            config, trainX, trainY, trainI, trainL,
            testX, testY, testI, testL, 
            pretrained = pretrained, wandb=wandb)

        model = {"shapes": net_shapes,
                 "knee": params_set[0],
                 "boundary_1": params_set[1],
                 "boundary_2": params_set[2],
                 "boundary_3": params_set[3]}

        uio.save_pkfile(model_path+"/"+config.model_save_name, model)

        wandb.save(model_path+"/"+config.model_save_name)

testX[1].shape, testY[1], testI[1].shape, testL[1]

# +

base = 20
rank = np.arange(1, base + 1)

util = np.log(base + 1) - np.log(rank)
util
# -


