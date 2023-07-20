import os
import time
import wandb
import pickle
import operator
import numpy as np
import utils.io as uio
import multiprocessing as mp
import state_generator as sg
import models.contrained_policynet as kgpn

from core.loss import SGD
from core.utils import *
from configs.contrained_policynet_config import Config


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


def fit_criterion(left, op, right):
    
    return op(left,right)


def constraint_filtering(constaint_objective_set, conditions, tolerance = 0.1):
    result = []
    
    # check objectives of each individuals in the set
    for idx, constaint_objective in enumerate(constaint_objective_set):
        
        status = []
        
        # loop all conditions 
        for condition in conditions:

            obj_value = constaint_objective[condition['obj_idx']]

            status.append(fit_criterion(obj_value,condition['operation'],condition['criterion']))
            
        pass_flag = all(status)
                
        if(pass_flag):
            result.append(idx)
        
    return result


# np.random.seed(1)
# a = np.random.randn(10)
# np.random.seed(1)
# b = np.random.randn(10)
# np.random.seed(1)
# c = np.random.randn(10)
# print(a)
# print(b)
# print(c)

def train(X, Y, I, L, net_shapes, params_group, optimizer, 
          utility, pool, individual_size, sigma, boundary_size=4, conditions=None):

    noise_seed = np.random.randint(
        0, 2 ** 32 - 1, size=individual_size, dtype=np.uint32).repeat(2)    # mirrored sampling

    pick_seed = np.random.randint(
        0, len(params_group[0]), size=individual_size, dtype=np.uint32).repeat(2)    # mirrored sampling

    # distribute training in parallel
    jobs = [pool.apply_async(kgpn.get_reward, (net_shapes, params_group[0][pick_seed[k_id]], X, Y, I, L, sigma,
                                               [noise_seed[k_id], k_id],)) for k_id in range(individual_size*2)]

    rewards = np.array([j.get()[0] for j in jobs])
    
    new_params = np.array([j.get()[1] for j in jobs])

    constaint_objective_set = np.array([j.get()[2] for j in jobs])
    # rank kid id by reward
    
    valid_idx = constraint_filtering(constaint_objective_set, conditions)
        
    ''' '''
    ### if the number of the valid solutions is larger than the boundary size (knee + objectives)
    if (len(valid_idx) >= boundary_size):
        #print(valid_idx)
        rewards = rewards[valid_idx]
        new_params = new_params[valid_idx]
    else:
        print("valid solutions insufficent:",valid_idx)
    
    
    # 18 objectives
    #non_domin_rank = fast_non_dominated_sort(constaint_objective_set)
    
    # 3 objectives
    non_domin_rank = fast_non_dominated_sort(rewards)

    non_dominted_set_idxs = non_domin_rank[0]
    
    rank_index = 1
    
    ### loop for add non_dominated rank into the set util it mees the boundary size
    while(len(non_dominted_set_idxs) < boundary_size): 
        print("non_dominted_set_idxs insufficent:",len(non_dominted_set_idxs))
        non_dominted_set_idxs.append(non_domin_rank[rank_index][0])
        rank_index += 1

    non_dominated_objectives = np.array(
        [rewards[idx] for idx in non_dominted_set_idxs])

    # knee_id in the non_dominated_set
    knee_idx = knee_mdd_selection(non_dominated_objectives)

    # get the best individual, aka the knee individual

    best_params = new_params[non_dominted_set_idxs[knee_idx]]

    best_rewards = rewards[non_dominted_set_idxs[knee_idx]]

    boundary_idxs = boundary_individuals_selection(non_dominated_objectives)

    knee_boundary_idxs = [non_dominted_set_idxs[x] for x in boundary_idxs]

    knee_boundary_idxs.insert(0, non_dominted_set_idxs[knee_idx])

    '''
    #add valid_idx to the knee_boundary_idxs for the next generation
    
    vali_add_count = 0
    
    for idx in valid_idx:
        if(idx not in knee_boundary_idxs):
            knee_boundary_idxs.append(idx)
            vali_add_count+=1
        if vali_add_count >= boundary_size:
            break
    '''
    
    print(non_dominted_set_idxs, knee_idx, knee_boundary_idxs)
    
    cumulative_update = np.zeros_like(
        params_group[0][0])       # initialize update values

    knee_boundary_set = [new_params[x] for x in knee_boundary_idxs]

    #variations
    for ui, k_id in enumerate(knee_boundary_idxs):
        # reconstruct noise using seed
        np.random.seed(noise_seed[k_id])
        cumulative_update += utility[ui] * np.random.randn(params_group[0][0].size)

    gradients = optimizer.get_gradients(cumulative_update/((len(knee_boundary_idxs))*sigma))

    for p in knee_boundary_set:
        p = p + gradients 

    #best_params = net_params + gradients
    #best_params = best_params + LR/(3*SIGMA) * cumulative_update
    #best_params = net_params + LR/(2*N_KID*SIGMA) * cumulative_update

    print(len(knee_boundary_set),rewards[knee_boundary_idxs[0]])
    print(len(knee_boundary_set),rewards[knee_boundary_idxs[1]])
    print(len(knee_boundary_set),rewards[knee_boundary_idxs[2]])
    #print(len(knee_boundary_set),rewards[knee_boundary_idxs[3]])
    return [knee_boundary_set, knee_boundary_set], best_rewards


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

    params_group = [params_set, params_set]
    
    for g in range(config.population_size):
        t0 = time.time()
        params_group, kid_rewards = train(trainX, trainY, trainI, trainL,
                                        net_shapes, params_group, optimizer, utility, pool, 
                                          config.individual_size, config.sigma, 
                                         conditions = config.conditions)

        # test trained net without noise
        net_r = kgpn.get_reward(
            net_shapes, params_group[0][0], trainX, trainY, trainI, trainL, None)

        # mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward

        testing_r = kgpn.get_reward(
            net_shapes, params_group[0][0], testX, testY, testI, testL, None, real_value = True)

        if (wandb is not None):

            if config.dataset_name == "ICBEB":
                wandb.log({'epoch': g,
                           "training/accuracy": 1-net_r[0][0],
                           "training/earliness": net_r[0][1],
                           #"training/reward": net_r[2],
                           "testing/accuracy": 1-testing_r[0][0],
                           "testing/earliness": testing_r[0][1],
                           #"testing/reward": testing_r[2],

                           'training_recall/1AVB': net_r[2][0],
                           'training_recall/AFIB': net_r[2][1],
                           'training_recall/CLBBB': net_r[2][2],
                           'training_recall/CRBBB': net_r[2][3],
                           'training_recall/NORM': net_r[2][4],
                           'training_recall/PAC': net_r[2][5],
                           'training_recall/STD_': net_r[2][6],
                           'training_recall/STE_': net_r[2][7],
                           'training_recall/VPC': net_r[2][8],
                           'training_recall/Average': np.mean(net_r[2][0:9]),

                           'training_earliness/1AVB': net_r[2][9],
                           'training_earliness/AFIB': net_r[2][10],
                           'training_earliness/CLBBB': net_r[2][11],
                           'training_earliness/CRBBB': net_r[2][12],
                           'training_earliness/NORM': net_r[2][13],
                           'training_earliness/PAC': net_r[2][14],
                           'training_earliness/STD_': net_r[2][15],
                           'training_earliness/STE_': net_r[2][16],
                           'training_earliness/VPC': net_r[2][17],

                           'testing_recall/1AVB': testing_r[2][0],
                           'testing_recall/AFIB': testing_r[2][1],
                           'testing_recall/CLBBB': testing_r[2][2],
                           'testing_recall/CRBBB': testing_r[2][3],
                           'testing_recall/NORM': testing_r[2][4],
                           'testing_recall/PAC': testing_r[2][5],
                           'testing_recall/STD_': testing_r[2][6],
                           'testing_recall/STE_': testing_r[2][7],
                           'testing_recall/VPC': testing_r[2][8],
                           'testing_recall/Average': np.mean(testing_r[2][0:9]),

                           'testing_earliness/1AVB': testing_r[2][9],
                           'testing_earliness/AFIB': testing_r[2][10],
                           'testing_earliness/CLBBB': testing_r[2][11],
                           'testing_earliness/CRBBB': testing_r[2][12],
                           'testing_earliness/NORM': testing_r[2][13],
                           'testing_earliness/PAC': testing_r[2][14],
                           'testing_earliness/STD_': testing_r[2][15],
                           'testing_earliness/STE_': testing_r[2][16],
                           'testing_earliness/VPC': testing_r[2][17],
                           })
            else:
                wandb.log({'epoch': g,
                           "training/accuracy": 1-net_r[0][0],
                           "training/earliness": net_r[0][1],
                           #"training/reward": net_r[2],
                           "testing/accuracy": 1-testing_r[0][0],
                           "testing/earliness": testing_r[0][1],
                           #"testing/reward": testing_r[2],

                           'training_recall/CD': net_r[2][0],
                           'training_recall/HYP': net_r[2][1],
                           'training_recall/MI': net_r[2][2],
                           'training_recall/NORM': net_r[2][3],
                           'training_recall/STTC': net_r[2][4],
                           'training_recall/Average': np.mean(net_r[2][0:5]),

                           'training_earliness/CD': net_r[2][5],
                           'training_earliness/HYP': net_r[2][6],
                           'training_earliness/MI': net_r[2][7],
                           'training_earliness/NORM': net_r[2][8],
                           'training_earliness/STTC': net_r[2][9],

                           'testing_recall/CD': testing_r[2][0],
                           'testing_recall/HYP': testing_r[2][1],
                           'testing_recall/MI': testing_r[2][2],
                           'testing_recall/NORM': testing_r[2][3],
                           'testing_recall/STTC': testing_r[2][4],
                           'testing_recall/Average': np.mean(testing_r[2][0:5]),

                           'testing_earliness/CD': testing_r[2][5],
                           'testing_earliness/HYP': testing_r[2][6],
                           'testing_earliness/MI': testing_r[2][7],
                           'testing_earliness/NORM': testing_r[2][8],
                           'testing_earliness/STTC': testing_r[2][9],
                           })
                    

            
        #print("here:  ",net_r)
        
        '''
        print(
            'Gen: ', g,
            '| Accuracy: ' , (1-net_r[0:9]), 
            '| Earliness: ' , (net_r[9:18]), 
            '| Gen_T: %.2f' % (time.time() - t0),)
        
        print('Overall Accuracy:', np.mean(1-net_r[0:9]), 'Overall Earliness:',np.mean(net_r[9:18]),)

        print(
            'Testing: ', g,
            '| Accuracy: ', (1-testing_r[0:9]), np.mean(1-testing_r[0:9]),
            '| Earliness: ', (testing_r[9:18]), np.mean(testing_r[9:18]),
        )
        print('Overall Accuracy:', np.mean(1-testing_r[0:9]), 'Overall Earliness:',np.mean(testing_r[9:18]),)
        '''
        
        print(
            'Gen: ', g,
            '| Accuracy: %.4f' % float(1-net_r[0][0]),
            '| Earliness: %.4f' % float(net_r[0][1]),
            #'| Reward: %.4f' % float(net_r[2]),
            '| Gen_T: %.2f' % (time.time() - t0),)

        print(
            'Gen: ', g,
            '\n| Accuracy: ' , net_r[2][:9],
            '\n| Earliness: ' , net_r[2][9:]
            )
        
        print(
            'Testing: ', g,
            '| Accuracy: %.4f' % float(1-testing_r[0][0]),
            '| Earliness: %.4f' % float(testing_r[0][1]),
            #'| Reward: %.4f' % float(testing_r[2]),

        )
        
        # if mar >= CONFIG['eval_threshold']: break

    return net_shapes, params_group[0]

# ICBEB ['1AVB' 'AFIB' 'CLBBB' 'CRBBB' 'NORM' 'PAC' 'STD_' 'STE_' 'VPC']
#
# ptbxl ['CD' 'HYP' 'MI' 'NORM' 'STTC']

if __name__ == "__main__":

    for seed in range(1, 6):

        #config = Config(seed = seed,message ="20210216-log+-reward-2000")
        #config = Config(seed = seed,message ="20210216-log+-reward-2000")
        conditions = [{
            "obj_idx": 0,
            "operation":operator.gt,
            "criterion":0.9
        },{
            "obj_idx": 2,
            "operation":operator.gt,
            "criterion":0.9
        },{
            "obj_idx": 3,
            "operation":operator.gt,
            "criterion":0.9
        },]
        
        
        config = Config(seed = seed,message ="20220104-constraint_Ablation_0.9",
                        population_size = 500, output_size = 9, learning_rate=0.01,
                        core_size = 15, individual_size = 40,
                        model_name="KGNA-Constraint",state_model_name="CNNLSTM",dataset_name="ICBEB",
                        conditions = conditions, model_save_name="constrain_Ablation_90.pkl")
        '''
        config = Config(seed = seed,message ="20220104-constraint_NONORM_1.00",
                        population_size = 100, output_size = 5, learning_rate=0.01,
                        core_size = 40, individual_size = 40,
                        model_name="KGNA-Constraint",state_model_name="CNNLSTM-500",dataset_name="ptbxl",
                        conditions = conditions, model_save_name="constrain_NONORM_100_TOR_0.1.pkl")
        '''
        
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

        
        wandb.init(project='etsc-ct', entity='hugo',
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

            pretrained = np.concatenate((weight_dict['weight'].flatten(),weight_dict['bias']))
            
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
