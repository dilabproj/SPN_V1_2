# -*- coding: utf-8 -*-
import numpy as np
from core.utils import *

def build_net(input_size, action_size, output_size):

    def linear(n_in, n_out):  # network linear layer
        
        #w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        #b = np.random.randn(n_out).astype(np.float32) * .1
        # He initialization
        w = np.random.randn(n_in * n_out).astype(np.float32) * np.sqrt(2/n_in)
        b = np.zeros(n_out).astype(np.float32)
        return (n_in, n_out), np.concatenate((w, b))

    shape_controller, param_controller = linear(input_size, action_size)
    shape_discriminator, param_discriminator = linear(input_size, output_size)

    return [shape_controller, shape_discriminator], np.concatenate((param_controller, param_discriminator))


def params_reshape(shapes, params):     # reshape to be a matrix

    p, start = [], 0

    for i, shape in enumerate(shapes):  # flat params to matrix
        n_w, n_b = shape[0] * shape[1], shape[1]
        p = p + [params[start: start + n_w].reshape(shape),
                 params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
        start += n_w + n_b
        
    return p


def softmax(x,axis=0):
        
    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum(axis=axis)

def sigmoid(x):
    
    return 1/(1 + np.exp(-x))

def sign(k_id): return -1. if k_id % 2 == 0 else 1.  # mirrored sampling


def model_run(params, X):

    labels = []

    stop_points = []

    for x in X:
        
        max_length = x.shape[0]

        stop_time = 0

        stop_probs = []

        for t in range(max_length):

            S_t = x[t]

            agent_output = S_t.dot(params[0]) + params[1]

            stop_probs.append(agent_output)

            stop_time = t
            
            if t < 1:
                softmax_stop_probs = sigmoid(stop_probs[0])
                if(softmax_stop_probs >= 1):
                    break
                continue
            else:
                softmax_stop_probs = softmax(stop_probs)
                sorted_stop_probs = np.sort(softmax_stop_probs[:-1])

                stopping_criteria = softmax_stop_probs[-1] + \
                    (softmax_stop_probs[-1]-sorted_stop_probs[-1])

                if(stopping_criteria >= 1):
                    break

        result = S_t.dot(params[2]) + params[3]

        stop_points.append(stop_time)

        labels.append(np.argmax(result, axis=1)[0])
        
    return labels, stop_points


# #### 修改

from sklearn.metrics import recall_score


def get_reward(shapes, params, X, y, index, length, sigma, seed_and_id=None, real_value = False, label_size = 9):
    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        params += sign(k_id) * sigma * np.random.randn(params.size)

    p = params_reshape(shapes, params)

    y_predict, y_estimate = model_run(p, X)
    
    real_index = []
    
    snippet_length = []
    
    for x in X:
        snippet_length.append(x.shape[0])
        
    snippet_length = np.array(snippet_length)
    
    if(real_value):
        #print(y_estimate)
        for idx, point in enumerate(y_estimate):
            real_index.append(index[idx][point])

        real_index = np.array(real_index)
        ear = np.mean((real_index/length.flatten()))
    else:
        ear = np.mean((y_estimate/snippet_length.flatten()))
    
    err_hist = [0] * label_size
    acc_hist = [0] * label_size
    cnt_hist = [0] * label_size
    
    y_hat = []
    
    rwd = 0

    acc = 0
    
    for idx, yest in enumerate(y_estimate):
        
        base = snippet_length[idx]
        
        rank = np.arange(1, base + 1)
        
        ans = y[idx][y_predict[idx]]
        
        pred_bin = y_predict[idx]
        
        true_bin = np.argmax(y[idx],axis=0) 
                
        if(ans):
            acc += 1
            y_hat.append(pred_bin)
            acc_hist[pred_bin]+=1
            cnt_hist[pred_bin]+=1
            
            util = np.maximum(1, np.log(base + 1) - np.log(rank))
            
            rwd += util[yest]
            
            if(real_value):
                err_hist[pred_bin] += index[idx][yest]/length[idx]
            else:
                err_hist[pred_bin] += y_estimate[idx]/snippet_length[idx]
            
        else:
            y_hat.append(true_bin)
            cnt_hist[true_bin]+=1
            util = np.maximum(1, np.log(base + 1) - np.log(rank))
            rwd -= util[yest]
            
            if(real_value):
                try:
                    err_hist[true_bin] += index[idx][yest]/length[idx]
                except:
                    print("bin:", true_bin, "   idx:", idx, "real_idx.shape", real_index, "   length:", length.shape)
            else:
                err_hist[true_bin] += y_estimate[idx]/snippet_length[idx]
     
    #conf_matrix = recall_score(y_true=y_hat, y_pred=y_predict, average=None)
    
    acc_hist = np.array(acc_hist)
    cnt_hist = np.array(cnt_hist)
    err_hist = np.array(err_hist)
    
    
    objective_list = np.concatenate([acc_hist/cnt_hist,  err_hist/cnt_hist])
    
    '''
    print('diy:', 1-acc_hist/cnt_hist)
    
    print("err:", err_hist/cnt_hist)
    
    print("    mean earliness:", np.mean(err_hist/cnt_hist)  )
    '''
    
    rwd = rwd / len(y_estimate)

    acc = 1 - (acc / len(y_estimate))

    rwd = (-1 * rwd) 
    
    return (acc, ear,rwd), params, objective_list


def get_results(shapes, params, X, y, index, length, sigma, seed_and_id=None, real_value = False):
    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        params += sign(k_id) * sigma * np.random.randn(params.size)

    p = params_reshape(shapes, params)

    y_predict, y_estimate = model_run(p, X)
    
    real_index = []
    
    snippet_length = []
    
    for x in X:
        snippet_length.append(x.shape[0])
        
    snippet_length = np.array(snippet_length)
    
    if(real_value):
        #print(y_estimate)
        for idx, point in enumerate(y_estimate):
            real_index.append(index[idx][point])

        real_index = np.array(real_index)
        ear = np.mean((real_index/length.flatten()))
    else:
        ear = np.mean((y_estimate/snippet_length.flatten()))
        
    return (y_predict, y, real_index, length)


def model_inference(params, X):

    output_or_not = False
    
    max_length = X.shape[0]

    stop_time = 0

    stop_probs = []

    for t in range(max_length):

        S_t = X[t]

        agent_output = S_t.dot(params[0]) + params[1]

        stop_probs.append(agent_output)

        stop_time = t
           
        if t < 1:
            softmax_stop_probs = sigmoid(stop_probs[0])
            if(softmax_stop_probs >= 1):
                output_or_not = True
                break
            continue
        else:
            softmax_stop_probs = softmax(stop_probs)
            sorted_stop_probs = np.sort(softmax_stop_probs[:-1])

            stopping_criteria = softmax_stop_probs[-1] + \
                (softmax_stop_probs[-1]-sorted_stop_probs[-1])

            if(stopping_criteria >= 1):
                output_or_not = True
                break

    result = S_t.dot(params[2]) + params[3]
    
    result = np.argmax(result, axis=1)[0]
    
    return result, stop_time, output_or_not, softmax_stop_probs


def inference(shapes,params,X):
    X = np.array(X)
    p = params_reshape(shapes, params)
    y_predict, y_estimate, y_flag, probs_list = model_inference(p, X)
    return y_predict, y_estimate, y_flag, probs_list
