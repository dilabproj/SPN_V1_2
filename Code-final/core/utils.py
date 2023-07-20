import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def entropy(p):
    return -(p*torch.log(p+1e-12)).sum(1)

def build_t_index(batchsize, sequencelength):
    # linear increasing t index for time regularization
    """
    t_index
                          0 -> T
    tensor([[ 0.,  1.,  2.,  ..., 97., 98., 99.],
            [ 0.,  1.,  2.,  ..., 97., 98., 99.],
    batch   [ 0.,  1.,  2.,  ..., 97., 98., 99.],
            ...,
            [ 0.,  1.,  2.,  ..., 97., 98., 99.],
            [ 0.,  1.,  2.,  ..., 97., 98., 99.],
            [ 0.,  1.,  2.,  ..., 97., 98., 99.]])
    """
    t_index = torch.ones(batchsize, sequencelength) * torch.arange(sequencelength).type(torch.FloatTensor)
    if torch.cuda.is_available():
        return t_index.cuda()
    else:
        return t_index

def build_yhaty(logprobabilities, targets):
    batchsize, seqquencelength, nclasses = logprobabilities.shape

    eye = torch.eye(nclasses).type(torch.ByteTensor)
    if torch.cuda.is_available():
        eye = eye.cuda()

    # [b, t, c]
    targets_one_hot = eye[targets].bool()

    # implement the y*\hat{y} part of the loss function
    
    y_haty = torch.masked_select(logprobabilities, targets_one_hot)

    return y_haty.view(batchsize, seqquencelength).exp()

def loss_early_reward(logprobabilities, pts, targets, alpha=0.5, ptsepsilon = 10, power=1):

    batchsize, seqquencelength, nclasses = logprobabilities.shape
    t_index = build_t_index(batchsize=batchsize, sequencelength=seqquencelength)

    #pts_ = torch.nn.functional.softmax((pts + ptsepsilon), dim=1)
    if ptsepsilon is not None:
        ptsepsilon = ptsepsilon / seqquencelength
        #pts += ptsepsilon

    #pts_ = torch.nn.functional.softmax((pts + ptsepsilon), dim=1)
    pts_ = (pts + ptsepsilon)

    b,t,c = logprobabilities.shape
    #loss_classification = F.nll_loss(logprobabilities.view(b*t,c), targets.view(b*t))

    targets = targets.unsqueeze(-1).repeat(1,seqquencelength)

    xentropy = F.nll_loss(logprobabilities.transpose(1, 2).unsqueeze(-1), 
        targets.unsqueeze(-1),reduction='none').squeeze(-1)
    loss_classification = alpha * ((pts_ * xentropy)).sum(1).mean()

    yyhat = build_yhaty(logprobabilities, targets)
    earliness_reward = (1-alpha) * ((pts) * (yyhat)**power * (1 - (t_index / seqquencelength))).sum(1).mean()

    loss = loss_classification - earliness_reward

    stats = dict(
        loss=loss,
        loss_classification=loss_classification,
        earliness_reward=earliness_reward
    )

    return loss, stats

def exponentialDecay(N):
    tau = 1 
    tmax = 4 
    t = np.linspace(0, tmax, N)
    y = np.exp(-t/tau)
    y = torch.FloatTensor(y)
    return y/10.

def update_earliness_cpu(t_stops, earliness, length, raw_length, ratio):
    
    for index, val in enumerate(t_stops):
                
        stp = t_stops[index]
                
        rate = 1
        
        if(stp <= length[index]):
            
            if(length[index] > raw_length*ratio):
                rate = stp/length[index]
                earliness += stp/length[index]
            else:
                rate = stp/(raw_length*ratio)
                earliness += stp/(raw_length*ratio)
        else:
            earliness += 1
            
    return earliness


def update_earliness_eval_cpu(t_stops, earliness, length, raw_length, ratio):
    
    for index, val in enumerate(t_stops):
                
        stp = t_stops[index]
                
        rate = 1
        
        if(stp <= length[index]):
            rate = stp/length[index]
            earliness += stp/length[index]
        else:
            earliness += 1
            
    return earliness


def update_earliness(t_stops, earliness, length, raw_length, ratio):
    
    for index, val in enumerate(t_stops):
                
        stp = t_stops.cpu().numpy()[index]
                
        rate = 1
        
        if(stp <= length[index]):
            
            if(length[index] > raw_length*ratio):
                rate = stp/length[index]
                earliness += stp/length[index]
            else:
                rate = stp/(raw_length*ratio)
                earliness += stp/(raw_length*ratio)
        else:
            earliness += 1
            
    return earliness

def update_earliness_eval(t_stops, earliness, length, raw_length, ratio):
    
    for index, val in enumerate(t_stops):
                
        stp = t_stops.cpu().numpy()[index]
                
        rate = 1
        
        if(stp <= length[index]):
            rate = stp/length[index]
            earliness += stp/length[index]
        else:
            earliness += 1
            
    return earliness


def update_performance(predicted, correctness, y_pred, y_true, y_label, y_list):

    for index, val in enumerate(predicted):
        y_pred.append(val.cpu().detach().numpy())
        if(y_list[index][val.cpu().detach().numpy()]):
            correctness += 1
            y_true.append(val.cpu().detach().numpy())
        else:

            y_true.append(y_label[index].cpu().numpy())

    return correctness, y_pred, y_true


def update_performance_label(predicted, correctness, y_pred, y_true, y_label, y_list):

    for index, val in enumerate(predicted):
        y_pred.append(val.cpu().detach().numpy())
        
        if(val.cpu().detach().numpy() in y_list[index]):
            correctness += 1
            y_true.append(val.cpu().detach().numpy())
        else:

            y_true.append(y_label[index].cpu().numpy())

    return correctness, y_pred, y_true
