import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from core.layers import Attention
from core.layers import BaseCNN, BaseRNN, Controller, Controller_New, RelaxedController
from core.layers import BaselineNetwork, Discriminator
from core.loss import FocalLoss
from core.ecbase import EarlyClassificationModel
from torch.nn.modules.normalization import LayerNorm
from models.snippet_backbone import HeartNetIEEE,D3CNN,ResCNN

class SPN(nn.Module):

    def __init__(self, input_size = 12, hidden_size = 256, hidden_output_size = 1, output_size = 9, fc_size = 4):
        super(SPN, self).__init__()
        self.loss_func = FocalLoss()#nn.CrossEntropyLoss()#
        self.CELL_TYPE = "LSTM"
        self.INPUT_SIZE = input_size
        self.HIDDEN_SIZE = hidden_size
        self.HIDDEN_OUTPUT_SIZE = hidden_output_size
        self.OUTPUT_SIZE = output_size
        # --- Sub-networks ---
        #self.BaseCNN = HeartNetIEEE(input_size).cuda()
        #self.BaseCNN = ResCNN(input_size).cuda()
        #self.BaseCNN = D3CNN(input_size).cuda()
                
        self.BaseCNN = BaseCNN(input_size, hidden_size, output_size, fc_size).cuda()
        self.BaseRNN = BaseRNN(hidden_size, hidden_size, self.CELL_TYPE).cuda()
        self.Controller = Controller(hidden_size, hidden_output_size).cuda()
        self.BaselineNetwork = BaselineNetwork(hidden_size, hidden_output_size).cuda()
        self.Discriminator = Discriminator(hidden_size, output_size).cuda()
        #self.BaseCNN.load_state_dict(torch.load("models/hb.pth"))
            
    def initHidden(self, batch_size, weight_size):
        """Initialize hidden states"""
        if self.CELL_TYPE == "LSTM":
            h = (torch.zeros(1, batch_size, weight_size).cuda(),
                 torch.zeros(1, batch_size, weight_size).cuda())
        else:
            h = torch.zeros(1, batch_size, weight_size).cuda()
        return h


    def forward(self, X):
        
        hidden = self.initHidden(len(X),self.HIDDEN_SIZE)
        #print(hidden[0].shape)
        #print(hidden[1].shape)
        #print(len(X), len(hidden))
        #print('HHHHASDHADAHDOHO')
        
        min_length = 1000
        max_length = 0
        
        for x in X:
            if min_length > x.shape[0]:
                min_length = x.shape[0]
            if max_length < x.shape[0]:
                max_length = x.shape[0]
        
        tau_list = np.zeros(X.shape[0], dtype=int)
        state_list = np.zeros(X.shape[0], dtype=int)
        
        log_pi = []
        baselines = []
        halt_probs = []
        
        flag = False
        count = 0
                
        for t in range(max_length):
            slice_input = []
            cnn_input = None # cpu
               
            if self.training:        
                for idx, x in enumerate(X):
                    slice_input.append(torch.from_numpy(x[tau_list[idx],:,:]).float())
                    cnn_input = torch.stack(slice_input, dim=0)
            else:
                for idx, x in enumerate(X):
                    slice_input.append(torch.from_numpy(x[t,:,:]).float())
                    cnn_input = torch.stack(slice_input, dim=0)

            cnn_input = cnn_input.cuda()
            #print('cnn_input', cnn_input.shape)
            S_t = self.BaseCNN(cnn_input)
            #print('S_t1', S_t.shape)
            S_t = S_t.unsqueeze(0)
            #print('S_t2', S_t.shape)
            #print(hidden[0].shape)
            #print(hidden[1].shape)
            #print(len(X), len(hidden))
            #print(S_t.shape)
            #print('HHHHASDHADAHDOHO')
            #hahahah
            S_t, hidden = self.BaseRNN(S_t,hidden) # Run sequence model
            
            #S_t = hidden[-1]
            
            S_t = hidden[0][-1]
            
            #T_t = torch.from_numpy(tau_list).float().unsqueeze(1).cuda()
            
            #C_t = torch.cat((T_t, S_t), 1)
            
            rate = t
            if(rate>10): rate = 10
            
            if self.training:
                a_t, p_t, w_t, probs = self.Controller(S_t, 0.01, train=True)
            else:
                a_t, p_t, w_t, probs = self.Controller(S_t, 0.05 * rate)
                #a_t, p_t, w_t, probs = self.Controller(S_t, 0.1* rate) #MOST Project Test the training performance
            
            b_t = self.BaselineNetwork(S_t) # Compute the baseline
            
            baselines.append(b_t)
            
            log_pi.append(p_t)
            
            halt_probs.append(w_t)
                        
            y_hat = self.Discriminator(S_t) # Classify the time series
            
            if self.training:
                for idx, a in enumerate(a_t):
                    
                    if(a == 0 and tau_list[idx] < X[idx].shape[0]-1):
                        tau_list[idx]+=1
                        
                    else:
                        state_list[idx] = 1
                        
                if (np.mean(state_list)>=1): break # break condition in training phrase
                    
            else:
                for idx, a in enumerate(a_t):
                    
                    tau_list[idx] = t
                    
                    if(a == 1):
                        flag = True
                        #print('Halt: ',t,'->',a.cpu().detach(),'=',probs.cpu().detach())
                    #else:
                        #print('Keep: ',t,'->',a.cpu().detach(),'=',0.05*rate)
                    
                    '''    
                    if( t/max_length>0.2):
                        flag = True
                    '''
                   
                if(flag): break # break condition in testing phrase
                    
        self.log_pi = torch.stack(log_pi).transpose(1, 0).squeeze(2)
        
        self.halt_probs = torch.stack(halt_probs).transpose(1, 0)
        
        
        self.baselines = torch.stack(baselines).squeeze(2).transpose(1, 0)
            
        self.tau_list = tau_list
        
        #print("Halting Point:", t, " Max Length:", max_length,"tau_list: ",self.tau_list)
        
        return y_hat, tau_list

    def applyLoss(self, y_hat, labels, alpha = 0.3, beta = 0.001, gamma = 0.5):
        # --- compute reward ---
        _, predicted = torch.max(y_hat, 1)
        r = (predicted.float().detach() == labels.float()).float() #checking if it is correct

        
        r = r*2 - 1 # return 1 if correct and -1 if incorrect
                
        #print(r.shape, r,self.tau_list)
        
        R = torch.from_numpy(np.zeros((self.baselines.shape[0],self.baselines.shape[1]))).float().cuda()
        
        for idx in range(r.shape[0]):
            for jdx in range(self.tau_list[idx]+1):
                R[idx][jdx] = r[idx] * (jdx+1)
        
        # --- subtract baseline from reward ---
        adjusted_reward = R - self.baselines.detach()
        
        #print("------>",R.shape, R, adjusted_reward)
        
        # --- compute losses ---
        self.loss_b = F.mse_loss(self.baselines, R) # Baseline should approximate mean reward
        self.loss_c = self.loss_func(y_hat, labels) # Make accurate predictions
        self.loss_r = torch.sum(-self.log_pi*adjusted_reward, dim=1).mean()
        self.time_penalty = torch.sum(self.halt_probs, dim=1).mean()
        
        # --- collect all loss terms ---
        loss = self.loss_c + self.loss_r + self.loss_b + beta * self.time_penalty
        
        return loss, self.loss_c, self.loss_r, self.loss_b, self.time_penalty

class EARLIEST(nn.Module):

    def __init__(self, N_FEATURES=1, N_CLASSES=2, HIDDEN_DIM=50, CELL_TYPE="LSTM",
                 N_LAYERS=1, DF=1., LAMBDA=0.001, FS = 250):
        super(EARLIEST, self).__init__()

        # --- Hyperparameters ---
        self.CELL_TYPE = "LSTM"
        self.HIDDEN_DIM = HIDDEN_DIM
        self.DF = DF
        self.FS = FS
        self.LAMBDA = torch.tensor([LAMBDA], requires_grad=False)
        self.N_LAYERS = N_LAYERS
        self._epsilon = 0.999
        self._rewards = 0

        # --- Sub-networks ---
        self.BaseRNN = BaseRNN(N_FEATURES, HIDDEN_DIM, self.CELL_TYPE)
        self.Controller = Controller(HIDDEN_DIM+1, 1, False) # Add +1 for timestep input
        self.BaselineNetwork = BaselineNetwork(HIDDEN_DIM, 1)
        self.Discriminator = Discriminator(HIDDEN_DIM, N_CLASSES)

    def initHidden(self, batch_size):
        """Initialize hidden states"""
        if self.CELL_TYPE == "LSTM":
            h = (torch.zeros(self.N_LAYERS, batch_size, self.HIDDEN_DIM),
                 torch.zeros(self.N_LAYERS, batch_size, self.HIDDEN_DIM))
        else:
            h = torch.zeros(self.N_LAYERS, batch_size, self.HIDDEN_DIM)
        return h

    def forward(self, X):
        
        baselines = []
        log_pi = []
        halt_probs = []
        attention = []
        hidden_states = []
        hidden = self.initHidden(X.shape[0]) # Initialize hidden states - input is batch size
        step = 1
        
        #X = torch.transpose(X, 1, 2)
        
        print("here,",X.shape)
        
        for t in range(X.shape[1]):
            x_t = X[:,t:t+1,:] # add time dim back in
            
            S_t, hidden = self.BaseRNN(x_t, hidden) # Run sequence model
            
            S_t = S_t.squeeze(0) # remove time dim
            t = torch.tensor([t], dtype=torch.float).view(1, 1) # collect timestep
            hidden_states.append(S_t)
            
            S_t_with_t = torch.cat((S_t, t), dim=1) # Add timestep as input to controller
            
            
            a_t, p_t, w_t, probs = self.Controller(S_t_with_t, self._epsilon) # Compute halting-probability and sample an action
            b_t = self.BaselineNetwork(S_t) # Compute the baseline
            baselines.append(b_t)
            log_pi.append(p_t)
            halt_probs.append(w_t)
            if a_t == 1:
                break

        y_hat = self.Discriminator(S_t) # Classify the time series
        self.baselines = torch.stack(baselines).transpose(1, 0)
        self.baselines = self.baselines.view(1, -1)
        self.log_pi = torch.stack(log_pi).transpose(1, 0).squeeze(2)
        self.halt_probs = torch.stack(halt_probs).transpose(1, 0)
        self.halting_point = t+1 # Adjust timestep indexing just for plotting
        self.locations = self.halting_point
        return y_hat, t+1

    def applyLoss(self, y_hat, labels):
        # --- compute reward ---
        _, predicted = torch.max(y_hat, 1)
        r = (predicted.float().detach() == labels.float()).float()
        
        print("r shape: ",r.shape)
        print("halting: ",self.halting_point.shape, self.halting_point)
        r = r*2 - 1
        R = r.unsqueeze(1).repeat(1, int(self.halting_point.squeeze()))

        print("R shape: ",R.shape, R)
        print("baseline shape: ",self.baselines.shape,self.baselines)
        
        # --- discount factor ---
        discount = [self.DF**i for i in range(int(self.halting_point.item()))]
        discount = np.array(discount).reshape(1, -1)
        discount = np.flip(discount, 1)
        discount = torch.from_numpy(discount.copy()).float().view(1, -1)
        R = R * discount
        self._rewards += torch.sum(R) # Collect the sum of rewards for plotting

        # --- subtract baseline from reward ---
        adjusted_reward = R - self.baselines.detach()

        # --- compute losses ---
        self.loss_b = F.mse_loss(self.baselines, R) # Baseline should approximate mean reward
        self.loss_c = F.cross_entropy(y_hat, labels) # Make accurate predictions
        self.loss_r = torch.sum(-self.log_pi*adjusted_reward, dim=1) # Controller should lead to correct predictions from the discriminator
        self.time_penalty = torch.sum(self.halt_probs, dim=1).mean() # Penalize late predictions

        # --- collect all loss terms ---
        loss = (0.01*self.loss_r \
                + self.loss_c \
                + 0.01*self.loss_b \
                + 0.0001*self.time_penalty)
        return loss


class EARLIEST_NEW(nn.Module):
    """Code for the paper titled: Adaptive-Halting Policy Network for Early Classification
    Paper link: https://dl.acm.org/citation.cfm?id=3330974
    Method at a glance: An RNN is trained to model time series
    with respect to a classification task. A controller network
    decides at each timestep whether or not to generate the
    classification. Once a classification is made, the RNN
    stops processing the time series.
    Parameters
    ----------
    ninp : int
        number of features in the input data.
    nclasses : int
        number of classes in the input labels.
    nhid : int
        number of dimensions in the RNN's hidden states.
    rnn_type : str
        which RNN memory cell to use: {LSTM, GRU, RNN}.
        (if defining your own, leave this alone)
    lam : float32
        earliness weight -- emphasis on earliness.
    nlayers : int
        number of layers in the RNN.
    """
    def __init__(self, ninp=1, nclasses=1, nhid=50, rnn_type="LSTM",
                 nlayers=1, lam=0.0):
        super(EARLIEST_NEW, self).__init__()

        # --- Hyperparameters ---
        self.ninp = ninp
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.lam = lam
        self.nclasses = nclasses

        # --- Sub-networks ---
        self.Controller = Controller_New(nhid+1, 1)
        self.BaselineNetwork = BaselineNetwork(nhid+1, 1)
        if rnn_type == "LSTM":
            self.RNN = torch.nn.LSTM(ninp, nhid)
        else:
            self.RNN = torch.nn.GRU(ninp, nhid)
        self.out = torch.nn.Linear(nhid, nclasses)

        print(self.ninp, self.rnn_type,self.nhid,self.nlayers,self.nclasses)
        
    def initHidden(self, bsz):
        """Initialize hidden states"""
        if self.rnn_type == "LSTM":
            return (torch.zeros(self.nlayers, bsz, self.nhid),
                    torch.zeros(self.nlayers, bsz, self.nhid))
        else:
            return torch.zeros(self.nlayers, bsz, self.nhid)

    def forward(self, X, epoch=0, test=False):
        X = torch.transpose(X, 0, 1)
        """Compute halting points and predictions"""
        if test: # Model chooses for itself during testing
            self.Controller._epsilon = 0.0
        else:
            self.Controller._epsilon = self._epsilon # explore/exploit trade-off
        T, B, V = X.shape
        baselines = [] # Predicted baselines
        actions = [] # Which classes to halt at each step
        log_pi = [] # Log probability of chosen actions
        halt_probs = []
        halt_points = -torch.ones((B, self.nclasses))
        hidden = self.initHidden(X.shape[1])
        predictions = torch.zeros((B, self.nclasses), requires_grad=True)
        all_preds = []

        # --- for each timestep, select a set of actions ---
        for t in range(T):
            # run Base RNN on new data at step t
            RNN_in = X[t].unsqueeze(0)
            
            output, hidden = self.RNN(RNN_in, hidden)

            # predict logits for all elements in the batch
            logits = self.out(output.squeeze())

            # compute halting probability and sample an action
            time = torch.tensor([t], dtype=torch.float, requires_grad=False).view(1, 1, 1).repeat(1, B, 1)
            c_in = torch.cat((output, time), dim=2).detach()
            a_t, p_t, w_t = self.Controller(c_in)

            # If a_t == 1 and this class hasn't been halted, save its logits
            predictions = torch.where((a_t == 1) & (predictions == 0), logits, predictions)

            # If a_t == 1 and this class hasn't been halted, save the time
            halt_points = torch.where((halt_points == -1) & (a_t == 1), time.squeeze(0), halt_points)

            # compute baseline
            b_t = self.BaselineNetwork(torch.cat((output, time), dim=2).detach())

            
            if B == 1:
                actions.append(a_t)
                baselines.append(b_t)
                log_pi.append(p_t)
            else:
                actions.append(a_t.squeeze())
                baselines.append(b_t.squeeze())
                log_pi.append(p_t)
            halt_probs.append(w_t)
            if (halt_points == -1).sum() == 0:  # If no negative values, every class has been halted
                break

        # If one element in the batch has not been halting, use its final prediction
        if B == 1:
            logits = torch.where(predictions == 0.0, logits, predictions)
        else:
            logits = torch.where(predictions == 0.0, logits, predictions).squeeze()
        halt_points = torch.where(halt_points == -1, time, halt_points).squeeze(0)
        self.locations = np.array(halt_points + 1)
        self.baselines = torch.stack(baselines).squeeze(1).transpose(0, 1)
        if B == 1:
            self.log_pi = torch.stack(log_pi).squeeze(2).transpose(0, 1)
        else:
            self.log_pi = torch.stack(log_pi).squeeze(1).squeeze(2).transpose(0, 1)

        self.halt_probs = torch.stack(halt_probs).transpose(0, 1).squeeze(2)
        
        self.actions = torch.stack(actions).transpose(0, 1)

        # --- Compute mask for where actions are updated ---
        # this lets us batch the algorithm and just set the rewards to 0
        # when the method has already halted one instances but not another.
        self.grad_mask = torch.zeros_like(self.actions)
        for b in range(B):
            self.grad_mask[b, :(1 + halt_points[b, 0]).long()] = 1
        
        if B == 1:
            return logits, (1+halt_points).mean()/(T+1)
        else:
            return logits.squeeze(), (1+halt_points).mean()/(T+1)

    def applyLoss(self, logits, y):
        # --- compute reward ---
        _, y_hat = torch.max(torch.softmax(logits, dim=1), dim=1)
        self.r = (2*(y_hat.float().round() == y.float()).float()-1).detach().unsqueeze(1)
        self.R = self.r * self.grad_mask

        # --- rescale reward with baseline ---
        b = self.grad_mask * self.baselines
        self.adjusted_reward = self.R - b.detach()

        # If you want a discount factor, that goes here!
        # It is used in the original implementation.

        # --- compute losses ---
        MSE = torch.nn.MSELoss()
        CE = torch.nn.CrossEntropyLoss()
        self.loss_b = MSE(b, self.R) # Baseline should approximate mean reward
        self.loss_r = (-self.log_pi*self.adjusted_reward).sum()/self.log_pi.shape[1] # RL loss
        self.loss_c = CE(logits, y) # Classification loss
        self.wait_penalty = self.halt_probs.sum(1).mean() # Penalize late predictions
        self.lam = torch.tensor([self.lam], dtype=torch.float, requires_grad=False)
        loss = self.loss_r + self.loss_b + self.loss_c + self.lam*(self.wait_penalty)
        # It can help to add a larger weight to self.loss_c so early training
        # focuses on classification: ... + 10*self.loss_c + ...
        return loss


class MDDNN(nn.Module):
    def __init__(self, input_size, time_dim, freq_dim, num_classes):
        super(MDDNN, self).__init__()
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=4,stride=1,padding=2),
            nn.BatchNorm1d(64,momentum=0.01),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2), 
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=16,stride=1,padding=8),
            nn.BatchNorm1d(32,momentum=0.01),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
        )
        
        self.lstm = nn.LSTM(32, 32, batch_first=True, num_layers=2)
                
        self.conv1f = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=4,stride=1,padding=2),
            nn.BatchNorm1d(64,momentum=0.01),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2), 
        )
        
        self.conv2f = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=16,stride=1,padding=8),
            nn.BatchNorm1d(32,momentum=0.01),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
        )
        
        self.lstmf = nn.LSTM(32, 32, batch_first=True, num_layers=2)
        
        self.fc = nn.Sequential(
            nn.Linear((self.time_dim*8 + self.freq_dim*32) , 32),
            nn.Dropout(0.1)
        )
        
        self.output = nn.Sequential(
            nn.Linear(32, num_classes),
            #nn.Softmax(dim=1)
        )
        
    def forward(self, x, xf):
        
        x = self.conv1(x)
        
        x = self.conv2(x)

        x = torch.transpose(x, 1, 2)

        x, (xhn, xcn) = self.lstm(x)

        x = x.reshape(x.shape[0],-1)
        
        xf = self.conv1f(xf)
        xf = self.conv2f(xf)
        
        xf = torch.transpose(xf, 1, 2)
        xf, (xfhn, xfcn) = self.lstmf(xf)
        xf = xf.reshape(xf.shape[0],-1)
        
        out = torch.cat((x, xf), 1)
        
        out = self.fc(out)

        out = self.output(out)
        
        return out

SEQUENCE_PADDINGS_VALUE=0

def entropy(p):
    return -(p*torch.log(p)).sum(1)

class DualOutputRNN(EarlyClassificationModel):
    def __init__(self, input_dim=1, hidden_dims=3, nclasses=5, num_rnn_layers=1, dropout=0.2, bidirectional=False,
                 use_batchnorm=False, use_attention=False, use_layernorm=True, init_late=True):

        super(DualOutputRNN, self).__init__()

        self.nclasses=nclasses
        self.use_batchnorm = use_batchnorm
        self.use_attention = use_attention
        self.use_layernorm = use_layernorm
        self.d_model = num_rnn_layers*hidden_dims

        if not use_batchnorm and not self.use_layernorm:
            self.in_linear = nn.Linear(input_dim, input_dim, bias=True)

        if use_layernorm:
            # perform
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.lstmlayernorm = nn.LayerNorm(hidden_dims)

        self.inpad = nn.ConstantPad1d((3, 0), 0)
        self.inconv = nn.Conv1d(in_channels=input_dim,
                  out_channels=hidden_dims,
                  kernel_size=3)

        self.lstm = nn.LSTM(input_size=hidden_dims, hidden_size=hidden_dims, num_layers=num_rnn_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        if bidirectional: # if bidirectional we have twice as many hidden dims after lstm encoding...
            hidden_dims = hidden_dims * 2

        if use_attention:
            self.attention = Attention(hidden_dims, attention_type="dot")

        if use_batchnorm:
            self.bn = nn.BatchNorm1d(hidden_dims)

        self.linear_class = nn.Linear(hidden_dims, nclasses, bias=True)
        self.linear_dec = nn.Linear(hidden_dims, 1, bias=True)

        if init_late:
            torch.nn.init.normal_(self.linear_dec.bias, mean=-2e1, std=1e-1)

    def _logits(self, x):

        # get sequence lengths from the index of the first padded value
        # lengths = torch.argmax((x[:, 0, :] == SEQUENCE_PADDINGS_VALUE), dim=1)

        # print(lengths)
        
        # if no padded values insert sequencelength as sequencelength
        # lengths[lengths == 0] = maxsequencelength

        #lengths = torch.tensor(input_length)
        # sort sequences descending to prepare for packing
        #lengths, idxs = lengths.sort(0, descending=True)

        # order x in decreasing seequence lengths
        #x = x[idxs]

        #x = x.transpose(1,2)

        if not self.use_batchnorm and not self.use_layernorm:
            x = self.in_linear(x)

        if self.use_layernorm:
            x = self.inlayernorm(x)

        # b,d,t -> b,t,d
        b, t, d = x.shape

        # pad left
        x_padded = self.inpad(x.transpose(1,2))
        # conv
        x = self.inconv(x_padded).transpose(1,2)
        # cut left side of convolved length
        x = x[:, -t:, :]
        
        #packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        outputs, last_state_list = self.lstm.forward(x)
        #outputs, unpacked_lens = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)      
        
        if self.use_layernorm:
            outputs = self.lstmlayernorm(outputs)

        if self.use_batchnorm:
            b,t,d = outputs.shape
            o_ = outputs.view(b, -1, d).permute(0,2,1)
            outputs = self.bn(o_).permute(0, 2, 1).view(b,t,d)

        if self.use_attention:
            h, c = last_state_list

            query = c[-1]

            #query = self.bn_query(query)

            outputs, weights = self.attention(query.unsqueeze(1), outputs)
            #outputs, weights = self.attention(outputs, outputs)

            # repeat outputs to match non-attention model
            outputs = outputs.expand(b,t,d)

        logits = self.linear_class.forward(outputs)
        deltas = self.linear_dec.forward(outputs)

        deltas = torch.sigmoid(deltas).squeeze(2)

        pts, budget = self.attentionbudget(deltas)

        if self.use_attention:
            pts = weights

        return logits, deltas, pts, budget

    def forward(self,x):
        logits, deltas, pts, budget = self._logits(x)

        logprobabilities = F.log_softmax(logits, dim=2)
        # stack the lists to new tensor (b,d,t,h,w)
        return logprobabilities, deltas, pts, budget

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to "+path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state,**kwargs),path)

    def load(self, path):
        print("loading model from "+path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot
