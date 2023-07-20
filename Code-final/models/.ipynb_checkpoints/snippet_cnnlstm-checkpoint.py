import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from core.layers import BaseCNN, BaseRNN, Discriminator
from core.loss import FocalLoss

# Ref. : A deep convolutional neural network model to classify heartbeats 
#( https://www.sciencedirect.com/science/article/pii/S0010482517302810?via%3Dihub )
class D3CNN(nn.Module): 
    def __init__(self, input_size = 12, hidden_size = 256, hidden_output_size = 1, output_size = 9):
        super().__init__()
    
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=5, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )
        
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=10, kernel_size=4,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )
        
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=4,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )

        self.dense_1 = nn.Sequential(
            nn.Linear(20*124, 256),
            nn.ReLU(),
        )

        self.dense_2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(x.size(0), 20*124)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x   


#https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8952723&tag=1
class HeartNetIEEE(nn.Module): 
    def __init__(self, input_size = 12, hidden_size = 256, hidden_output_size = 1, output_size = 9):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 246, 256),
            nn.Linear(256, 256),
        )
        

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.features(x)
        x = x.view(x.size(0), 128 * 246)
        x = self.classifier(x)
        return x


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
        '''
        @params:
                         x: input, tensor of shape (batch_size, n_channels, seq_len)
                 @return: The result of the maximum pooling of the time series, a tensor of shape (batch_size, n_channels, 1)
        '''
        return F.max_pool1d(x, kernel_size=x.shape[2]) # kenerl_size=seq_len


class ResidualBlock(nn.Module):
    def __init__(self, input_size = 32, out_size = 32, kernal_size = 5, pool_size = 2):
        super(ResidualBlock, self).__init__()
        #padding = dilation * (kernel -1) / 2
        self.conv1 = nn.Conv1d(input_size, out_size, kernel_size=kernal_size, padding=2)
        self.conv2 = nn.Conv1d(input_size, out_size, kernel_size=kernal_size, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size)
        
    def forward(self,x, shortcut):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.add(shortcut)
        x = self.relu(x)
        x = self.pool(x)
        
        return x


#Residual CNN from https://arxiv.org/abs/1805.00794
#https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8419425&fbclid=IwAR3It8GPhacfPhTMJLWNUGEPdLnunyON36MzN2S_rtcWbBPi-7zpUTGAzo4
class ResCNN(nn.Module): 
    def __init__(self, input_size = 12, hidden_size = 256, hidden_output_size = 1, output_size = 9):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, padding=1),
            nn.ReLU()
        )
        
        self.block1 = ResidualBlock()
        self.block2 = ResidualBlock()
        self.block3 = ResidualBlock()
        self.block4 = ResidualBlock()
        self.block5 = ResidualBlock()
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.globalmax = nn.Sequential(
            GlobalMaxPool1d(),
            nn.Dropout(0.2) 
        )
        
        self.dense = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        
    def forward(self, x):
        x = x.permute(0,2,1) 
        x = self.conv1(x)
        x = self.block1(x,x)
        x = self.block2(x,x)
        x = self.block3(x,x)
        x = self.block4(x,x)
        x = self.block5(x,x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        
        x = self.globalmax(x)
        x = x.view(x.size(0), 256)
          
        x = self.dense(x)
        
        return x


class snippet_cnnlstm(nn.Module):

    def __init__(self, input_size = 12, 
                 hidden_size = 256, 
                 hidden_output_size = 1, 
                 output_size = 9, 
                 core_model ="CNNLSTM",
                 isCuda = True):

        super(snippet_cnnlstm, self).__init__()
        self.loss_func = FocalLoss()
        self.CELL_TYPE = "LSTM"
        self.INPUT_SIZE = input_size
        self.HIDDEN_SIZE = hidden_size
        self.HIDDEN_OUTPUT_SIZE = hidden_output_size
        self.OUTPUT_SIZE = output_size
        self.CORE = core_model
        self.isCuda = isCuda

        # --- Backbones ---
        
        print(core_model)
        
        if (core_model == "D3CNN"):
            self.BaseCNN = D3CNN(input_size, hidden_size, hidden_output_size, output_size)
        elif(core_model == "HeartNetIEEE"):
            self.BaseCNN = HeartNetIEEE(input_size, hidden_size, hidden_output_size, output_size)
        elif(core_model == "ResCNN"):
            self.BaseCNN = ResCNN(input_size, hidden_size, hidden_output_size, output_size)
        else:
            self.BaseCNN = BaseCNN(input_size, hidden_size, output_size)
            self.BaseRNN = BaseRNN(hidden_size, hidden_size, self.CELL_TYPE).cuda()
        
        self.Discriminator = Discriminator(hidden_size, output_size)
        
        if(isCuda):
            self.BaseCNN = self.BaseCNN.cuda()
            self.Discriminator = self.Discriminator.cuda()
        
        
    def initHidden(self, batch_size, weight_size, isCuda = True):

        """Initialize hidden states"""

        if(isCuda):
            if self.CELL_TYPE == "LSTM":
                h = (torch.zeros(1, batch_size, weight_size).cuda(),
                     torch.zeros(1, batch_size, weight_size).cuda())
            else:
                h = torch.zeros(1, batch_size, weight_size).cuda()
        else:
            if self.CELL_TYPE == "LSTM":
                h = (torch.zeros(1, batch_size, weight_size),
                     torch.zeros(1, batch_size, weight_size))
            else:
                h = torch.zeros(1, batch_size, weight_size)
                
        return h


    def forward(self, X):
        
        hidden = self.initHidden(len(X), self.HIDDEN_SIZE, self.isCuda)
        min_length = 1000
        max_length = 0
        for x in X:
            if min_length > x.shape[0]:
                min_length = x.shape[0]
            if max_length < x.shape[0]:
                max_length = x.shape[0]
        
        tau_list = np.zeros(len(X), dtype=int)
        
        for t in range(max_length):
            slice_input = []
            cnn_input = None # cpu
            for idx, x in enumerate(X):
                slice_input.append(x[tau_list[idx],:,:])
                cnn_input = torch.stack(slice_input, dim=0)

            if(self.CORE == "CNNLSTM"):
                S_t = self.BaseCNN(cnn_input)

                cnn_input.detach()

                S_t = S_t.unsqueeze(0)

                S_t, hidden = self.BaseRNN(S_t, hidden) # Run sequence model

                for idx in range(len(X)):
                    if(tau_list[idx] < X[idx].shape[0]-1):
                        tau_list[idx]+=1
                S_t = hidden[0][-1]
            elif(self.CORE == "CNNLSTM-500"):
                S_t = self.BaseCNN(cnn_input)

                cnn_input.detach()

                S_t = S_t.unsqueeze(0)

                S_t, hidden = self.BaseRNN(S_t, hidden) # Run sequence model

                for idx in range(len(X)):
                    if(tau_list[idx] < X[idx].shape[0]-1):
                        tau_list[idx]+=1
                S_t = hidden[0][-1]
            else:
                S_t = self.BaseCNN(cnn_input)
                cnn_input.detach()
            
            result = self.Discriminator(S_t)
          
        return result 

    def predict(self, X):

        hidden = self.initHidden(len(X), self.HIDDEN_SIZE)
        min_length = 1000
        max_length = 0
        for x in X:
            if min_length > x.shape[0]:
                min_length = x.shape[0]
            if max_length < x.shape[0]:
                max_length = x.shape[0]
        
        tau_list = np.zeros(X.shape[0], dtype=int)
        
        Hidden_states = []
        for t in range(max_length):
            slice_input = []
            cnn_input = None # cpu
            for idx, x in enumerate(X):
                #print(x.shape)
                slice_input.append(torch.from_numpy(x[t,:,:]).float())
                cnn_input = torch.stack(slice_input, dim=0)
            
            if(self.isCuda):
                cnn_input = cnn_input.cuda()

            if(self.CORE == "CNNLSTM"):
                S_t = self.BaseCNN(cnn_input)
                cnn_input.detach()
                S_t = S_t.unsqueeze(0)

                S_t, hidden = self.BaseRNN(S_t,hidden) # Run sequence model

                S_t = hidden[0][-1]
            elif(self.CORE == "CNNLSTM-500"):
                S_t = self.BaseCNN(cnn_input)
                cnn_input.detach()
                S_t = S_t.unsqueeze(0)

                S_t, hidden = self.BaseRNN(S_t,hidden) # Run sequence model

                S_t = hidden[0][-1]
            else:
                S_t = self.BaseCNN(cnn_input)
                cnn_input.detach()
                S_t = S_t.unsqueeze(0)

                S_t, hidden = self.BaseRNN(S_t,hidden) # Run sequence model

                S_t = hidden[0][-1]
                
            Hidden_states.append(S_t.cpu().detach().numpy())
            
        return Hidden_states
    
    def inference(self, X, hidden):
        
        cnn_input = torch.from_numpy(X).float()
            
        if(self.isCuda):
            cnn_input = cnn_input.cuda()

        if(self.CORE == "CNNLSTM"):
            S_t = self.BaseCNN(cnn_input)
            cnn_input.detach()
            S_t = S_t.unsqueeze(0)
            S_t, hidden = self.BaseRNN(S_t,hidden) # Run sequence model
            S_t = hidden[0][-1]
        elif(self.CORE == "CNNLSTM-500"):
            S_t = self.BaseCNN(cnn_input)
            cnn_input.detach()
            S_t = S_t.unsqueeze(0)
            S_t, hidden = self.BaseRNN(S_t,hidden) # Run sequence model
            S_t = hidden[0][-1]
        else:
            S_t = self.BaseCNN(cnn_input)
            cnn_input.detach()
            S_t = S_t.unsqueeze(0)
            S_t, hidden = self.BaseRNN(S_t,hidden) # Run sequence model
            S_t = hidden[0][-1]
                
        return S_t.cpu().detach().numpy(), hidden




