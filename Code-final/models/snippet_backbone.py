import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# Ref. : A deep convolutional neural network model to classify heartbeats 
#( https://www.sciencedirect.com/science/article/pii/S0010482517302810?via%3Dihub )
class D3CNN(nn.Module): 
    def __init__(self, input_size):
        super().__init__()
    
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=5, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(2,2)
        )
        
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=10, kernel_size=4,stride=1,padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(2,2)
        )
        
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=4,stride=1,padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(2,2)
        )

        self.dense_1 = nn.Sequential(
            nn.Linear(20*124, 30),
            nn.LeakyReLU(),
        )

        self.dense_2 = nn.Sequential(
            nn.Linear(30, 256),
            nn.LeakyReLU(),
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


#https://ieeexplore.ieee.org/document/8952723
class HeartNetIEEE(nn.Module): 
    def __init__(self, input_size, num_classes=8):
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
            #nn.Linear(128, 256)
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.features(x)
        x = x.view(x.size(0), 128 * 246)
        x = self.classifier(x)
        return x

#Residual CNN from https://arxiv.org/abs/1805.00794
#https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8419425&fbclid=IwAR3It8GPhacfPhTMJLWNUGEPdLnunyON36MzN2S_rtcWbBPi-7zpUTGAzo4
class ResCNN(nn.Module): 
    def __init__(self, input_size):
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
                
        self.dense = nn.Sequential(
            nn.Linear(896, 32),
            nn.ReLU(),
            nn.Linear(32, 256),
        )
        
    def forward(self, x):
        x = x.permute(0,2,1) 
        x = self.conv1(x)
        x = self.block1(x,x)
        x = self.block2(x,x)
        x = self.block3(x,x)
        x = self.block4(x,x)
        x = self.block5(x,x)
        
        
        x = x.view(x.size(0), 896)
          
        x = self.dense(x)
        
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
    def __init__(self, input_size = 32, out_size = 32, kernal_size = 5, pool_size = 5):
        super(ResidualBlock, self).__init__()
        #padding = dilation * (kernel -1) / 2
        self.conv1 = nn.Conv1d(input_size, out_size, kernel_size=kernal_size, padding=2)
        self.conv2 = nn.Conv1d(input_size, out_size, kernel_size=kernal_size, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size, stride=2)
        
    def forward(self,x, shortcut):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.add(shortcut)
        x = self.relu(x)
        x = self.pool(x)
        
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_block(inplanes, planes, stride)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_block(planes, planes)
        self.bn2 = norm_layer(planes)
        self.dropout = nn.Dropout()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out

