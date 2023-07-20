import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from core.layers import BaseCNN, BaseRNN, Discriminator
from core.loss import FocalLoss

from torchvision import models


class Flatten(nn.Module):
    def forward(self, input):
        return input.contiguous().view(input.size(0), -1)


def conv_block(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=17, stride=stride,
                     padding=8, groups=groups, bias=False, dilation=dilation)


def conv_subsumpling(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


class ECGHB(nn.Module):
    def __init__(self, input_size):
        super().__init__()
    
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=5,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1)   
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1)    
        ) 

        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1)    
        ) 

        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=256, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            GlobalMaxPool1d(),
            nn.Dropout(0.2)    
        ) 

        self.dense = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = x.view(x.size(0), 256)
        x = self.dense(x)
        return x    


# # Ref. : A deep convolutional neural network model to classify heartbeats 
# #( https://www.sciencedirect.com/science/article/pii/S0010482517302810?via%3Dihub )
# class D3CNN(nn.Module): 
#     def __init__(self, input_size):
#         super().__init__()
#     
#         self.conv_1 = nn.Sequential(
#             nn.Conv1d(in_channels=input_size, out_channels=5, kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2,2)
#         )
#         
#         self.conv_2 = nn.Sequential(
#             nn.Conv1d(in_channels=5, out_channels=10, kernel_size=4,stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2,2)
#         )
#         
#         self.conv_3 = nn.Sequential(
#             nn.Conv1d(in_channels=10, out_channels=20, kernel_size=4,stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2,2)
#         )
#
#         self.dense_1 = nn.Sequential(
#             nn.Linear(20*124, 30),
#             nn.ReLU(),
#         )
#
#         self.dense_2 = nn.Sequential(
#             nn.Linear(30, 20),
#             nn.ReLU(),
#         )
#
#     def forward(self, x):
#         x = x.permute(0,2,1)
#         x = self.conv_1(x)
#         x = self.conv_2(x)
#         x = self.conv_3(x)
#         x = x.view(x.size(0), 20*124)
#         x = self.dense_1(x)
#         x = self.dense_2(x)
#         return x    

# Ref. : A deep convolutional neural network model to classify heartbeats 
#( https://www.sciencedirect.com/science/article/pii/S0010482517302810?via%3Dihub )
class D3CNN(nn.Module): 
    def __init__(self, input_size):
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
            nn.Linear(256, 128), 
            #nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.features(x)
        x = x.view(x.size(0), 128 * 246)
        x = self.classifier(x)
        return x


class ZolotyhNet(nn.Module): #https://arxiv.org/abs/2002.00254
    def __init__(self, input_size, num_classes=8):
        super().__init__()

        self.features_up = nn.Sequential(
            nn.Conv1d(input_size, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            Flatten(),
        )

        self.features_down = nn.Sequential(
            Flatten(),
            nn.Linear(12000,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 62)
        )

        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        x = x.permute(0,2,1) 
        out_up = self.features_up(x)
        out_down = self.features_down(x)
        
        #print(out_up.shape,out_down.shape)
        
        out = out_up + out_down

        #out = self.classifier(out_middle)

        return out


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


class EcgResNet34(nn.Module): #https://arxiv.org/pdf/1707.01836.pdf

    def __init__(self, layers=(1, 5, 5, 5), num_classes=256, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, block=BasicBlock):

        super(EcgResNet34, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = conv_block(12, self.inplanes, stride=1,)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_subsumpling(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0,2,1) 
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

