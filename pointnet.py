from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from datasets import PartDataset
from utils import Timer
import torch.nn.functional as F

# Define PointNet model for classification
class T_Net(nn.Module):
    def __init__(self):
        super(T_Net, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1   = nn.Linear(1024, 512)
        self.fc2   = nn.Linear(512, 256)
        self.fc3   = nn.Linear(256, 9)
        self.relu  = nn.ReLU()

        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(1024)
        self.bn4   = nn.BatchNorm1d(512)
        self.bn5   = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]              # [32,    3, 2500]
        x = F.relu(self.bn1(self.conv1(x)))  # [32,   64, 2500]
        x = F.relu(self.bn2(self.conv2(x)))  # [32,  128, 2500]
        x = F.relu(self.bn3(self.conv3(x)))  # [32, 1024, 2500]
        x = torch.max(x, 2, keepdim=True)[0] # [32, 1024, 1]
        x = x.view(-1, 1024)                 # [32, 1024]
        x = F.relu(self.bn4(self.fc1(x)))    # [32, 512]
        x = F.relu(self.bn5(self.fc2(x)))    # [32, 256]
        x = self.fc3(x)                      # [32, 9]

        """ 9 dimension -> 3 x 3"""
        iden = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class OrderInvariance(nn.Module):
    def __init__(self, symmetric='max'):
        super(OrderInvariance, self).__init__()
        self.T_net = T_Net()

        self.symmetric = symmetric
                
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(1024)

        self.unsort_MLP = torch.nn.Conv1d(2500, 1, 1);

    def forward(self, x):
        # x: [32, 3, 2500]
        batchsize= x.size()[0]                      # 32
        n_pts    = x.size()[2]                      # 2500
        """ T-NET """
        trans    = self.T_net(x)                    # [32, 3, 3]
        x        = x.transpose(2,1)                 # [32, 2500, 3]
#         # batch-level matrix multiplication
#         x        = torch.bmm(x, trans) 

        """ Permutation invariance """
        if self.symmetric == 'unsorted':
           x     = self.unsort_MLP( x );

        x        = x.transpose(2,1)                 # [ 32, 3, 2500]
        x        = F.relu(self.bn1(self.conv1(x)))  # [32,  64, 2500]
        x        = F.relu(self.bn2(self.conv2(x)))  # [32, 128, 2500]
        x        = self.bn3(self.conv3(x))          # [32,1024, 2500]

        """ permutation types """
        if   self.symmetric == 'max':
             x  = torch.max(x, 2, keepdim=True)[0] # [32,1024, 1]
        elif self.symmetric == 'avg':
             x  = torch.mean(x, 2, keepdim=True)   # [32,1024, 1]        
        else:
             x  = x;
        x        = x.view(-1, 1024)                 # [32,1024]

        return x, trans


class PointNetCls(nn.Module):
    def __init__(self, k = 2, symmetric='unsorted'):
        super(PointNetCls, self).__init__()

        self.symmetric = symmetric;
        self.name  = '{:s}_{:s}'.format(self.__class__.__name__, self.symmetric)
        
        self.feat = OrderInvariance(symmetric=symmetric) ;

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.feat(x)                    # [32, 1024]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)    