from __future__ import print_function
import sys

import tensorflow as tf
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import math

class Octopus(nn.Module):
    def __init__(self, num_classes=2):
        super(Octopus, self).__init__()
        self.num_classes = num_classes
        self.classes =[]
        for i in range(num_classes):
            self.classes.append(0) 

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 15, (7,7), (2,2), (3,3)),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        self.conv2_reduce = nn.Sequential(
            nn.Conv2d(15, 11, (1, 1), (1, 1), (0, 0)),
            nn.BatchNorm2d(11),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(11, 45, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(45),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        #Inception 3 
        self.inc_3a = InceptionModule(45, out_1x1=14, out_3x3_r=21,\
            out_5x5_r=4, out_3x3=28, out_5x5=7, out_pool_proj=7)
        self.inc_3b = InceptionModule(56, out_1x1=28, out_3x3_r=28,\
            out_5x5_r=7, out_3x3=42, out_5x5=21, out_pool_proj=14)
        self.pool3 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode = True)
        #Inception 4a-4b
        self.inc_4a = InceptionModule(105, out_1x1=43, out_3x3_r=22,\
            out_5x5_r=4, out_3x3=46, out_5x5=11, out_pool_proj=15)
        self.inc_4b = InceptionModule(115, out_1x1=36, out_3x3_r=25,\
            out_5x5_r=6, out_3x3=50, out_5x5=15, out_pool_proj=15)
        #Branching Layers 
        self.branches = []
        for c in range(self.num_classes):
            self.branches.append(OctopusBranch(in_channels=116).cuda())
        #Aux output
        self.aux_classifier = OctopusAux(in_channels=116, num_classes= self.num_classes)
        self.main_classifier = nn.Softmax()
    
    def forward(self, x):
        #Shared layers
        x = self.conv1(x)
        x = self.conv2_reduce(x)
        x = self.conv2(x)
        x = self.inc_3a(x)
        x = self.inc_3b(x)
        x = self.pool3(x)
        x = self.inc_4a(x)
        x = self.inc_4b(x)
        #Aux classifier
        aux_out = self.aux_classifier(x)
        #Branching layers
        for c in range(self.num_classes):
            cls = self.branches[c](x)
            self.classes[c] = (cls)  
        
        #Main classifier
        main_cat = torch.cat(self.classes,1)
        main_out = self.main_classifier(main_cat)
        return main_out, aux_out


'''
    Branch for Octopus Architecture using Inception Layers
'''
class OctopusBranch(nn.Module):
    def __init__(self, in_channels):
        super(OctopusBranch, self).__init__()
        #Inception 4c-4e
        self.inc_4c = InceptionModule(116, out_1x1=3, out_3x3_r=3,\
            out_5x5_r=1, out_3x3=6, out_5x5=2, out_pool_proj=2)
        self.inc_4d = InceptionModule(13, out_1x1=4, out_3x3_r=5,\
            out_5x5_r=2, out_3x3=10, out_5x5=3, out_pool_proj=3)
        self.inc_4e = InceptionModule(20, out_1x1=8, out_3x3_r=5,\
            out_5x5_r=1, out_3x3=10, out_5x5=4, out_pool_proj=4)
        self.pool4 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode = True)
        #Inception 5
        self.inc_5a = InceptionModule(26, out_1x1=9, out_3x3_r=6,\
            out_5x5_r=2, out_3x3=11, out_5x5=5, out_pool_proj=5)
        self.inc_5b = InceptionModule(30, out_1x1=13, out_3x3_r=7,\
            out_5x5_r=2, out_3x3=13, out_5x5=5, out_pool_proj=5)
        #Branch Output
        self.branch_out = nn.Sequential(
        nn.AvgPool2d((7, 7), (1, 1), ceil_mode = True),
        Flatten(),
        nn.Linear(36,1))

    def forward(self, x):
        x = self.inc_4c(x)
        x = self.inc_4d(x)
        x = self.inc_4e(x)
        x = self.pool4(x)
        x = self.inc_5a(x)
        x = self.inc_5b(x)
        x = self.branch_out(x)
        return x

'''
    Inception Module
'''
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1,out_3x3_r,out_5x5_r, out_3x3, out_5x5,out_pool_proj):
        super(InceptionModule, self).__init__()
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, (1, 1), (1, 1), (0, 0)),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU())
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3_r, (1, 1), (1, 1), (0, 0)),
            nn.BatchNorm2d(out_3x3_r),
            nn.ReLU(),
            nn.Conv2d(out_3x3_r, out_3x3, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU())
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5_r, (1, 1), (1, 1), (0, 0)),
            nn.BatchNorm2d(out_5x5_r),
            nn.ReLU(),
            nn.Conv2d(out_5x5_r, out_5x5, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU())
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d((3, 3), (1, 1), (1, 1)),
            nn.Conv2d(in_channels, out_pool_proj, (1, 1), (1, 1), (0, 0)),
            nn.BatchNorm2d(out_pool_proj),
            nn.ReLU())

    def forward(self, x):
        a = self.branch_1x1(x)
        b = self.branch_3x3(x)
        c = self.branch_5x5(x)
        d = self.branch_pool(x)
        outputs = [a, b, c, d]
        return torch.cat(outputs,1)
        

'''
    Auxiliary Output for after layer 4b
'''
class OctopusAux(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(OctopusAux, self).__init__()
        self.aux = nn.Sequential(
            nn.AvgPool2d((5,5),(3,3),ceil_mode = True),
            nn.Conv2d(116, 128, (1,1),(1,1)),
            Flatten(),
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.4),
            nn.Softmax())
        
    def forward(self, x):
        x = self.aux(x)
        return x

'''
    Flatten from 3D to vector for dense layers
'''
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),-1)


            
        
