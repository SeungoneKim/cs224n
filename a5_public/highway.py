#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self,embed_size):
        super(Highway, self).__init__()
        self.Wproj = nn.Linear(embed_size,embed_size,bias=True)
        self.Wgate = nn.Linear(embed_size,embed_size,bias=True)
        
    def forward(self, Xconv_out):
        Xproj = F.relu(self.Wproj(Xconv_out))
        Xgate = F.sigmoid(self.Wgate(Xconv_out))

        Xhighway = Xgate*Xproj + (1-Xgate)*Xconv_out

        return Xhighway

### END YOUR CODE 

