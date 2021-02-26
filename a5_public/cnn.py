#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,input_channel_num,output_channel_num,kernel_size):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(input_channel_num,output_channel_num,kernel_size)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
    def forward(self, input):
        activations = F.relu(self.conv(input))
        output = self.maxpool(activations).squeeze(dim=2)
        return output

### END YOUR CODE

