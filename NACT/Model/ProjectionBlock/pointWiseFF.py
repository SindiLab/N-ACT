from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import *
import torch.nn.functional as F

class PointWiseFeedForward(nn.Module):

    def __init__(self, inp_dim:int, hidden_dims=[1024, 512], use_1x1Conv:bool=False):
        '''
        A point-wise FF network (from Vawani et al. -> )
            
        Params
        ------
        inp_dim:int
            The input dimension (coming from the attention layer)
        hidden_dims: List
            A list of two integers that determine the hidden layers
        use_1x1Conv: bool 
            Whether we want to use a 1x1 conv to represent the pointwise-fully connected network 
        '''
        
        super(PointWiseFeedForward, self).__init__()
        self.inp_dim = inp_dim
        self.hidden_dims = hidden_dims
        # in our experiments, nn.Linear is faster than nn.Conv1d
        self.conv = use_1x1Conv
        
        if self.conv:
            params = {'in_channels': self.inp_dim, 'out_channels': self.hidden_dims[0],
                      'kernel_size': 1, 'stride': 1, 'bias': True}
            self.conv1 = nn.Sequential(nn.Conv1d(**params), nn.ReLU())
            params = {'in_channels': self.hidden_dims[0], 'out_channels': self.hidden_dims[1],
                      'kernel_size': 1, 'stride': 1, 'bias': True}
            self.conv2 = nn.Conv1d(**params)
        else:
            self.conv1 = nn.Sequential(nn.Linear(self.inp_dim, self.hidden_dims[0]), nn.ReLU())
            self.conv2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
            
        self.normalization = nn.LayerNorm(self.inp_dim)

    def forward(self, inputs):
        '''
        The forward call of the PWFF mechanism
            
        Params
        ------
        inputs: Tensor
            The input that we need to pass through the network
        
        Returns
        -------
        outputs: Tensor 
            The outputs of the forward pass of the network
        
        '''
        if self.conv:
            inputs = inputs.permute(0, 1)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

        # Residual connection
        outputs += inputs

        # Layer normalization
        if self.conv:
            outputs = self.normalization(outputs.permute(0, 2, 1))
        else:
            outputs = self.normalization(outputs)

        return outputs