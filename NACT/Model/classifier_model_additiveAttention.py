""" 
Additive attention + FFNN is based on paper by Colin Raffel and Daniel P. Ellis
https://arxiv.org/abs/1512.08756
"""
from ..utils import *
# in case one wants to try out sparsemax as well
# from .SparseMax.SparseMax import Sparsemax
## our results showed that SparseMax performs worse than Softmax, which is expected

import torch
import torch.nn as nn
import torch.nn.functional as F  

class NACT_AddAttn(torch.nn.Module):
    """
    Attributes:
        batch_size (int): The batch size, used for resizing the tensors.
        T (int): The length of the sequence.
        D_in (int): The dimension of each element of the sequence.
        D_out (int): The dimension of the desired predicted quantity.
        hidden (int): The dimension of the hidden state.
    """
    def __init__(self, batch_size=10,
                 T=10, 
                 D_in=17789, 
                 D_out=11, 
                 hidden=100):
        super(NACT_AddAttn, self).__init__()
        
        # Net Config
        self.batch_size = batch_size
        self.n_features = D_in
        self.out_dim = D_out
        self.hidden = hidden
        
        # attention_layer
        self.attention = nn.Linear(self.n_features, self.n_features)
        
                                        
        # hidden layers
        self.network = nn.Sequential(   nn.Linear(self.n_features, 1024),
                                        nn.Tanh(),
                                        nn.Linear(1024, 512),
                                        nn.Tanh(),
                                        nn.Linear(512, 256),
                                        nn.Tanh(),
                                        nn.Linear(256, 128),
                                        nn.Tanh(),
                                        nn.Linear(128, 64),
                                        nn.Tanh(),
                                        nn.Linear(64, self.out_dim),
                                        nn.Tanh()
                                    )
        
### Here will be the best configuration so far:
    def forward(self, x, training=True):
        """
        Forward pass for the Feed Forward Attention network.
        """
        
        self.training = training
        # attention
        alpha = self.softmax(self.attention(x));
        x_c = self.context(alpha, x)
        output = self.network(x_c)
        
        return output, alpha, x_c
        
    def softmax(self, e_t):
        """
        Step 3:
        Compute the probabilities alpha_t
        In : torch.Size([batch_size, sequence_length, 1])
        Out: torch.Size([batch_size, sequence_length, 1])
        """
        #### SparseMax
        # sparsemax = Sparsemax(dim=1)
        # alphas = sparsemax(e_t)

        #### SoftMax
        softmax = torch.nn.Softmax(dim=1)
        alphas = softmax(e_t)
        return alphas

    def context(self, alpha_t, x_t):
        """
        Step 4:
        Compute the context vector c
        In : torch.Size([batch_size, sequence_length, 1]), torch.Size([batch_size, sequence_length, sequence_dim])
        Out: torch.Size([batch_size, 1, hidden_dimensions])
        """
        return torch.mul(alpha_t, x_t)