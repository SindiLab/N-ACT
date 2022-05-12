import math
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple

class Projection(nn.Module):
    def __init__(self, model_dim:int=5000, num_heads:int=10, dropout:float=0.0):
        
        ''' 
        Multi-Headed Projection
        
        Params
        ------
        model_dim: int
            The dimension of the projection module
        dropout: float
            The rate of dropout for the attention nodes
        num_heads: int
            The number of heads
        
        '''
        super(Projection, self).__init__()
        # making sure the numbers match
        assert model_dim % num_heads == 0, "Dimension of the attention module mod number of heads should == 0"
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.d_head = int(self.model_dim/self.num_heads)
        self.dropout = dropout
        # linear projections done in one tensor (similar to Vaswani et al.) for efficiency
        self.projection = nn.Linear(self.model_dim, self.d_head * self.num_heads)
        
        # initializing dropout and normalization layers
        self.output_dropout = nn.Dropout(p=self.dropout)
        self.normalization =  nn.LayerNorm(self.model_dim)

    def forward(self, x_context:Tensor) -> Tuple[Tensor]:
        """
        Forward pass for computing the multi-headed attention
        
        Params
        ------
        x_context: Tensor
            The input tensor
            
        Returns
        -------
        x_norm: Tensor
            The gene scores (after residual + layer norm) 
        
        """
        batch_size = x_context.size(0)

        # Linear projections
        x_proj = self.projection(x_context)
        
        x_all_heads = x_proj.view(batch_size, -1, self.num_heads, self.d_head)
        x = x_all_heads.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        
        # Restore input shape
        x_inp_shape = torch.cat(torch.chunk(x, self.num_heads, dim=0), dim=2)
        x = x_inp_shape.squeeze()
        
        #-------- Residual connection (works best in our experiments)
        x += x_context
        
        x_norm = self.normalization(x)  # (N, T_q, C)

        return x_norm