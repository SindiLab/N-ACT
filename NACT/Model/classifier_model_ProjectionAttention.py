from ..utils import *

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parallel import data_parallel

## local
from .ProjectionBlock.projector_block import Projection
from .ProjectionBlock.pointWiseFF import PointWiseFeedForward

class NACT_ProjectionAttention(torch.nn.Module):
    def __init__(self, batch_size:int=10, D_in:int=5000, D_out:int=11,
                 dropout:float=0.0, num_heads:int=10, device:str='cpu'):
        """
        Params
        ------
            batch_size: int
                The batch size, used for resizing the tensors.
            D_in: int
                The dimension of each element of the sequence.
            D_out: int 
                The dimension of the desired predicted quantity.
            dropout: float
                The dropout probability
            num_heads: int
        """

        super(NACT_ProjectionAttention, self).__init__()
        self.device = device;
        # Net Config
        self.batch_size = batch_size
        self.n_features = D_in
        self.out_dim = D_out
        self.d_attention = D_in;

        #------------layers
        # additive attention layer
        self.attention = nn.Linear(self.n_features, self.n_features)

        # multiheaded attention layers
        self.proj_block = Projection(model_dim=self.d_attention,
                                               num_heads=num_heads,
                                               dropout=dropout)
        
        self.proj_block2 = Projection(model_dim=self.d_attention,
                                               num_heads=num_heads,
                                               dropout=dropout)

        self.pwff = PointWiseFeedForward(self.d_attention,
                                         hidden_dims=[128, self.d_attention],
                                         use_1x1Conv=False)

        
        self.nact_out_layer = nn.Sequential(nn.Linear(self.d_attention, self.out_dim),
                                            nn.LeakyReLU())
        
        print("Number of MH Correlative-Attention Blocks: 2")

#---------- With Residual Connection (Performs much better!)
    def forward(self, x:Tensor, training:bool=True, device = 'cpu'):
        """
        Forward pass for the Feed Forward Attention network.
        
        Params
        ------
        x: Tensor
            Input data for training
        training:bool
            The mode that we are calling the forward function. True means we are training the model

        Returns
        -------
        logits: Tensor
            A tensor of logits for predictions
        gse: Tensor
            A tensor of last gene stacked event (gse) from the last multi-head layer
        attn: Tensor
            A tensor of attention from the last multi-head layer
            
        """
        self.training = training
        
        if self.training:
            device = self.device
        else:
            device = device
        #------- Residual connection makes a big difference in our experiments -------#
        ##------- Make sure to keep (or at least ablation test) x_activated + x_c -------##
        batch_size = x.shape[0];
        
        alpha = self.softmax(self.attention(x));
        # "context" here is just torch.mul(alpha,x)
        x_c = self.context(alpha, x)
        ## x_c == Gamma in our paper!
        
        gse = self.attend(self.proj_block, x_c, device=device);
        x_activated = self.pwff(gse);
        
        gse2 = self.attend(self.proj_block2, x_activated+x_c, device=device);
        x_activated2 = self.pwff(gse2);
        
        logits = self.predict(x_activated2+x_c, device=device)
                
        #------- if we want to go full donkey kong -------#
        ## our experiments did not show notable improvements
        # gse3 = self.attend(self.proj_block3, x_activated2 + x_reduced, device=device);
        # x_activated3 = self.pwff(gse3);
        # logits = self.predict(x_activated3 + x_reduced, device=device)

        # gse4 = self.attend(self.proj_block3, x_activated3 + x_reduced, device=device);
        # x_activated4 = self.pwff(gse4);
        # logits = self.predict(x_activated4 + x_reduced, device=device)
        
        return logits, alpha, x_c

        
    def encode(self, x, device='cpu'):
        """
        Call to encoder for CPUs and parallel GPUs
        """
        if device == 'cpu':
            x_reduced = self.nact_dim_red(x);

        else:
            x_reduced= data_parallel(self.nact_dim_red, x)

        return x_reduced


    def attend(self, multi_head_module, x_c, device='cpu'):
        """
        Call the attention mechanism for CPUs and parallel GPUs
        """
        if device == 'cpu':
            context = multi_head_module(x_context=x_c);

        else:
            context = data_parallel(multi_head_module, inputs=(x_c))

        return context
    

    def predict(self, context, device='cpu'):
        """
        
        Call to encoder for CPUs and parallel GPUs
        
        """
        if device == 'cpu':
            x_out = self.nact_out_layer(context);

        else:
            x_out = data_parallel(self.nact_out_layer, context)

        return x_out
    
    
    def softmax(self, e_t):
        """
        Softmax (or sparsemax) activation
        """
        #------ SparseMax
        # from .SparseMax import Sparsemax
        # sparsemax = Sparsemax(dim=1)
        # alphas = sparsemax(e_t)

        #------ SoftMax
        softmax = torch.nn.Softmax(dim=1)
        alphas = softmax(e_t)
        return alphas
    
    

    def context(self, alpha_t, x_t):
        """
        The "Context" call, which is just the Hadamard product between x_t and attention values
        """
        return torch.mul(alpha_t, x_t)
