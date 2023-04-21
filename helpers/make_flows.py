import numpy as np

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform, PiecewiseCubicCouplingTransform

from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.lu import LULinear
from nflows.transforms.standard import IdentityTransform
from nflows.nn.nets import ResidualNet

from helpers.dense import dense_net


def make_masked_AR_flow(num_layers, num_features, num_hidden_features, num_blocks = 8, num_bins = 10, spline_type = "PiecewiseRationalQuadratic", perm_type = "Reverse", tail_bound = 3.5, tails = "linear", dropout = 0.05):
    
    transforms = []
    
    for _ in range(num_layers):
        
        if spline_type == "PiecewiseRationalQuadratic":
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features = num_features, hidden_features = num_hidden_features, num_blocks = num_blocks, tail_bound = tail_bound, context_features = 1, tails = tails, num_bins = num_bins, dropout_probability = dropout))     
            
        if perm_type == "Reverse":
            transforms.append(ReversePermutation(features = num_features)) 
            
    return transforms


def make_coupling_flow(num_stack, num_features, num_hidden_features, num_hidden_layers, num_bins = 4, spline_type = "PiecewiseRationalQuadratic", perm_type = "Reverse", tail_bound = 3.5, tails = "linear"):
    
    # first make the mask
    n_mask = int(np.ceil(num_features / 2))
    mx = [1] * n_mask + [0] * int(num_features - n_mask)

    # then make the maker
    # this has to be an nn.module that takes as first arg the input dim and second the output dim
    def maker(input_dim, output_dim):
        return dense_net(input_dim, output_dim, layers=[num_hidden_features] * num_hidden_layers, context_features=1)

    transforms = []

    for _ in range(num_stack):

        if spline_type == "PiecewiseRationalQuadratic":
            transforms.append(PiecewiseRationalQuadraticCouplingTransform(mx, maker, tail_bound = tail_bound, tails = tails, num_bins = num_bins))      
        
        if perm_type == "Reverse":
            transforms.append(ReversePermutation(features = num_features)) 
            
    return transforms


def make_joint_flow(num_layers, num_features, num_hidden_features, num_nodes, num_blocks_coup = 8, num_blocks_AR = 8,  num_bins = 10, perm_type = "Reverse", tail_bound = 3.5, tails = "linear", dropout = 0.05):
    
    # first make the mask
    n_mask = int(np.ceil(num_features / 2))
    mx = [1] * n_mask + [0] * int(num_features - n_mask)

    # then make the maker
    # this has to be an nn.module that takes as first arg the input dim and second the output dim
    def maker(input_dim, output_dim):
        return dense_net(input_dim, output_dim, layers=[num_nodes] * num_blocks_coup, context_features=1)

    transforms = []

    for _ in range(num_layers):
        
        transforms.append(PiecewiseRationalQuadraticCouplingTransform(mx, maker, tail_bound = tail_bound, tails = tails, num_bins = num_bins))     
        
        transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features = num_features, hidden_features = num_hidden_features, num_blocks = num_blocks_AR, tail_bound = tail_bound, context_features = 1, tails = tails, num_bins = num_bins, dropout_probability = dropout))     
        
        if perm_type == "Reverse":
            transforms.append(ReversePermutation(features = num_features)) 
            
    return transforms






        