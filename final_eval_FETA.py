#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import os
from numba import cuda 
import argparse

from helpers.synthesis_plots import *

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-n", "--num_signal_to_inject", help = "num signal to inject")
parser.add_argument("-c", "--cuda_slot", help = "CUDA slot")

 
# Read arguments from command line
args = parser.parse_args()


"""
"""
"""
COMPUTING PARAMETERS
"""
"""
"""


os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda_slot
device = cuda.get_current_device()
device.reset()

# set the number of threads that pytorch will use
torch.set_num_threads(2)

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "Using device: " + str( device ), flush=True)

"""
"""
"""
RUN PARAMETERS
"""
"""
"""

seed = 1
n_features = 5


index_start = 0
index_stop = 20

eval_feta = True
eval_cathode = True
eval_curtains = True
eval_salad = True

results_dir = f"/global/ml4hep/spss/rrmastandrea/NF_results_FETA/nsig_inj{args.num_signal_to_inject}_seed{seed}"


os.makedirs(results_dir, exist_ok=True)
os.makedirs(scatterplot_dir, exist_ok=True)

scaled_data_dir = "/global/home/users/rrmastandrea/scaled_data/"



epochs_NN =  100
batch_size_NN = 128 
lr_NN = 0.001
patience_NN = 10 

"""
"""
"""
STS DATA
"""
"""
"""


STS_bkg_dataset = np.load(f"{scaled_data_dir}/STS_bkg.npy")
STS_sig_dataset = np.load(f"{scaled_data_dir}/STS_sig.npy")

dat_samples_train = np.load(f"{scaled_data_dir}/nsig_injected_{args.num_signal_to_inject}/data.npy")


"""
"""
"""
EVAL 
"""
"""
"""

    
# load in the data samples
feta_samples = np.load(f"{scaled_data_dir}/nsig_injected_{args.num_signal_to_inject}/feta_o6.npy")
cathode_samples = np.load(f"{scaled_data_dir}/nsig_injected_{args.num_signal_to_inject}/cathode.npy")
curtains_samples = np.load(f"{scaled_data_dir}/nsig_injected_{args.num_signal_to_inject}/curtains.npy")
salad_samples = np.load(f"{scaled_data_dir}/nsig_injected_{args.num_signal_to_inject}/salad.npy")
base_salad_weights = np.load(f"{scaled_data_dir}/nsig_injected_{args.num_signal_to_inject}/salad_weights.npy").reshape(-1, 1)

num_synth_events = feta_samples.shape[0] + cathode_samples.shape[0] + curtains_samples.shape[0] + salad_samples.shape[0] 



blank_weights_data = np.ones((dat_samples_train.shape[0], 1))


for seed_NN in range(index_start, index_stop, 1):
    
    np.random.seed(seed_NN)


    if eval_feta:
        print(f"Evaluating feta with {args.num_signal_to_inject} events (seed {seed_NN} of {index_stop})...")
        
        roc, feta_results = discriminate_datasets_weighted(results_dir, f"feta_{seed_NN}", feta_samples[:,:n_features], dat_samples_train[:,:n_features], np.ones((feta_samples.shape[0], 1)), blank_weights_data, STS_bkg_dataset[:,:n_features], STS_sig_dataset[:,:n_features], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        
        results_file = f"{results_dir}/feta_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")
        np.save(f"{results_dir}/feta_results_seedNN{seed_NN}_nsig{args.num_signal_to_inject}", feta_results)

        print()
        print(5*"*")
        print()

    if eval_cathode:
        
        print(f"Evaluating cathode with {args.num_signal_to_inject} events (seed {seed_NN} of {index_stop})...")
        
        roc, cathode_results = discriminate_datasets_weighted(results_dir, f"cathode_{seed_NN}", cathode_samples[:,:n_features], dat_samples_train[:,:n_features], np.ones((cathode_samples.shape[0], 1)), blank_weights_data, STS_bkg_dataset[:,:n_features], STS_sig_dataset[:,:n_features], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        
        results_file = f"{results_dir}/cathode_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")
        np.save(f"{results_dir}/cathode_results_seedNN{seed_NN}_nsig{args.num_signal_to_inject}", cathode_results)

        print()
        print(5*"*")
        print()


    if eval_curtains:

        print(f"Evaluating curtains with {args.num_signal_to_inject} events (seed {seed_NN} of {index_stop})...")
        
        roc, curtains_results = discriminate_datasets_weighted(results_dir, f"curtains_{seed_NN}", curtains_samples[:,:n_features], dat_samples_train[:,:n_features], np.ones((curtains_samples.shape[0], 1)), blank_weights_data, STS_bkg_dataset[:,:n_features], STS_sig_dataset[:,:n_features], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        
        results_file = f"{results_dir}/curtains_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")
            
        np.save(f"{results_dir}/curtains_results_seedNN{seed_NN}_nsig{args.num_signal_to_inject}", curtains_results)

        print()
        print(5*"*")
        print()
        

    if eval_salad:

        print(f"Evaluating salad with {args.num_signal_to_inject} events (seed {seed_NN} of {index_stop})...")
        
        roc, salad_results = discriminate_datasets_weighted(results_dir, f"salad_{seed_NN}", salad_samples[:,:n_features], dat_samples_train[:,:n_features], base_salad_weights, blank_weights_data, STS_bkg_dataset[:,:n_features], STS_sig_dataset[:,:n_features], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        
        
        results_file = f"{results_dir}/salad_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")
            
        np.save(f"{results_dir}/salad_results_seedNN{seed_NN}_nsig{args.num_signal_to_inject}", salad_results)

        print()
        print(5*"*")
        print()
        


"""
"
"
SUPERVISED CLASSIFIER
"
"""

if int(args.num_signal_to_inject) == 0:
    
    
    # load in the non STS labeled samples
    # load in the reverse rescales
    path_to_minmax = "/global/home/users/rrmastandrea/FETA/LHCO_STS/data/col_minmax.npy"
    col_minmax = np.load(path_to_minmax)
    true_samples_dir = f"/global/home/users/rrmastandrea/FETA/LHCO_STS/data/"

    true_sup_bkg = np.load(os.path.join(true_samples_dir, f"true_sup_bkg.npy"))
    true_sup_sig = np.load(os.path.join(true_samples_dir, f"true_sup_sig.npy"))
    true_sup_bkg = minmaxscale(true_sup_bkg, col_minmax, lower = 0, upper = 1, forward = True)
    true_sup_sig = minmaxscale(true_sup_sig, col_minmax, lower = 0, upper = 1, forward = True)


    for seed_NN in range(index_start, index_stop, 1):
    
        print(f"Evaluating full sup (seed {seed_NN} of {index_stop})...")


        roc, full_sup_results = discriminate_datasets_weighted(results_dir, f"full_sup_{seed_NN}",true_sup_bkg[:,:n_features], true_sup_sig[:,:n_features], np.ones((true_sup_bkg.shape[0], 1)), np.ones((true_sup_sig.shape[0], 1)), STS_bkg_dataset[:,:n_features], STS_sig_dataset[:,:n_features], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/full_sup_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")
            
        np.save(f"{results_dir}/full_sup_results_seedNN{seed_NN}", full_sup_results)

        print()
        print(20*"*")
        print()
        
        
        

print("Done!")
         
