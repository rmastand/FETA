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
parser.add_argument("-c", "--cuda_slot", help = "which gpu")
parser.add_argument("-o", "--oversampled_num", help = "factor to oversample by")

 
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

eval_oversampled = True
eval_nonsampled = False


index_start = 20
index_stop = 50


epochs_NN =  100
batch_size_NN = 128
lr_NN = 0.001
patience_NN = 10


context_endpoints = (2500, 4500)


bands_dict = {"ob1": [2500, 2900],
              "sb1": [2900, 3300],
              "sr" : [3300, 3700],
              "sb2": [3700, 4100],
              "ob2": [4100, 4500]}

binning_scheme = np.linspace(-3.5, 3.5, 50)

results_dir = f"/clusterfs/ml4hep/rrmastandrea/oversampling_results/nsig_inj{args.num_signal_to_inject}_seed{seed}/"
os.makedirs(results_dir, exist_ok=True)
scaled_data_dir = "/global/home/users/rrmastandrea/oversamp_studies/"





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
feta_samples_osamp = np.load(f"{scaled_data_dir}/nsig_injected_{args.num_signal_to_inject}/feta_o{args.oversampled_num}.npy")
feta_samples_nonsamp = np.load(f"{scaled_data_dir}/nsig_injected_{args.num_signal_to_inject}/feta.npy")


blank_weights_osamp = np.ones((feta_samples_osamp.shape[0], 1))
blank_weights_nonsamp = np.ones((feta_samples_nonsamp.shape[0], 1))
blank_weights_data = np.ones((dat_samples_train.shape[0], 1))


for seed_NN in range(index_start, index_stop, 1):
        

    if eval_oversampled:
        print(f"Evaluating feta oversampled x{args.oversampled_num} (seed {seed_NN + 1} of {index_stop})...")
        
        roc = discriminate_datasets_weighted(results_dir, f"feta_o{args.oversampled_num}_{seed_NN}", feta_samples_osamp[:,:-1], dat_samples_train[:,:-1], blank_weights_osamp, blank_weights_data, STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/feta_o{args.oversampled_num}_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(5*"*")
        print()

    if eval_nonsampled:
        print(f"Evaluating feta notoversampled (seed {seed_NN + 1} of {index_stop})...")
        
        roc = discriminate_datasets_weighted(results_dir, f"feta_{seed_NN}", feta_samples_nonsamp[:,:-1], dat_samples_train[:,:-1], blank_weights_nonsamp, blank_weights_data, STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/feta_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(5*"*")
        print()







