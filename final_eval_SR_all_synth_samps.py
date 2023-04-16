#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import os
from numba import cuda 
import argparse

from helpers.composite_helpers import *

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-n", "--num_signal_to_inject", help = "num signal to inject")
 
# Read arguments from command line
args = parser.parse_args()


"""
"""
"""
COMPUTING PARAMETERS
"""
"""
"""



os.environ["CUDA_VISIBLE_DEVICES"]= "0"
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
eval_feta_nonsamp = False # not oversampled
eval_cathode = True
eval_curtains = True
eval_salad = True
eval_full_sup = True
eval_combined = False


# parameters for combined samples
num_samples = 400000
# coefficients for mixing
# recommended to have them sum to 1 but there's no check on that

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

results_dir = f"/clusterfs/ml4hep/rrmastandrea/NF_results/nsig_inj{num_signal_to_inject}_seed{seed}/"
os.makedirs(results_dir, exist_ok=True)
scaled_data_dir = "/global/home/users/rrmastandrea/scaled_data/"






"""
"""
"""
STS DATA
"""
"""
"""


STS_bkg_dataset = np.load(f"{scaled_data_dir}/STS_bkg.npy")
STS_sig_dataset = np.load(f"{scaled_data_dir}/STS_sig.npy")

dat_samples_train = np.load(f"{scaled_data_dir}/nsig_injected_{num_signal_to_inject}/data.npy")


"""
"""
"""
EVAL 
"""
"""
"""


    
# load in the data samples
#feta_samples = np.load(f"{scaled_data_dir}/nsig_injected_{num_signal_to_inject}/feta.npy")
cathode_samples = np.load(f"{scaled_data_dir}/nsig_injected_{num_signal_to_inject}/cathode.npy")
curtains_samples = np.load(f"{scaled_data_dir}/nsig_injected_{num_signal_to_inject}/curtains.npy")
salad_samples = np.load(f"{scaled_data_dir}/nsig_injected_{num_signal_to_inject}/salad.npy")
salad_weights = np.load(f"{scaled_data_dir}/nsig_injected_{num_signal_to_inject}/salad_weights.npy").reshape(-1, 1)


blank_weights_samples = np.ones((num_samples, 1))
blank_weights_data = np.ones((dat_samples_train.shape[0], 1))


for seed_NN in range(index_start, index_stop, 1):
        
    # select samples
    indices_feta = np.random.choice(len(feta_samples), size = num_samples)
    selected_feta = feta_samples[indices_feta]
    
    indices_cathode = np.random.choice(len(cathode_samples), size = num_samples)
    selected_cathode = cathode_samples[indices_cathode]
    
    indices_curtains = np.random.choice(len(curtains_samples), size = num_samples)
    selected_curtains = curtains_samples[indices_curtains]
    
    indices_salad = np.random.choice(len(salad_samples), size = num_samples)
    selected_salad = salad_samples[indices_salad]
    selected_salad_weights = salad_weights[indices_salad]

    # concatenate 
    # shuffling *should* happen int the dataloader
    synth_samples = np.concatenate((selected_feta, selected_cathode, selected_curtains, selected_salad))
    synth_weights = np.concatenate((blank_weights_samples, blank_weights_samples, blank_weights_samples, selected_salad_weights))


    if eval_feta:
        print(f"Evaluating feta (seed {seed_NN} of {index_stop})...")
        
        roc = discriminate_datasets_weighted(results_dir, f"feta_{seed_NN}", selected_feta[:,:-1], dat_samples_train[:,:-1], blank_weights_samples, blank_weights_data, STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/feta_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(5*"*")
        print()

    if eval_cathode:
        
        print(f"Evaluating cathode (seed {seed_NN} of {index_stop})...")
        
        roc = discriminate_datasets_weighted(results_dir, f"cathode_{seed_NN}", selected_cathode[:,:-1], dat_samples_train[:,:-1], blank_weights_samples, blank_weights_data, STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/cathode_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(5*"*")
        print()


    if eval_curtains:

        print(f"Evaluating curtains (seed {seed_NN} of {index_stop})...")
        roc = discriminate_datasets_weighted(results_dir, f"curtains_{seed_NN}", selected_curtains[:,:-1], dat_samples_train[:,:-1], blank_weights_samples, blank_weights_data, STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        
  
        results_file = f"{results_dir}/curtains_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(5*"*")
        print()
        

    if eval_salad:

        print(f"Evaluating salad (seed {seed_NN} of {index_stop})...")
        roc = discriminate_datasets_weighted(results_dir, f"salad_{seed_NN}", selected_salad[:,:-1], dat_samples_train[:,:-1], selected_salad_weights, blank_weights_data, STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        
        
        results_file = f"{results_dir}/salad_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(5*"*")
        print()
        

    if eval_combined:

        print(f"Evaluating combined samples (seed {seed_NN} of {index_stop})...")

        roc = discriminate_datasets_weighted(results_dir, f"combined_{seed_NN}", synth_samples[:,:-1], dat_samples_train[:,:-1], synth_weights, blank_weights_data, STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)

        results_file = f"{results_dir}/combined_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(5*"*")
        print()





"""
"
"
SUPERVISED CLASSIFIER
"
"""


if eval_full_sup:
    
    results_dir = f"/global/home/users/rrmastandrea/NF_results_2/nsig_inj0_seed1/"
    os.makedirs(results_dir, exist_ok=True)
    
    
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


        roc = discriminate_datasets_weighted(results_dir, f"full_sup_{seed_NN}",true_sup_bkg[:,:-1], true_sup_sig[:,:-1], np.ones((true_sup_bkg.shape[0], 1)), np.ones((true_sup_sig.shape[0], 1)), STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/full_sup_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(20*"*")
        print()
        

print("Done!")
          








