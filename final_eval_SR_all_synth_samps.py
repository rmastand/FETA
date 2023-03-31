#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import os
from numba import cuda 

from helpers.composite_helpers import *

"""
"""
"""
COMPUTING PARAMETERS
"""
"""
"""

os.environ["CUDA_VISIBLE_DEVICES"]="3"
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


num_signal_to_inject = 1200
index_start = 0
index_stop = 20

eval_feta = True
eval_cathode = True
eval_curtains = True
eval_full_sup = True
eval_combined = True


# parameters for combined samples
num_synth_samples = 600000
# coefficients for mixing
# recommended to have them sum to 1 but there's no check on that
c_feta = .3333333333
c_cathode = 0.3333333333
c_curtains = 0.3333333333

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

# for the combined samples
num_feta = int(c_feta*num_synth_samples)
num_cathode = int(c_cathode*num_synth_samples)
num_curtains = int(c_curtains*num_synth_samples)
    
# load in the data samples
feta_samples = np.load(f"{scaled_data_dir}/nsig_injected_{num_signal_to_inject}/feta.npy")
cathode_samples = np.load(f"{scaled_data_dir}/nsig_injected_{num_signal_to_inject}/cathode.npy")
curtains_samples = np.load(f"{scaled_data_dir}/nsig_injected_{num_signal_to_inject}/curtains.npy")


for seed_NN in range(index_start, index_stop, 1):

    if eval_feta:
        print(f"Evaluating feta (seed {seed_NN} of {index_stop})...")
     
        roc = analyze_band_transform(results_dir, f"feta_{seed_NN}", feta_samples[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/feta_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(5*"*")
        print()

    if eval_cathode:
        
        print(f"Evaluating cathode (seed {seed_NN} of {index_stop})...")
        
        roc = analyze_band_transform(results_dir, f"cathode_{seed_NN}", cathode_samples[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/cathode_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(5*"*")
        print()


    if eval_curtains:

        print(f"Evaluating curtains (seed {seed_NN} of {index_stop})...")
        roc = analyze_band_transform(results_dir, f"curtains_{seed_NN}", curtains_samples[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        
  
        results_file = f"{results_dir}/curtains_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(5*"*")
        print()
        

    if eval_combined:

        print(f"Evaluating combined samples (seed {seed_NN} of {index_stop})...")

        # select samples
        indices_feta = np.random.choice(len(feta_samples), size = num_feta)
        selected_feta = feta_samples[indices_feta]
        indices_cathode = np.random.choice(len(cathode_samples), size = num_cathode)
        selected_cathode = cathode_samples[indices_cathode]
        indices_curtains = np.random.choice(len(curtains_samples), size = num_curtains)
        selected_curtains = curtains_samples[indices_curtains]

        # concatenate and shuffle
        synth_samples = np.concatenate((selected_feta, selected_cathode, selected_curtains))
        np.random.shuffle(synth_samples)
        
        print(synth_samples.shape)

        roc = analyze_band_transform(results_dir, f"combined_{seed_NN}", synth_samples[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)

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


        roc = analyze_band_transform(results_dir, f"full_sup_{seed_NN}",true_sup_bkg[:,:-1], true_sup_sig[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/full_sup_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(20*"*")
        print()
        

print("Done!")
          








