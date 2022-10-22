#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import os
from numba import cuda 

from helpers.composite_helpers import *


# load in the reverse rescales
path_to_minmax = "/global/home/users/rrmastandrea/CURTAINS_SALAD/LHCO_STS/data/col_minmax.npy"
col_minmax = np.load(path_to_minmax)

"""
"""
"""
COMPUTING PARAMETERS
"""
"""
"""

os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
num_signal_to_inject = 0
oversampnum = 5

eval_sim2real = False
eval_sim2real_oversamp = True
eval_cathode = False
eval_curtains = False
eval_full_sup = False


epochs_NN =  100
batch_size_NN = 128
lr_NN = 0.001
patience_NN = 20


context_endpoints = (2500, 4500)


bands_dict = {"ob1": [2500, 2900],
              "sb1": [2900, 3300],
              "sr" : [3300, 3700],
              "sb2": [3700, 4100],
              "ob2": [4100, 4500]}

binning_scheme = np.linspace(-3.5, 3.5, 50)

results_dir = f"/global/home/users/rrmastandrea/NF_results/nsig_inj{num_signal_to_inject}_seed{seed}/"
os.makedirs(results_dir, exist_ok=True)


"""
"""
"""
STS DATA
"""
"""
"""

STS_dir = "/global/home/users/rrmastandrea/CURTAINS_SALAD/LHCO_STS/data/"

STS_bkg = ToyDataset(STS_dir, "STS_bkg.npy")
STS_sig = ToyDataset(STS_dir, "STS_sig.npy")

STS_bkg_dataset = STS_bkg.pull_from_mass_range([bands_dict["sr"]])
STS_sig_dataset = STS_sig.pull_from_mass_range([bands_dict["sr"]])

STS_bkg_dataset = minmaxscale(STS_bkg_dataset.data, col_minmax, lower = 0, upper = 1, forward = True)
STS_sig_dataset = minmaxscale(STS_sig_dataset.data, col_minmax, lower = 0, upper = 1, forward = True)





"""
"""
"""
EVAL SIM2REAL
"""
"""
"""

# sim2real

sim2real_exp_dir = f"/global/home/users/rrmastandrea/CURTAINS_SALAD/LHCO_{num_signal_to_inject}sig_f/"

num_layers_BD_sim = 1
num_hidden_features_BD_sim = 128
num_blocks = 15

num_layers_s2d = 2
num_nodes_s2d = 16



loc_id_BD_sim = f"BD_sim_Masked_PRQ_AR_{num_layers_BD_sim}layers_{num_hidden_features_BD_sim}hidden_{num_blocks}blocks_{seed}seed"
loc_id_s2d = f"PRQ_Coupling_{num_layers_s2d}layers_{num_nodes_s2d}nodes_{seed}seed"
BD_sim_training_dir = os.path.join(sim2real_exp_dir, f"saved_models_{loc_id_BD_sim}/")
s2d_training_dir = os.path.join(BD_sim_training_dir, f"saved_models_{loc_id_s2d}/")
s2d_samples_dir = os.path.join(s2d_training_dir, f"npy_samples/")

oversamples_dir = os.path.join(s2d_training_dir, f"oversampling_{oversampnum}/")



# load in the dat samples
dat_samples_train = np.load(os.path.join(s2d_samples_dir, f"sr_DAT.npy")) 
dat_samples_train = minmaxscale(dat_samples_train, col_minmax, lower = -3, upper = 3, forward = False)
dat_samples_train = minmaxscale(dat_samples_train, col_minmax, lower = 0, upper = 1, forward = True)

# cathode
cathode_exp_dir = f"/global/home/users/rrmastandrea/CATHODE/CATHODE_models/nsig_inj{num_signal_to_inject}/seed{seed}/"

# curtains
curtains_exp_dir = f"/global/home/users/rrmastandrea/curtains/images/NSF_CURT_{num_signal_to_inject}sig_seed{seed}/Transformer/evaluation/"


for seed_NN in range(0, 20, 1):
    
    print(f"On classifier seed {seed_NN}...")

    if eval_sim2real:

        print("Evaluating sim2real...")

        transformed_sim_samples_train = np.load(os.path.join(s2d_samples_dir, f"sr_transSIM.npy"))
        transformed_sim_samples_train = minmaxscale(transformed_sim_samples_train, col_minmax, lower = -3, upper = 3, forward = False)
        transformed_sim_samples_train = minmaxscale(transformed_sim_samples_train, col_minmax, lower = 0, upper = 1, forward = True)

        roc = analyze_band_transform(results_dir, f"sim2real_{seed_NN}", transformed_sim_samples_train[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/sim2real_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(20*"*")
        print()
        
        
    if eval_sim2real_oversamp:

        print("Evaluating sim2real oversampled...")

        oversampled_sim_samples_train = np.load(os.path.join(oversamples_dir, f"transBD.npy"))
        oversampled_sim_samples_train = minmaxscale(oversampled_sim_samples_train, col_minmax, lower = -3, upper = 3, forward = False)
        oversampled_sim_samples_train = minmaxscale(oversampled_sim_samples_train, col_minmax, lower = 0, upper = 1, forward = True)

        roc = analyze_band_transform(results_dir, f"sim2real_oversamp{oversampnum}_{seed_NN}", oversampled_sim_samples_train[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/sim2real_oversamp_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(20*"*")
        print()


    if eval_cathode:

        print("Evaluating cathode...")

        cathode_trans_samps = np.load(os.path.join(cathode_exp_dir, f"SR_samples.npy"))
        
        roc = analyze_band_transform(results_dir, f"cathode_{seed_NN}", cathode_trans_samps[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/cathode_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(20*"*")
        print()


    if eval_curtains:

        print("Evaluating curtains...")

        curtains_trans_samps = np.load(os.path.join(curtains_exp_dir, f"samples_sb1_2_to_sr.npy"))
        curtains_trans_samps = minmaxscale(curtains_trans_samps, col_minmax, lower = 0, upper = 1, forward = True)

        roc = analyze_band_transform(results_dir, f"curtains_{seed_NN}", curtains_trans_samps[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/curtains_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(20*"*")
        print()
          
print("Done!")



"""
"
"
SUPERVISED CLASSIFIER
"
"""



if eval_full_sup:
    
    results_dir = f"/global/home/users/rrmastandrea/NF_results/nsig_inj0_seed1/"
    os.makedirs(results_dir, exist_ok=True)
    
    true_samples_dir = f"/global/home/users/rrmastandrea/CURTAINS_SALAD/LHCO_STS/data/"

 
    print("Evaluating fully supervised case...")
    
    true_samples_dir = f"/global/home/users/rrmastandrea/CURTAINS_SALAD/LHCO_STS/data/"

    true_sup_bkg = np.load(os.path.join(true_samples_dir, f"true_sup_bkg.npy"))
    true_sup_sig = np.load(os.path.join(true_samples_dir, f"true_sup_sig.npy"))

    true_sup_bkg = minmaxscale(true_sup_bkg, col_minmax, lower = 0, upper = 1, forward = True)
    true_sup_sig = minmaxscale(true_sup_sig, col_minmax, lower = 0, upper = 1, forward = True)







    for seed_NN in range(10, 20, 1):
    
        print(f"On classifier seed {seed_NN}...")

        
       

        roc = analyze_band_transform(results_dir, f"full_sup_{seed_NN}",true_sup_bkg[:,:-1], true_sup_sig[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/full_sup_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(20*"*")
        print()
          
print("Done!")




