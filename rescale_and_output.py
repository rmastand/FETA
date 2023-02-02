#!/usr/bin/env python
# coding: utf-8

"""
Rescales everything to (0,1)
"""

import numpy as np
import os

from helpers.composite_helpers import *


# load in the reverse rescales
path_to_minmax = "/global/home/users/rrmastandrea/FETA/LHCO_STS/data/col_minmax.npy"
col_minmax = np.load(path_to_minmax)

"""
"""
"""
RUN PARAMETERS
"""
"""
"""

seed = 1
n_features = 5
num_signal_to_inject = 500
oversampnum = 6

rescale_feta_oversamp = True
rescale_cathode = True
rescale_curtains = True

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

output_dir = f"/global/home/users/rrmastandrea/scaled_data/"
os.makedirs(output_dir, exist_ok=True)

output_dir_nsig = f"{output_dir}/nsig_injected_{num_signal_to_inject}/"
os.makedirs(output_dir_nsig, exist_ok=True)


"""
"""
"""
STS DATA
"""
"""
"""

STS_dir = "/global/home/users/rrmastandrea/FETA/LHCO_STS/data/"

STS_bkg = ToyDataset(STS_dir, "STS_bkg.npy")
STS_sig = ToyDataset(STS_dir, "STS_sig.npy")

STS_bkg_dataset = STS_bkg.pull_from_mass_range([bands_dict["sr"]])
STS_sig_dataset = STS_sig.pull_from_mass_range([bands_dict["sr"]])

STS_bkg_dataset = minmaxscale(STS_bkg_dataset.data, col_minmax, lower = 0, upper = 1, forward = True)
STS_sig_dataset = minmaxscale(STS_sig_dataset.data, col_minmax, lower = 0, upper = 1, forward = True)

np.save(f"{output_dir}/STS_bkg", STS_bkg_dataset)
np.save(f"{output_dir}/STS_sig", STS_sig_dataset)


"""
"""
"""
RESCALE 
"""
"""
"""


sim2real_exp_dir = f"/global/home/users/rrmastandrea/FETA/LHCO_{num_signal_to_inject}sig_f/"

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

np.save(f"{output_dir_nsig}/data", dat_samples_train)

  
if rescale_feta_oversamp:
    
    print("Rescaling feta oversampled...")
    oversampled_sim_samples_train = np.load(os.path.join(oversamples_dir, f"transBD.npy"))
    oversampled_sim_samples_train = minmaxscale(oversampled_sim_samples_train, col_minmax, lower = -3, upper = 3, forward = False)
    oversampled_sim_samples_train = minmaxscale(oversampled_sim_samples_train, col_minmax, lower = 0, upper = 1, forward = True)
    np.save(f"{output_dir_nsig}/feta", oversampled_sim_samples_train)

# cathode
cathode_exp_dir = f"/global/home/users/rrmastandrea/CATHODE/CATHODE_models/nsig_inj{num_signal_to_inject}/seed{seed}/"

if rescale_cathode:

    print("Rescaling cathode...")
    cathode_trans_samps = np.load(os.path.join(cathode_exp_dir, f"SR_samples.npy"))
    # rescale the last index (masses)
    
    mmin = np.min(cathode_trans_samps[:,-1])
    mmax = np.max(cathode_trans_samps[:,-1])
    
    cathode_trans_samps[:,-1] = 0.33 + 0.34*(cathode_trans_samps[:,-1] - mmin)/(mmax - mmin)
    np.save(f"{output_dir_nsig}/cathode", cathode_trans_samps)

# curtains
curtains_exp_dir = f"/global/home/users/rrmastandrea/curtains/images/NSF_CURT_{num_signal_to_inject}sig_seed{seed}/Transformer/evaluation/"    
 
if rescale_curtains:

    print("Rescaling curtains...")
    
    
    samps = np.load(os.path.join(curtains_exp_dir, f"samples_sb1_2_to_sr.npz"))
    d = dict(zip(("data1{}".format(k) for k in samps), (samps[k] for k in samps)))
    curtains_trans_samps = d["data1arr_1"]
    curtains_trans_samps = minmaxscale(curtains_trans_samps, col_minmax, lower = 0, upper = 1, forward = True)

    np.save(f"{output_dir_nsig}/curtains", curtains_trans_samps)
        
