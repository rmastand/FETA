#!/usr/bin/env python
# coding: utf-8

"""
Rescales everything to (0,1)
"""

import numpy as np
import os
import argparse

import yaml
with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)

from helpers.composite_helpers import *


project_id = workflow["project_id"]
context_endpoints = workflow["context_endpoints"]
bands_dict = workflow["bands_dict"]
# load in the reverse rescales
path_to_minmax = f"/global/home/users/rrmastandrea/FETA/LHCO_STS_{project_id}/data/col_minmax.npy"
col_minmax = np.load(path_to_minmax)

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-n", "--num_signal_to_inject", help = "num signal to inject")
 
# Read arguments from command line
args = parser.parse_args()

"""
"""
"""
RUN PARAMETERS
"""
"""
"""


seed = 2
n_features = 5
oversampnum = 6

rescale_feta_oversamp = True
rescale_feta_nonsamp = False
rescale_cathode = False
rescale_curtains = False
rescale_herwig = False




output_dir = f"/global/home/users/rrmastandrea/scaled_data_{project_id}_seed_{seed}/"
#output_dir = f"/global/home/users/rrmastandrea/oversamp_studies_8_2_32_10/"
os.makedirs(output_dir, exist_ok=True)

output_dir_nsig = f"{output_dir}/nsig_injected_{args.num_signal_to_inject}/"
os.makedirs(output_dir_nsig, exist_ok=True)


"""
"""
"""
STS DATA
"""
"""
"""

STS_dir = f"/global/home/users/rrmastandrea/FETA/LHCO_STS_{project_id}/data/"

#STS_bkg = ToyDataset(STS_dir, "STS_bkg.npy")
STS_sig = ToyDataset(STS_dir, "STS_sig.npy")

#STS_bkg_dataset = STS_bkg.pull_from_mass_range([bands_dict["sr"]])
STS_sig_dataset = STS_sig.pull_from_mass_range([bands_dict["sr"]])

#STS_bkg_dataset = minmaxscale(STS_bkg_dataset.data, col_minmax, lower = 0, upper = 1, forward = True)
STS_sig_dataset = minmaxscale(STS_sig_dataset.data, col_minmax, lower = 0, upper = 1, forward = True)

#np.save(f"{output_dir}/STS_bkg", STS_bkg_dataset)
np.save(f"{output_dir}/STS_sig", STS_sig_dataset)


"""
"""
"""
RESCALE 
"""
"""
"""


sim2real_exp_dir = f"/global/home/users/rrmastandrea/FETA/LHCO_{args.num_signal_to_inject}sig_{project_id}/"

num_layers_BD_sim = 1
num_hidden_features_BD_sim = 128
num_blocks = 15

num_stack_s2d = 8
num_hidden_layers_s2d = 2
num_hidden_features_s2d = 32
num_bins_s2d = 10


loc_id_BD_sim = f"BD_sim_Masked_PRQ_AR_{num_layers_BD_sim}layers_{num_hidden_features_BD_sim}hidden_{num_blocks}blocks_{seed}seed"
loc_id_s2d = f"PRQ_Coupling_{num_stack_s2d}layers_{num_hidden_layers_s2d}hiddenlayers_{num_hidden_features_s2d}hiddenfeatures_{num_bins_s2d}bins_{seed}seed"
BD_sim_training_dir = os.path.join(sim2real_exp_dir, f"saved_models_{loc_id_BD_sim}/")
s2d_training_dir = os.path.join(BD_sim_training_dir, f"saved_models_{loc_id_s2d}/")
s2d_samples_dir = os.path.join(s2d_training_dir, f"npy_samples/")
oversamples_dir = os.path.join(s2d_training_dir, f"oversampling_{oversampnum}/")

# load in the dat samples

dat_dir = f"{sim2real_exp_dir}/data/classif_train_dat.npy"
dat_samples_train = np.load(dat_dir) 
dat_samples_train = minmaxscale(dat_samples_train, col_minmax, lower = 0, upper = 1, forward = True)

np.save(f"{output_dir_nsig}/data", dat_samples_train)

  
if rescale_feta_oversamp:
    
    print("Rescaling feta oversampled...")
    oversampled_sim_samples_train = np.load(os.path.join(oversamples_dir, f"transBD.npy"))
    oversampled_sim_samples_train = minmaxscale(oversampled_sim_samples_train, col_minmax, lower = -3, upper = 3, forward = False)
    oversampled_sim_samples_train = minmaxscale(oversampled_sim_samples_train, col_minmax, lower = 0, upper = 1, forward = True)
    np.save(f"{output_dir_nsig}/feta_o{oversampnum}", oversampled_sim_samples_train)
    
    
if rescale_feta_nonsamp:
    print("Rescaling feta (not oversampled)...")
    nonsampled_sim_samples_train = np.load(os.path.join(s2d_samples_dir, f"sr_transSIM.npy"))
    nonsampled_sim_samples_train = minmaxscale(nonsampled_sim_samples_train, col_minmax, lower = -3, upper = 3, forward = False)
    nonsampled_sim_samples_train = minmaxscale(nonsampled_sim_samples_train, col_minmax, lower = 0, upper = 1, forward = True)
    np.save(f"{output_dir_nsig}/feta", nonsampled_sim_samples_train)

# cathode
#cathode_exp_dir = f"/global/ml4hep/spss/rrmastandrea/CATHODE/CATHODE_models/nsig_inj{args.num_signal_to_inject}/seed{seed}/"
cathode_exp_dir = f"/global/home/users/rrmastandrea/CATHODE/CATHODE_models_{project_id}/nsig_inj{args.num_signal_to_inject}/seed{seed}/"

if rescale_cathode:

    print("Rescaling cathode...")
    cathode_trans_samps = np.load(os.path.join(cathode_exp_dir, f"SR_samples.npy"))
    # rescale the last index (masses)
    
    mmin = np.min(cathode_trans_samps[:,-1])
    mmax = np.max(cathode_trans_samps[:,-1])
    
    #cathode_trans_samps[:,-1] = 0.33 + 0.34*(cathode_trans_samps[:,-1] - mmin)/(mmax - mmin)
    cathode_trans_samps[:,-1] = 0.45 + 0.10*(cathode_trans_samps[:,-1] - mmin)/(mmax - mmin)
    np.save(f"{output_dir_nsig}/cathode", cathode_trans_samps)

# curtains
curtains_exp_dir = f"/global/home/users/rrmastandrea/curtains/images/NSF_CURT_{args.num_signal_to_inject}sig_seed{seed}_{project_id}/Transformer/evaluation/"    
 
if rescale_curtains:

    print("Rescaling curtains...")
    samps = np.load(os.path.join(curtains_exp_dir, f"samples_sb1_2_to_sr.npz"))
    d = dict(zip(("data1{}".format(k) for k in samps), (samps[k] for k in samps)))
    curtains_trans_samps = d["data1arr_1"]
    curtains_trans_samps = minmaxscale(curtains_trans_samps, col_minmax, lower = 0, upper = 1, forward = True)

    np.save(f"{output_dir_nsig}/curtains", curtains_trans_samps)
    
    
    
# herwig

def pull_from_mass_range(dataset, mass_ranges):

        """
        mass_ranges is a list of lists [low_bound, high_bound]
        """

        selected_data = []

        for mass_range in mass_ranges:
            loc = np.where((dataset[:, 5] >= mass_range[0]) & (dataset[:, 5] < mass_range[1]))[0]
            selected_data.append( dataset[loc, :] )
        selected_data = np.concatenate(selected_data)
        np.random.shuffle(selected_data)

        return selected_data



herwig_dir = f"/global/home/users/rrmastandrea/FETA/LHCO_{args.num_signal_to_inject}sig_f/data/"    
 
if rescale_herwig:

    print("Rescaling herwig...")
    
    
    herwig_samps = np.load(os.path.join(herwig_dir, f"train_sim.npy"))
    herwig_samps = pull_from_mass_range(herwig_samps, [bands_dict["sr"]])
    herwig_samps = minmaxscale(herwig_samps, col_minmax, lower = 0, upper = 1, forward = True)

    np.save(f"{output_dir_nsig}/herwig", herwig_samps)
        
