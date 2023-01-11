#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import torch
import os

from helpers.composite_helpers import *


from numba import cuda 


"""
"""
"""
COMPUTING PARAMETERS
"""
"""
"""

os.environ["CUDA_VISIBLE_DEVICES"]="2"
device = cuda.get_current_device()
device.reset()

# set the number of threads that pytorch will use
torch.set_num_threads(2)

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "Using device: " + str( device ), flush=True)

torch.manual_seed(8)
np.random.seed(8)


"""
"""
"""
RUN PARAMETERS
"""
"""
"""

hyperparameters_dict_eval = {"n_epochs": 200,
                          "batch_size": 256,
                          "lr": 0.001,
                          "num_bootstrap": 1,
                           "patience": 10}


# directories

curtains_dir = "/global/home/users/rrmastandrea/curtains/fresh/"

n_points = 100000
n_features = 5
dataset_config_string = f"LHCO_minmax_bidirectional/"

exp_dir = os.path.join(curtains_dir, dataset_config_string)
data_dir = os.path.join(curtains_dir, dataset_config_string, "data")

print("Making results directory at", exp_dir, "...")
os.makedirs(exp_dir, exist_ok=True)

data_dir = os.path.join(exp_dir, "data")


# dataset generation parameters
context_endpoints = (3000, 4000)


bands_dict = {"ob1": [3000, 3200],
              "sb1": [3200, 3400],
              "sr" : [3400, 3600],
              "sb2": [3600, 3800],
              "ob2": [3800, 4000]}

binning_scheme = np.linspace(-3.5, 3.5, 50)


"""
"""
"""
LOAD IN THE DATASETS AND PROCESS
"""
"""
"""

dataset_sim = ToyDataset(data_dir, "data_sim.npy")
dataset_dat = ToyDataset(data_dir, "data_dat.npy")
               

print("Num SIM events:", len(dataset_sim))
print("Num DAT events:", len(dataset_dat))
print()

# Pull data from SB1 + SB2
training_dataset_sim = dataset_sim.pull_from_mass_range([bands_dict["sb1"], bands_dict["sb2"]])
training_dataset_dat = dataset_dat.pull_from_mass_range([bands_dict["sb1"], bands_dict["sb2"]])

print("SIM sb1 + sb2 shape:", len(training_dataset_sim))
print("DAT sb1 + sb2 shape:", len(training_dataset_dat))
print()

# Preprocess the data
print("Preproccessing data...")
print()
training_dataset_sim.minmaxscale()
training_dataset_dat.minmaxscale()


# split into train val
val_split = 0.2

sim_val_size = int(val_split * len(training_dataset_sim))
dat_val_size = int(val_split * len(training_dataset_dat))

sim_train_size = len(training_dataset_sim) - sim_val_size
dat_train_size = len(training_dataset_dat) - dat_val_size


sim_train_dataset, sim_val_dataset = torch.utils.data.random_split(training_dataset_sim, [sim_train_size, sim_val_size])
dat_train_dataset, dat_val_dataset = torch.utils.data.random_split(training_dataset_dat, [dat_train_size, dat_val_size])


"""
"
"
LEARN BASE DENSITIES
"
"
"""

# Training SIM base density
num_layers_BD_sim = 8
num_hidden_features_BD_sim = 64
hyperparameters_dict_BD_sim = {"n_epochs": 50,
                              "batch_size": 128,
                              "lr": 0.0001,
                              "weight_decay": 0.0001}

loc_id_BD_sim = f"BDSIM_Masked_PRQ_AR_{num_layers_BD_sim}layers_{num_hidden_features_BD_sim}hidden_LRCos" # to distingush the saved models

# Define a flow architecture
transforms_BD_sim = make_masked_AR_flow(num_layers_BD_sim, n_features, num_hidden_features_BD_sim)
base_dist_sim = ConditionalDiagonalNormal(shape=[n_features], context_encoder=nn.Linear(1, 2*n_features))

#
#

# Training DAT base density
num_layers_BD_dat = 8
num_hidden_features_BD_dat = 64
hyperparameters_dict_BD_dat = {"n_epochs": 50,
                              "batch_size": 128,
                              "lr": 0.0001,
                              "weight_decay": 0.0001}

loc_id_BD_dat = f"BDDAT_Masked_PRQ_AR_{num_layers_BD_dat}layers_{num_hidden_features_BD_dat}hidden_LRCos" # to distingush the saved models

# Define a flow architecture
transforms_BD_dat = make_masked_AR_flow(num_layers_BD_dat, n_features, num_hidden_features_BD_dat)
base_dist_dat = ConditionalDiagonalNormal(shape=[n_features], context_encoder=nn.Linear(1, 2*n_features))


#
#


# Create and train
#create_and_train_flow("BDSIM", exp_dir, loc_id_BD_sim, transforms_BD_sim, base_dist_sim, hyperparameters_dict_BD_sim, device, sim_train_dataset, sim_val_dataset, early_stop = True)

#evaluate_base_density(hyperparameters_dict_BD_sim, "BDSIM", exp_dir, loc_id_BD_sim, device, bands_dict, n_features, dataset_sim, binning_scheme, hyperparameters_dict_eval)



# Create and train
#create_and_train_flow("BDDAT", exp_dir, loc_id_BD_dat, transforms_BD_dat, base_dist_dat, hyperparameters_dict_BD_dat, device, dat_train_dataset, dat_val_dataset, early_stop = True)



#evaluate_base_density(hyperparameters_dict_BD_dat, "BDDAT", exp_dir, loc_id_BD_dat, device, bands_dict, n_features, dataset_dat, binning_scheme, hyperparameters_dict_eval)


        


"""
"
"
EVALUATE SIM -> DAT
"
"
"""

# only uses the sim BD

evaluate_simBD_to_datBD(hyperparameters_dict_BD_sim, hyperparameters_dict_BD_dat, exp_dir, loc_id_BD_sim, loc_id_BD_dat, device, bands_dict, n_features, dataset_sim, dataset_dat, binning_scheme, hyperparameters_dict_eval)
