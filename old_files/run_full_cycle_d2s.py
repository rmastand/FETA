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

os.environ["CUDA_VISIBLE_DEVICES"]="3"
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
                          "num_bootstrap": 10}


# directories

curtains_dir = "/global/home/users/rrmastandrea/curtains/fresh/"

n_points = 100000
n_features = 5
dataset_config_string = f"LHCO_D2S/"

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

binning_scheme = np.linspace(-4, 4, 50)


"""
"""
"""
LOAD IN THE DATASETS AND PROCESS
"""
"""
"""

dataset_sim = ToyDataset(os.path.join(data_dir, "data_sim.npy"))
dataset_dat = ToyDataset(os.path.join(data_dir,"data_dat.npy"))
               

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
LEARN BASE DENSITY
"
"
"""

# Training the BD
# This will be the upper subdirectory in saved_models/

num_layers_BD = 8
num_hidden_features_BD = 64
hyperparameters_dict_BD = {"n_epochs": 100,
                          "batch_size": 128,
                          "lr": 0.0005,
                          "weight_decay": 0.01}

loc_id_BD = f"Masked_PRQ_AR_{num_layers_BD}layers_{num_hidden_features_BD}hidden_LRCos" # to distingush the saved models

# Define a flow architecture
transforms_BD = make_masked_AR_flow(num_layers_BD, n_features, num_hidden_features_BD)
base_dist = ConditionalDiagonalNormal(shape=[n_features], context_encoder=nn.Linear(1, 2*n_features))

# Create and train
create_and_train_flow("BD", exp_dir, loc_id_BD, transforms_BD, base_dist, hyperparameters_dict_BD, device, dat_train_dataset, dat_val_dataset)


"""
"
"
EVALUATE BASE DENSITY
"
"
"""

evaluate_base_density(hyperparameters_dict_BD, exp_dir, loc_id_BD, device, bands_dict, n_features, dataset_dat, binning_scheme, hyperparameters_dict_eval)
        

"""
"
"
LEARN SIM -> DAT
"
"
"""

# Training s2d
# This will be another (of many) subdirectory in saved_models/

#num_layers_s2d = 2
#num_blocks_s2d = 4
#num_nodes_s2d = 64
#loc_id_s2d = f"PRQ_Coupling_{num_layers_s2d}layers_{num_blocks_s2d}blocks_{num_nodes_s2d}nodes_LRCos" # to distingush the saved models

num_layers_s2d = 8
num_hidden_features_s2d = 64
hyperparameters_dict_s2d = {"n_epochs": 50,
                          "batch_size": 256,
                          "lr": 0.0005,
                          "weight_decay": 0.0001}
loc_id_s2d = f"Masked_PRQ_AR_{num_layers_s2d}layers_{num_hidden_features_s2d}hidden_LRCos" # to distingush the saved models

# Define a flow architecture
architecture_file_s2d = "model_architecture_s2d.txt"
#transforms_s2d = make_coupling_flow(num_layers_s2d, n_features, num_nodes_s2d, num_blocks = num_blocks_s2d)
transforms_s2d = make_masked_AR_flow(num_layers_s2d, n_features, num_hidden_features_s2d)

# Special things for s2d training
inner_dir = os.path.join(exp_dir, f"saved_models_{loc_id_BD}/")

print("Loading the base density model ...")
print()
config_string_BD = "epochs{0}_lr{1}_wd{2}_bs{3}".format(hyperparameters_dict_BD["n_epochs"], hyperparameters_dict_BD["lr"], hyperparameters_dict_BD["weight_decay"], hyperparameters_dict_BD["batch_size"])
checkpoint_path_BD = os.path.join(inner_dir, f"BD_{config_string_BD}")

flow_BD = torch.load(f"{checkpoint_path_BD}_best_model.pt")
# send network to device
flow_BD.to(device)
# freeze the trained model
for param in flow_BD.parameters():
    param.requires_grad = False
flow_BD.eval()

# Create and train
create_and_train_flow("s2d", inner_dir, loc_id_s2d, transforms_s2d, flow_BD, hyperparameters_dict_s2d, device, sim_train_dataset, sim_val_dataset)


"""
"
"
EVALUATE SIM -> DAT
"
"
"""

evaluate_s2d_density(hyperparameters_dict_BD, hyperparameters_dict_s2d, exp_dir, loc_id_BD, loc_id_s2d, device, bands_dict, n_features, dataset_dat, dataset_sim, binning_scheme, hyperparameters_dict_eval)
