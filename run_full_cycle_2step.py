#!/usr/bin/env python
# coding: utf-8

# fancy slack notifications
from knockknock import slack_sender
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import os
from numba import cuda 

from helpers.composite_helpers import *

message = "0 1 test"

webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="is-my-code-done", user_mentions=["@Radha Mastandrea"])

def main(message):
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

    seed = 1
    
    oversample = 4
 

    """
    """
    """
    RUN PARAMETERS
    """
    """
    """

    hyperparameters_dict_eval = {"n_epochs": 100,
                              "batch_size": 128,
                              "lr": 0.001,
                               "patience": 20
                                }
    

    # directories

    curtains_dir = "/global/home/users/rrmastandrea/CURTAINS_SALAD/"

    n_features = 5
    num_signal_to_inject = 1000
    dataset_config_string = f"LHCO_{num_signal_to_inject}sig_f/"
    STS_config_sting = f"LHCO_STS/"

    exp_dir = os.path.join(curtains_dir, dataset_config_string)
    data_dir = os.path.join(exp_dir, "data")

    print("Making results directory at", exp_dir, "...")
    os.makedirs(exp_dir, exist_ok=True)
    
    # load in the reverse rescales
    path_to_minmax = "/global/home/users/rrmastandrea/CURTAINS_SALAD/LHCO_STS/data/col_minmax.npy"
    col_minmax = np.load(path_to_minmax)

   
    # dataset generation parameters
    context_endpoints = (2500, 4500)


    bands_dict = {"ob1": [2500, 2900],
                  "sb1": [2900, 3300],
                  "sr" : [3300, 3700],
                  "sb2": [3700, 4100],
                  "ob2": [4100, 4500]}

    binning_scheme = np.linspace(-3.5, 3.5, 50)


    """
    """
    """
    LOAD IN THE DATASETS AND PROCESS
    """
    """
    """

    npull_dataset_train_sim = ToyDataset(data_dir, "train_sim.npy")
    npull_dataset_val_sim = ToyDataset(data_dir, "val_sim.npy")
    npull_dataset_train_dat = ToyDataset(data_dir, "train_dat.npy")
    npull_dataset_val_dat = ToyDataset(data_dir, "val_dat.npy")

    print("Num SIM events in SB:", len(npull_dataset_train_sim)+len(npull_dataset_val_sim))
    print("Num DAT events in SB:", len(npull_dataset_train_dat)+len(npull_dataset_val_dat))
    print()

    # Preprocess the data
    print("Preproccessing data...")
    print()
    dataset_train_sim = npull_dataset_train_sim.pull_from_mass_range([bands_dict["sb1"], bands_dict["sb2"]])
    dataset_val_sim = npull_dataset_val_sim.pull_from_mass_range([bands_dict["sb1"], bands_dict["sb2"]])
    dataset_train_dat = npull_dataset_train_dat.pull_from_mass_range([bands_dict["sb1"], bands_dict["sb2"]])
    dataset_val_dat = npull_dataset_val_dat.pull_from_mass_range([bands_dict["sb1"], bands_dict["sb2"]])
    
    dataset_train_sim = minmaxscale(dataset_train_sim.data, col_minmax, lower = -3, upper = 3, forward = True)
    dataset_val_sim = minmaxscale(dataset_val_sim.data, col_minmax, lower = -3, upper = 3, forward = True)
    dataset_train_dat = minmaxscale(dataset_train_dat.data, col_minmax, lower = -3, upper = 3, forward = True)
    dataset_val_dat = minmaxscale(dataset_val_dat.data, col_minmax, lower = -3, upper = 3, forward = True)


    """
    "
    "
    LEARN SIM BASE DENSITY
    "
    "
    """

    # Training the BD
    # This will be the upper subdirectory in saved_models/

    num_layers_BD_sim = 8
    num_hidden_features_BD_sim = 64
    hyperparameters_dict_BD_sim = {"n_epochs": 60,
                              "batch_size": 128,
                              "lr": 0.0005,
                              "weight_decay": 0.001}

    loc_id_BD_sim = f"BD_sim_Masked_PRQ_AR_{num_layers_BD_sim}layers_{num_hidden_features_BD_sim}hidden_{seed}seed"
    BD_sim_training_dir = os.path.join(exp_dir, f"saved_models_{loc_id_BD_sim}/")
    BD_sim_samples_dir = os.path.join(BD_sim_training_dir, f"npy_samples/")
    
    config_string_BD_sim = "epochs{0}_lr{1}_wd{2}_bs{3}".format(hyperparameters_dict_BD_sim["n_epochs"], hyperparameters_dict_BD_sim["lr"], hyperparameters_dict_BD_sim["weight_decay"], hyperparameters_dict_BD_sim["batch_size"])
    checkpoint_path_BD_sim = os.path.join(BD_sim_training_dir, f"BDSIM_{config_string_BD_sim}")
    
    # Define a flow architecture
    transforms_BD_sim = make_masked_AR_flow(num_layers_BD_sim, n_features, num_hidden_features_BD_sim)
    base_dist_sim = StandardNormal(shape=[n_features])

    # Create and train
    #create_and_train_flow("BDSIM", BD_sim_training_dir, transforms_BD_sim, base_dist_sim, hyperparameters_dict_BD_sim, device, dataset_train_sim, dataset_val_sim, early_stop = False, seed = seed)

    #make_base_density_samples(hyperparameters_dict_BD_sim, "BDSIM", BD_sim_training_dir, BD_sim_samples_dir, device, bands_dict, n_features, npull_dataset_val_sim, binning_scheme, col_minmax)

    #evaluate_base_density(BD_sim_samples_dir, hyperparameters_dict_BD_sim, "BDSIM", BD_sim_training_dir, device, bands_dict, n_features, hyperparameters_dict_eval)


    """
    "
    "
    LEARN SIM -> DAT ON SB
    "
    "
    """

    # Training s2d
    # This will be another (of many) subdirectory in saved_models/

    num_layers_s2d = 2
    num_nodes_s2d = 16
    hyperparameters_dict_s2d = {"n_epochs": 40,
                              "batch_size": 256,
                              "lr": 0.0004,
                              "weight_decay": 0.0001}
    
    loc_id_s2d = f"PRQ_Coupling_{num_layers_s2d}layers_{num_nodes_s2d}nodes_{seed}seed"
    # training dir is inside the BD dir
    s2d_training_dir = os.path.join(BD_sim_training_dir, f"saved_models_{loc_id_s2d}/")
    s2d_samples_dir = os.path.join(s2d_training_dir, f"npy_samples/")
    
    # Define a flow architecture
    transforms_s2d = make_coupling_flow(num_layers_s2d, n_features, num_nodes_s2d)
    

    flow_BD = torch.load(f"{checkpoint_path_BD_sim}_best_model.pt")
    # send network to device
    flow_BD.to(device)
    # freeze the trained model
    for param in flow_BD.parameters():
        param.requires_grad = False
    flow_BD.eval()

    # Create and train
    #create_and_train_flow("TRANS", s2d_training_dir, transforms_s2d, flow_BD, hyperparameters_dict_s2d, device, dataset_train_dat, dataset_val_dat, early_stop = False, seed = seed)


    """
    "
    "
    EVALUATE SIM -> DAT ON SB
    "
    "
    """
    
    
    #make_s2d_samples(["sb1", "sb2"], hyperparameters_dict_BD_sim, hyperparameters_dict_s2d, BD_sim_training_dir, s2d_training_dir, s2d_training_dir, device, bands_dict, n_features, npull_dataset_val_sim, npull_dataset_val_dat, binning_scheme, col_minmax, direct = False)
    
    
    #evaluate_s2d(["sb1", "sb2"], s2d_samples_dir, s2d_training_dir, hyperparameters_dict_eval, device, bands_dict, n_features)
    
    """
    "
    "
    EVALUATE SIM -> DAT ON SR
    "
    "
    """
    
    
    
    classif_train_sim = ToyDataset(data_dir, "classif_train_sim.npy")
    classif_train_dat = ToyDataset(data_dir, "classif_train_dat.npy")
    
    make_s2d_samples(["sr"], hyperparameters_dict_BD_sim, hyperparameters_dict_s2d, BD_sim_training_dir, s2d_training_dir, s2d_training_dir, device, bands_dict, n_features, classif_train_sim, classif_train_dat, binning_scheme, col_minmax, direct = False, oversample = oversample)
    

    
    
    return(message)


if __name__=='__main__':
    main(message)
