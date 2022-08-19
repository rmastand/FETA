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

message = "final (??) training for the 2 step"

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

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device = cuda.get_current_device()
    device.reset()

    # set the number of threads that pytorch will use
    torch.set_num_threads(2)

    # set gpu device
    #device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device( "cpu")
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

    hyperparameters_dict_eval = {"n_epochs": 70,
                              "batch_size": 128,
                              "lr": 0.0005,
                              "num_bootstrap": 1,
                                "patience": 10}
    
    use_old_CC = True


    # directories

    curtains_dir = "/global/home/users/rrmastandrea/CURTAINS_SALAD/"

    n_features = 5
    dataset_config_string = f"LHCO_2step/"

    exp_dir = os.path.join(curtains_dir, dataset_config_string)
    data_dir = os.path.join(exp_dir, "data")

    print("Making results directory at", exp_dir, "...")
    os.makedirs(exp_dir, exist_ok=True)

   
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


    # train-val split
    val_size = 0.2
    sim_train_dataset, sim_val_dataset = make_train_val_split(training_dataset_sim, val_size)
    dat_train_dataset, dat_val_dataset = make_train_val_split(training_dataset_dat, val_size)

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
    hyperparameters_dict_BD_sim = {"n_epochs": 50,
                              "batch_size": 128,
                              "lr": 0.0001,
                              "weight_decay": 0.0001}

    loc_id_BD_sim = f"BD_sim_Masked_PRQ_AR_{num_layers_BD_sim}layers_{num_hidden_features_BD_sim}hidden_LRCos"
    BD_sim_training_dir = os.path.join(exp_dir, f"saved_models_{loc_id_BD_sim}/")
    BD_sim_samples_dir = os.path.join(BD_sim_training_dir, f"npy_samples/")
    
    config_string_BD_sim = "epochs{0}_lr{1}_wd{2}_bs{3}".format(hyperparameters_dict_BD_sim["n_epochs"], hyperparameters_dict_BD_sim["lr"], hyperparameters_dict_BD_sim["weight_decay"], hyperparameters_dict_BD_sim["batch_size"])
    checkpoint_path_BD_sim = os.path.join(BD_sim_training_dir, f"BDSIM_{config_string_BD_sim}")
    
    # Define a flow architecture
    transforms_BD_sim = make_masked_AR_flow(num_layers_BD_sim, n_features, num_hidden_features_BD_sim)
    base_dist_sim = StandardNormal(shape=[n_features])

    # Create and train
    create_and_train_flow("BDSIM", BD_sim_training_dir, transforms_BD_sim, base_dist_sim, hyperparameters_dict_BD_sim, device, sim_train_dataset, sim_val_dataset, early_stop = True)

    make_base_density_samples(hyperparameters_dict_BD_sim, "BDSIM", BD_sim_training_dir, BD_sim_samples_dir, device, bands_dict, n_features, dataset_sim, binning_scheme)

    evaluate_base_density(BD_sim_samples_dir, hyperparameters_dict_BD_sim, "BDSIM", BD_sim_training_dir, device, bands_dict, n_features, dataset_sim, binning_scheme, hyperparameters_dict_eval, use_old_CC = use_old_CC)


    """
    "
    "
    LEARN SIM -> DAT
    "
    "
    """

    # Training s2d
    # This will be another (of many) subdirectory in saved_models/

    num_layers_s2d = 2
    num_nodes_s2d = 32
    hyperparameters_dict_s2d = {"n_epochs": 50,
                              "batch_size": 128,
                              "lr": 0.0001,
                              "weight_decay": 0.0001}
    
    loc_id_s2d = f"PRQ_Coupling_{num_layers_s2d}layers_{num_nodes_s2d}nodes_LRCos"
    # traiign dir is inside the BD dir
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
    #create_and_train_flow("TRANS", s2d_training_dir, transforms_s2d, flow_BD, hyperparameters_dict_s2d, device, dat_train_dataset, dat_val_dataset, early_stop = True)


    """
    "
    "
    EVALUATE SIM -> DAT
    "
    "
    """
    
    
    #make_s2d_samples(hyperparameters_dict_BD_sim, hyperparameters_dict_s2d, BD_sim_training_dir, s2d_training_dir, s2d_training_dir, device, bands_dict, n_features, dataset_sim, dataset_dat, binning_scheme, direct = False)
    
    
    #evaluate_s2d(s2d_samples_dir, hyperparameters_dict_eval, device, bands_dict, n_features, dataset_sim, dataset_dat, binning_scheme, use_old_CC = use_old_CC)
    
    return(message)


if __name__=='__main__':
    main(message)
