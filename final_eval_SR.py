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

    """
    """
    """
    RUN PARAMETERS
    """
    """
    """
    
    seed = 1
    n_features = 5
    num_signal_to_inject = 1000

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
   
    results_dir = f"/global/home/users/rrmastandrea/NF_results/nsig_ing{num_signal_to_inject}_seed{seed}/"
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
    STS_bkg_dataset.minmaxscale()
    STS_sig_dataset = STS_sig.pull_from_mass_range([bands_dict["sr"]])
    STS_sig_dataset.minmaxscale()
    
    """
    """
    """
    EVAL SR SIM2REAL
    """
    """
    """
   
    sim2real_exp_dir = f"/global/home/users/rrmastandrea/CURTAINS_SALAD/LHCO_{num_signal_to_inject}sig_f/"
    
    num_layers_BD_sim = 8
    num_hidden_features_BD_sim = 64
    num_layers_s2d = 2
    num_nodes_s2d = 16
    
    loc_id_BD_sim = f"BD_sim_Masked_PRQ_AR_{num_layers_BD_sim}layers_{num_hidden_features_BD_sim}hidden_{seed}seed"
    loc_id_s2d = f"PRQ_Coupling_{num_layers_s2d}layers_{num_nodes_s2d}nodes_{seed}seed"
    BD_sim_training_dir = os.path.join(sim2real_exp_dir, f"saved_models_{loc_id_BD_sim}/")
    s2d_training_dir = os.path.join(BD_sim_training_dir, f"saved_models_{loc_id_s2d}/")
    s2d_samples_dir = os.path.join(s2d_training_dir, f"npy_samples/")
    
    # Load in npy samples for training the classifier
    transformed_sim_samples_train = {}
    dat_samples_train = {}   
    
    transformed_sim_samples_train = np.load(os.path.join(s2d_samples_dir, f"sr_transSIM.npy"))
    dat_samples_train = np.load(os.path.join(s2d_samples_dir, f"sr_DAT.npy")) 
    
    eval_sim2real = True
    
    if eval_sim2real:
        roc = analyze_band_transform(results_dir, "sim2real", transformed_sim_samples_train[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True)
        results_file = f"{results_dir}/sim2real.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for trans. SIM from DAT in band SR: {roc}\n")
            results.write(3*"\n")
        
    """
    """
    """
    EVAL SR CATHODE
    """
    """
    """

    cathode_exp_dir = f"/global/home/users/rrmastandrea/CATHODE/CATHODE_models/nsig_inj{num_signal_to_inject}/seed{seed}/"
    cathode_trans_samps = np.load(os.path.join(cathode_exp_dir, f"SR_samples.npy"))
    
    
    eval_cathode = False
    if eval_cathode:
        
        roc = analyze_band_transform(results_dir, "cathode", cathode_trans_samps[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True)
        results_file = f"{results_dir}/cathode.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for trans. SIM from DAT in band SR: {roc}\n")
            results.write(3*"\n")
        


if __name__=='__main__':
    main(message)

 
 