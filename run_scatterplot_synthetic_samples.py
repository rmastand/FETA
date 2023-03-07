#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import os
from numba import cuda 

from helpers.composite_helpers import *

from sklearn.model_selection import KFold


feta_dir = "/global/home/users/rrmastandrea/FETA"

scatterplot_dir = os.path.join(feta_dir, "scatterplot_all_synth_samples")
os.makedirs(scatterplot_dir, exist_ok=True)

scaled_data_dir = "/global/home/users/rrmastandrea/scaled_data/"

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

num_signal_to_inject = 1500
index_start = 0
index_stop = 5


eval_feta = False
eval_cathode = False
eval_curtains = False
eval_full_sup = False
eval_combined = True

# parameters for combined samples
num_synth_samples = 600000
# coefficients for mixing
# recommended to have them sum to 1 but there's no check on that
c_feta = .33
c_cathode = 0.33
c_curtains = 0.33


epochs_NN =  100
batch_size_NN = 256
lr_NN = 0.001
patience_NN = 5


def analyze_transform_for_scatter_kfold(idd, train_samp_1, train_samp_2, test_samp_1, test_samp_2, n_features, n_epochs, batch_size, lr, patience, device, early_stop = True, visualize = True, seed = None, k_folds = 5):
    
    if seed is not None:
        #print(f"Using seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    
    # transformed SIM has label 0, DAT has label 1
    # make the input and output data
    X_train = np.concatenate((train_samp_1, train_samp_2))
    y_train = np.concatenate((torch.zeros((train_samp_1.shape[0], 1)), torch.ones((train_samp_2.shape[0],1))))    
    
    # get weights in case we're oversampling
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train.reshape(-1)), y_train.reshape(-1))
    class_weights = dict(enumerate(class_weights))   
    
    X_test = np.concatenate((test_samp_1, test_samp_2))
    y_test = np.concatenate((torch.zeros((test_samp_1.shape[0], 1)), torch.ones((test_samp_2.shape[0],1))))
    
    print("Train data, labels shape:", X_train.shape, y_train.shape)
    print("Test data, labels  shape:", X_test.shape, y_test.shape)

    # send to device
    X_train = np_to_torch(X_train, device)
    X_test = np_to_torch(X_test, device)
    y_train = np_to_torch(y_train, device)
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_best_val_losses = []
    
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train)):
        
    
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        X_train_fold = X_train[train_ids]
        y_train_fold = y_train[train_ids]
        
        X_val_fold = X_train[val_ids]
        y_val_fold = y_train[val_ids]
        
        train_set = torch.utils.data.TensorDataset(X_train_fold, y_train_fold)
        val_set = torch.utils.data.TensorDataset(X_val_fold, y_val_fold)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True)
        
        #print("fold test data, labels shape:", X_train_fold.shape, y_train_fold.shape)
        #print("fold val data, labels  shape:", X_val_fold.shape, y_val_fold.shape)
        
        # initialze the network
        dense_net = NeuralNet(input_shape = n_features)
        criterion = F.binary_cross_entropy 
        optimizer = torch.optim.Adam(dense_net.parameters(), lr=lr)
        dense_net.to(device)
        
        if early_stop:
            early_stopping = EarlyStopping(patience=patience)
        
         # save the best model
        val_loss_to_beat = 10000
        best_epoch = -1

        epochs, losses, losses_val = [], [], []

        for epoch in tqdm(range(n_epochs)):
            losses_batch_per_e = []
            # batching    
            for batch_index, (batch_data, batch_labels) in enumerate(train_loader):

                # calculate the loss, backpropagate
                optimizer.zero_grad()

                # get the weights
                batch_weights = (torch.ones(batch_labels.shape, device=device)
                            - batch_labels)*class_weights[0] \
                            + batch_labels*class_weights[1]

                loss = criterion(dense_net(batch_data), batch_labels, weight = batch_weights)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses_batch_per_e.append(loss.detach().cpu().numpy())

            epochs.append(epoch)
            losses.append(np.mean(losses_batch_per_e))

            # validation
            with torch.no_grad():
                val_losses_batch_per_e = []
                
                for batch_index, (batch_data, batch_labels) in enumerate(val_loader):
                    # calculate the loss, backpropagate
                    optimizer.zero_grad()

                    # get the weights
                    batch_weights = (torch.ones(batch_labels.shape, device=device)
                                - batch_labels)*class_weights[0] \
                                + batch_labels*class_weights[1]

                    val_loss = criterion(dense_net(batch_data), batch_labels, weight = batch_weights) 
                    val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())

                losses_val.append(np.mean(val_losses_batch_per_e))

                # see if the model has the best val loss
                if np.mean(val_losses_batch_per_e) < val_loss_to_beat:
                    val_loss_to_beat = np.mean(val_losses_batch_per_e)
                    # save the model
                    model_path = f"{scatterplot_dir}/.{idd}_fold{fold}.pt"
                    torch.save(dense_net, model_path)
                    best_epoch = epoch

                if early_stop:
                    early_stopping(np.mean(val_losses_batch_per_e))

            if early_stopping.early_stop:
                break

        print(f"Done training fold {fold}. Best val loss {val_loss_to_beat} at epoch {best_epoch}")
        if visualize:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            ax.plot(epochs, losses)
            ax.plot(epochs, losses_val, label = "val")
            ax.legend()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"{idd}_fold{fold}")
            fig.show()

        # evaluate
        fold_best_val_losses.append(val_loss_to_beat)
        print(fold_best_val_losses)
            
    
    # load in the model / fold with the best val loss 
    best_model_index = np.argmin(fold_best_val_losses)
    best_model_path = f"{scatterplot_dir}/.{idd}_fold{best_model_index}.pt"
    print(f"Loading in best model for {best_model_path}, val loss {np.min(fold_best_val_losses)} from fold {best_model_index}")
    
    dense_net_eval = torch.load(best_model_path)
    dense_net_eval.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        outputs = dense_net_eval(X_test).detach().cpu().numpy()

    return outputs




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

            feta_results = analyze_transform_for_scatter_kfold(f"feta_seedNN{seed_NN}_nsig{num_signal_to_inject}", feta_samples[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN, early_stop = True)
            np.save(f"{scatterplot_dir}/feta_results_seedNN{seed_NN}_nsig{num_signal_to_inject}", feta_results)


    if eval_cathode:

            print(f"Evaluating cathode (seed {seed_NN} of {index_stop})...")

            cathode_results = analyze_transform_for_scatter_kfold(f"cathode_seedNN{seed_NN}_nsig{num_signal_to_inject}", cathode_samples[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN, early_stop = True)
            np.save(f"{scatterplot_dir}/cathode_results_seedNN{seed_NN}_nsig{num_signal_to_inject}", cathode_results)


    if eval_curtains:

            print(f"Evaluating curtains (seed {seed_NN} of {index_stop})...")

            curtains_results = analyze_transform_for_scatter_kfold(f"curtains_seedNN{seed_NN}_nsig{num_signal_to_inject}", curtains_samples[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN, early_stop = True)
            np.save(f"{scatterplot_dir}/curtains_results_seedNN{seed_NN}_nsig{num_signal_to_inject}", curtains_results)  
            
            
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
                        
            combined_results = analyze_transform_for_scatter_kfold(f"combined_seedNN{seed_NN}_nsig{num_signal_to_inject}", synth_samples[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN, early_stop = True)
            np.save(f"{scatterplot_dir}/combined_results_seedNN{seed_NN}_nsig{num_signal_to_inject}", combined_results)  




"""
"
"
SUPERVISED CLASSIFIER
"
"""

if eval_full_sup:
    
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

        full_sup_results = analyze_transform_for_scatter_kfold(f"full_sup_seedNN{seed_NN}",true_sup_bkg[:,:-1], true_sup_sig[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN, early_stop = True)
        np.save(f"{scatterplot_dir}/full_sup_results_seedNN{seed_NN}", full_sup_results)

print("Done!")



