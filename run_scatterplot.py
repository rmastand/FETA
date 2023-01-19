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
# load in the reverse rescales
path_to_minmax = f"{feta_dir}/LHCO_STS/data/col_minmax.npy"
col_minmax = np.load(path_to_minmax)


scatterplot_dir = os.path.join(feta_dir, "scatterplot_results")
os.makedirs(scatterplot_dir, exist_ok=True)

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
oversampnum = 6


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
        
        print("fold test data, labels shape:", X_train_fold.shape, y_train_fold.shape)
        print("fold val data, labels  shape:", X_val_fold.shape, y_val_fold.shape)
        
        # initialze the network
        dense_net = NeuralNet(input_shape = n_features)
        criterion = F.binary_cross_entropy 
        optimizer = torch.optim.Adam(dense_net.parameters(), lr=lr)
        dense_net.to(device)
        
        if early_stop:
            early_stopping = EarlyStopping(patience=patience)
            print(early_stopping.patience, early_stopping.counter, early_stopping.best_loss, early_stopping.early_stop)
   
        
         # save the best model
        val_loss_to_beat = 10000
        best_epoch = -1
        print(val_loss_to_beat)

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

STS_dir = f"{feta_dir}/LHCO_STS/data/"

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

sim2real_exp_dir = f"{feta_dir}/LHCO_{num_signal_to_inject}sig_f/"

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



seed_NN = 4



print("Evaluating sim2real oversampled...")

oversampled_sim_samples_train = np.load(os.path.join(oversamples_dir, f"transBD.npy"))
oversampled_sim_samples_train = minmaxscale(oversampled_sim_samples_train, col_minmax, lower = -3, upper = 3, forward = False)
oversampled_sim_samples_train = minmaxscale(oversampled_sim_samples_train, col_minmax, lower = 0, upper = 1, forward = True)

feta_results = analyze_transform_for_scatter_kfold(f"sim2real_oversamp{oversampnum}_seedNN{seed_NN}_nsig{num_signal_to_inject}", oversampled_sim_samples_train[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN, early_stop = True)
np.save(f"{scatterplot_dir}/feta_results_nsig{num_signal_to_inject}", feta_results)

print(3*"\n")



print("Evaluating cathode...")

cathode_trans_samps = np.load(os.path.join(cathode_exp_dir, f"SR_samples.npy"))
cathode_results = analyze_transform_for_scatter_kfold(f"cathode_seedNN{seed_NN}_nsig{num_signal_to_inject}", cathode_trans_samps[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN, early_stop = True)
np.save(f"{scatterplot_dir}/cathode_results_nsig{num_signal_to_inject}", cathode_results)
print(3*"\n")


print("Evaluating curtains...")
#curtains_trans_samps = np.load(os.path.join(curtains_exp_dir, f"samples_sb1_2_to_sr.npy"))
#curtains_trans_samps = minmaxscale(curtains_trans_samps, col_minmax, lower = 0, upper = 1, forward = True)
samps = np.load(os.path.join(curtains_exp_dir, f"samples_sb1_2_to_sr.npz"))
d = dict(zip(("data1{}".format(k) for k in samps), (samps[k] for k in samps)))
curtains_trans_samps = d["data1arr_1"]
curtains_trans_samps = minmaxscale(curtains_trans_samps, col_minmax, lower = 0, upper = 1, forward = True)

curtains_results = analyze_transform_for_scatter_kfold(f"curtains_seedNN{seed_NN}_nsig{num_signal_to_inject}", curtains_trans_samps[:,:-1], dat_samples_train[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN, early_stop = True)
np.save(f"{scatterplot_dir}/curtains_results_nsig{num_signal_to_inject}", curtains_results)    
print(3*"\n")



"""
"
"
SUPERVISED CLASSIFIER
"
"""


print("Evaluating fully supervised case...")

true_samples_dir = f"{feta_dir}/LHCO_STS/data/"

true_sup_bkg = np.load(os.path.join(true_samples_dir, f"true_sup_bkg.npy"))
true_sup_sig = np.load(os.path.join(true_samples_dir, f"true_sup_sig.npy"))
true_sup_bkg = minmaxscale(true_sup_bkg, col_minmax, lower = 0, upper = 1, forward = True)
true_sup_sig = minmaxscale(true_sup_sig, col_minmax, lower = 0, upper = 1, forward = True)

full_sup_results = analyze_transform_for_scatter_kfold(f"full_sup_seedNN{seed_NN}",true_sup_bkg[:,:-1], true_sup_sig[:,:-1], STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN, early_stop = True)
np.save(f"{scatterplot_dir}/full_sup_results_nsig{num_signal_to_inject}", full_sup_results)
print(3*"\n")

          
print("Done!")



