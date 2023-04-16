#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import os
from numba import cuda 
import argparse

from helpers.composite_helpers import *

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-n", "--num_signal_to_inject", help = "num signal to inject")
parser.add_argument("-c", "--cuda_slot", help = "which gpu")
 
# Read arguments from command line
args = parser.parse_args()


"""
"""
"""
COMPUTING PARAMETERS
"""
"""
"""



os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda_slot
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

eval_oversampled = True
eval_nonsampled = False
oversampled_num = 1


index_start = 0
index_stop = 20


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

results_dir = f"/clusterfs/ml4hep/rrmastandrea/oversampling_results/nsig_inj{args.num_signal_to_inject}_seed{seed}/"
os.makedirs(results_dir, exist_ok=True)
scaled_data_dir = "/global/home/users/rrmastandrea/oversamp_studies/"


def analyze_band_transform_with_weights(dir_to_save, idd, 
                                        train_samp_1, train_samp_2, 
                                        weights_samp_1, weights_samp_2,
                                        test_samp_1, test_samp_2, 
                                        n_features, n_epochs, 
                                        batch_size, lr, patience, device, 
                                        update_epochs = 1, early_stop = True, visualize = True, seed = None):
    
    
    # save the best model
    val_loss_to_beat = 10000
    best_epoch = -1
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
                    
    dense_net = NeuralNet(input_shape = n_features)
    criterion = F.binary_cross_entropy #nn.BCELoss()
    optimizer = torch.optim.Adam(dense_net.parameters(), lr=lr)

    dense_net.to(device)
    
    if early_stop:
        early_stopping = EarlyStopping(patience=patience)
    
    # transformed SIM has label 0, DAT has label 1
    # make the input and output data
    nn_train_data = np.concatenate((train_samp_1, train_samp_2))
    nn_train_labs = np.concatenate((torch.zeros((train_samp_1.shape[0], 1)), torch.ones((train_samp_2.shape[0],1))))
    nn_train_weights =  np.concatenate((weights_samp_1, weights_samp_2))
    
    print(f"Sample 1 shape: {train_samp_1.shape}")
    print(f"Sample 2 shape: {train_samp_2.shape}")
          
    # get CLASS weights
    class_weights = class_weight.compute_class_weight('balanced', np.unique(nn_train_labs.reshape(-1)), nn_train_labs.reshape(-1))
    class_weights = dict(enumerate(class_weights))
    print(class_weights)
    
    # train-test split
    val_size = 0.2
    
    # train-val split
    X_train, X_val, w_train, w_val, y_train, y_val = train_test_split(nn_train_data, nn_train_weights, nn_train_labs, test_size=val_size)
    
    # if no test data provided, use the val data
    if (test_samp_1 is None) or (test_samp_2 is None):
        print("Using val data as test data...")
        X_test = X_val
        y_test = y_val
    else:
        nn_test_data = np.concatenate((test_samp_1, test_samp_2))
        nn_test_labs = np.concatenate((torch.zeros((test_samp_1.shape[0], 1)), torch.ones((test_samp_2.shape[0],1))))
        # shuffle the data
        nn_train_data, nn_train_labs = shuffle(nn_train_data, nn_train_labs)
        X_test, y_test = shuffle(nn_test_data, nn_test_labs)
    
    print("Train data, labels shape:", X_train.shape, y_train.shape)
    print("Val data, labels shape:", X_val.shape, y_val.shape)
    print("Test data, labels  shape:", X_test.shape, y_test.shape)
   
    # send to device
    X_train = np_to_torch(X_train, device)
    X_val = np_to_torch(X_val, device)
    X_test = np_to_torch(X_test, device)
    y_train = np_to_torch(y_train, device)
    y_val = np_to_torch(y_val, device)
    w_train = np_to_torch(w_train, device)
    w_val = np_to_torch(w_val, device)

    epochs, epochs_val = [], []
    losses, losses_val = [], []

    for epoch in tqdm(range(n_epochs)):
        # batching
        indices_list = torch.split( torch.randperm( X_train.shape[0] ), batch_size )
        # initialise lists to store batch stats
        losses_batch_per_e = []
        
        for i, indices in enumerate( indices_list ): # going through the batches
            # calculate the loss, backpropagate
            optimizer.zero_grad()
            
            batch_data = X_train[indices]
            batch_labels = y_train[indices]
            batch_weights_salad = w_train[indices]
            
            # get the weights
            batch_weights_class = (torch.ones(batch_labels.shape, device=device)
                        - batch_labels)*class_weights[0] \
                        + batch_labels*class_weights[1]
            
            batch_weights = batch_weights_salad*batch_weights_class


            loss = criterion(dense_net(batch_data), batch_labels, weight = batch_weights)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            losses_batch_per_e.append(loss.detach().cpu().numpy())

        epochs.append(epoch)
        losses.append(np.mean(losses_batch_per_e))
        
        # validation
        if epoch % update_epochs == 0:
            
            with torch.no_grad():
                # batching
                val_indices_list = torch.split( torch.randperm( X_val.shape[0] ), batch_size )
                # initialise lists to store batch stats
                val_losses_batch_per_e = []

                for i, indices in enumerate( val_indices_list ): # going through the batches
                    # calculate the loss, backpropagate
                    optimizer.zero_grad()
                    
                    batch_data = X_val[indices]
                    batch_labels = y_val[indices]
                    batch_weights_salad = w_val[indices]

                    # get the weights
                    batch_weights_class = (torch.ones(batch_labels.shape, device=device)
                                - batch_labels)*class_weights[0] \
                                + batch_labels*class_weights[1]
                    
                    batch_weights = batch_weights_salad*batch_weights_class

                    
                    val_loss = criterion(dense_net(batch_data), batch_labels, weight = batch_weights) 

                    val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())

            epochs_val.append(epoch)
            losses_val.append(np.mean(val_losses_batch_per_e))
            
            # see if the model has the best val loss
            if np.mean(val_losses_batch_per_e) < val_loss_to_beat:
                val_loss_to_beat = np.mean(val_losses_batch_per_e)
                # save the model
                model_path = f"{dir_to_save}/.{idd}_best_model.pt"
                torch.save(dense_net, model_path)
                best_epoch = epoch
                
            if early_stop:
                early_stopping(np.mean(val_losses_batch_per_e))
                
        if early_stopping.early_stop:
            break

    print("Done training!")
    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(epochs, losses)
        ax.plot(epochs_val, losses_val, label = "val")
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(idd)
        fname = f"{dir_to_save}/losses_{idd}"
        fig.savefig(fname)

    # evaluate
               
    
    # load in the model with the best val loss
    
    print(f"Loading in best model for {model_path}, val loss {val_loss_to_beat} from epoch {best_epoch}")
    dense_net_eval = torch.load(model_path)
    dense_net_eval.eval()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        outputs = dense_net_eval(X_test).detach().cpu().numpy()
        predicted = np.round(outputs)

        # calculate auc 
        auc = roc_auc_score(y_test, outputs)
        fpr, tpr, _ = roc_curve(y_test, outputs)

    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(fpr, tpr)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC: " + str(auc))
        fname = f"{dir_to_save}/roc_{idd}"
        fig.savefig(fname)
        
    np.save(f"{dir_to_save}/fpr_{idd}", fpr)
    np.save(f"{dir_to_save}/tpr_{idd}", tpr)
        
    if auc < 0.5:
        auc = 1.0 - auc
    
    return auc





"""
"""
"""
STS DATA
"""
"""
"""


STS_bkg_dataset = np.load(f"{scaled_data_dir}/STS_bkg.npy")
STS_sig_dataset = np.load(f"{scaled_data_dir}/STS_sig.npy")

dat_samples_train = np.load(f"{scaled_data_dir}/nsig_injected_{args.num_signal_to_inject}/data.npy")


"""
"""
"""
EVAL 
"""
"""
"""

    
# load in the data samples
feta_samples_osamp = np.load(f"{scaled_data_dir}/nsig_injected_{args.num_signal_to_inject}/feta_o{oversampled_num}.npy")
feta_samples_nonsamp = np.load(f"{scaled_data_dir}/nsig_injected_{args.num_signal_to_inject}/feta.npy")


blank_weights_osamp = np.ones((feta_samples_osamp.shape[0], 1))
blank_weights_nonsamp = np.ones((feta_samples_nonsamp.shape[0], 1))
blank_weights_data = np.ones((dat_samples_train.shape[0], 1))


for seed_NN in range(index_start, index_stop, 1):
        

    if eval_oversampled:
        print(f"Evaluating feta oversampled x{oversampled_num} (seed {seed_NN + 1} of {index_stop})...")
        
        roc = analyze_band_transform_with_weights(results_dir, f"feta_o{oversampled_num}_{seed_NN}", feta_samples_osamp[:,:-1], dat_samples_train[:,:-1], blank_weights_osamp, blank_weights_data, STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/feta_o{oversampled_num}_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(5*"*")
        print()

    if eval_nonsampled:
        print(f"Evaluating feta notoversampled (seed {seed_NN + 1} of {index_stop})...")
        
        roc = analyze_band_transform_with_weights(results_dir, f"feta_{seed_NN}", feta_samples_nonsamp[:,:-1], dat_samples_train[:,:-1], blank_weights_nonsamp, blank_weights_data, STS_bkg_dataset[:,:-1], STS_sig_dataset[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device, visualize = True, seed = seed_NN)
        results_file = f"{results_dir}/feta_{seed_NN}.txt"

        with open(results_file, "w") as results:
            results.write(f"Discrim. power for STS bkg from STS sig in band SR: {roc}\n")
            results.write(3*"\n")

        print()
        print(5*"*")
        print()







