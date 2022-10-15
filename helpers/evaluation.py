import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import torch
import os

import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler


from tqdm import tqdm

from helpers.training import *
from helpers.datasets import *
import helpers.CURTAINS_classifier as oldCC
from helpers.utils import EarlyStopping



def transform_sim_to_dat_2step(flow, data_input, device):
    
    """
    uses the learned flow_s2d to transform 
    make sure the data is preprocessed!!!
    """
    
    feat = torch.from_numpy(data_input[:,:-1]).float().to(device)
    cont = torch.from_numpy(np.reshape(data_input[:,-1], (-1, 1))).float().to(device)

    with torch.no_grad():
        outputs, logabsdet = flow._transform.inverse(feat, context=flow._embedding_net(cont))
        
    return outputs.detach().cpu().numpy()


def transform_sim_to_dat_direct(flow_SIM, flow_DAT, sim_input, device):
    
    """
    uses flow_SIM to transform SIM to normal
    then uses flow_DAT._inverse to transform from normal to DAT
    """
    
    feat = sim_input[:,:-1].float().to(device)
    cont = torch.reshape(sim_input[:,-1], (-1, 1)).float().to(device)

    with torch.no_grad():
        outputs_normal_target, logabsdet = flow_SIM._transform(feat, context=flow_SIM._embedding_net(cont))
        outputs_dat_target, logabsdet = flow_DAT._transform.inverse(outputs_normal_target, context=flow_DAT._embedding_net(cont))
        
    return outputs_dat_target.detach().cpu().numpy()


def make_BD_samples_dict(bands_to_sample, bands_dict, n_features, dataset_sim, col_minmax, bd_flow, device, oversample = 1):
    """
    returns 2 arrays
    sim_samples: SB1 + SB2 samples from SIM data (what was used to train the base density)
    BD_samples: samples from the base density, using the masses from sim_samples as context
    
    """
    
    print(f"Oversampling: {oversample}")
    
    bands_edges = [bands_dict[band] for band in bands_to_sample]
    
    # create the sim_samples
    sim_samples = dataset_sim.pull_from_mass_range(bands_edges)
    sim_samples = minmaxscale(sim_samples.data, col_minmax, lower = -3, upper = 3, forward = True)

    # extract the (preprocessd) mass
    context_masses = torch.from_numpy(np.reshape(sim_samples[:,-1], (-1, 1))).float().to(device)
    
    # sample from the base density
    BD_samples = bd_flow.sample(1, context=context_masses).detach().cpu().numpy()
    BD_samples = BD_samples.reshape(BD_samples.shape[0], n_features)
    BD_samples = np.hstack((BD_samples, np.reshape(sim_samples[:,-1], (-1, 1))))
    
    if oversample > 1:
        for i in range(oversample - 1):
            new_BD_samples = bd_flow.sample(1, context=context_masses).detach().cpu().numpy()
            new_BD_samples = new_BD_samples.reshape(new_BD_samples.shape[0], n_features)
            new_BD_samples = np.hstack((new_BD_samples, np.reshape(sim_samples[:,-1], (-1, 1))))
            
            BD_samples = np.vstack((BD_samples, new_BD_samples))
       
    
    return sim_samples, BD_samples



def make_trans_samples_dict_2step(bands_to_transform, bands_dict, dataset_sim, dataset_dat, col_minmax, transform_flow, device):
    
    """
    Creates 3 dicts sim_samples, transformed_sim_samples, dat_samples
    
    bands_to_transform is a list of bands
    """
    
    sim_samples = {band:0 for band in bands_to_transform}
    transformed_sim_samples = {band:0 for band in bands_to_transform}
    dat_samples = {band:0 for band in bands_to_transform}

    for band in bands_to_transform:
        
        print("Making samples for band", band, "...")

        # get the sim sample
        eval_dataset_sim = dataset_sim.pull_from_mass_range([bands_dict[band]])
        eval_dataset_sim = minmaxscale(eval_dataset_sim.data, col_minmax, lower = -3, upper = 3, forward = True)

        # get the dat sample
        eval_dataset_dat = dataset_dat.pull_from_mass_range([bands_dict[band]])
        eval_dataset_dat = minmaxscale(eval_dataset_dat.data, col_minmax, lower = -3, upper = 3, forward = True)

        # transform the sim sample to dat 
        transformed_features_sim = transform_sim_to_dat_2step(transform_flow, eval_dataset_sim, device)
        
        # save to dicts       
        sim_samples[band] = eval_dataset_sim
        transformed_sim_samples[band] = np.hstack((transformed_features_sim, np.reshape(eval_dataset_sim[:,-1], (-1, 1))))
        dat_samples[band] = eval_dataset_dat
        
    return sim_samples, transformed_sim_samples, dat_samples



def make_trans_samples_dict_direct(bands_to_transform, bands_dict, dataset_sim, dataset_dat, flow_SIM, flow_DAT, device):
    
    """
    Creates 3 dicts sim_samples, transformed_sim_samples, dat_samples
    
    bands_to_transform is a list of bands
    """
    
    sim_samples = {band:0 for band in bands_to_transform}
    transformed_sim_samples = {band:0 for band in bands_to_transform}
    dat_samples = {band:0 for band in bands_to_transform}

    for band in bands_to_transform:
        
        print("Making samples for band", band, "...")

        # get the sim sample
        eval_dataset_sim = dataset_sim.pull_from_mass_range([bands_dict[band]])
        eval_dataset_sim.minmaxscale()

        # get the dat sample
        eval_dataset_dat = dataset_dat.pull_from_mass_range([bands_dict[band]])
        eval_dataset_dat.minmaxscale()

        # transform the sim sample to dat 
        transformed_features_sim = transform_sim_to_dat_direct(flow_SIM, flow_DAT, eval_dataset_sim, device)

        # save to dicts
        sim_samples[band] = eval_dataset_sim[:,:-1].detach().cpu().numpy()
        transformed_sim_samples[band] = transformed_features_sim
        dat_samples[band] = eval_dataset_dat[:,:-1].detach().cpu().numpy()
        
    return sim_samples, transformed_sim_samples, dat_samples


"""
NEURAL NET
"""

class NeuralNet(nn.Module):
    def __init__(self, input_shape):
        super(NeuralNet, self).__init__()

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        self.dropout1 = nn.Dropout2d(0.1)
        self.dropout2 = nn.Dropout2d(0.1)

        # First fully connected layer
        self.fc1 = nn.Linear(input_shape, 64) # first size is output of flatten
        # Second fully connected layer that outputs our labels
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        
    # x represents our data
    def forward(self, x):

        x = torch.flatten(x, 1)
        # Pass data through fc1
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
    
        # Apply softmax to x
        #output = F.log_softmax(x, dim=1)
        output = torch.sigmoid(x) # for BCE 
        
        return output
    
    
def analyze_band_transform(dir_to_save, idd, train_samp_1, train_samp_2, test_samp_1, test_samp_2, n_features, n_epochs, batch_size, lr, patience, device, update_epochs = 1, early_stop = True, visualize = True, seed = None):
    
    if seed is not None:
        #print(f"Using seed {seed}...")
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
    


    # get weights in case we're oversampling
    class_weights = class_weight.compute_class_weight('balanced', np.unique(nn_train_labs.reshape(-1)), nn_train_labs.reshape(-1))
    class_weights = dict(enumerate(class_weights))
    print(class_weights)
        
    
    # train-test split
    val_size = 0.2
    
    # train-val split
    X_train, X_val, y_train, y_val = train_test_split(nn_train_data, nn_train_labs, test_size=val_size)
    
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
   
    # apply preprocessing
    #scaler = MinMaxScaler()
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_val = scaler.transform(X_val)
    #X_test = scaler.transform(X_test)
    
    for i in range(X_train.shape[1]):
        print(f"Feature {i} min, max for train: ({np.min(X_train[:,i])},{np.max(X_train[:,i])}), val: ({np.min(X_val[:,i])},{np.max(X_val[:,i])}), test: ({np.min(X_test[:,i])},{np.max(X_test[:,i])})")  
        


    # send to device
    X_train = np_to_torch(X_train, device)
    X_val = np_to_torch(X_val, device)
    X_test = np_to_torch(X_test, device)
    y_train = np_to_torch(y_train, device)
    y_val = np_to_torch(y_val, device)
    
    

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

                    # get the weights
                    batch_weights = (torch.ones(batch_labels.shape, device=device)
                                - batch_labels)*class_weights[0] \
                                + batch_labels*class_weights[1]

                    
                    val_loss = criterion(dense_net(batch_data), batch_labels, weight = batch_weights) 

                    val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())

            epochs_val.append(epoch)
            losses_val.append(np.mean(val_losses_batch_per_e))
            
            if early_stop:
                early_stopping(np.mean(val_losses_batch_per_e))
                
        if early_stopping.early_stop:
            break

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

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        outputs = dense_net(X_test).detach().cpu().numpy()
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


def analyze_band_transforms_CURTAILS_old(band, idd, transformed_sim_samples, dat_samples, classif_dir, n_epochs, batch_size, lr, visualize = False, test_samp1 = False, test_samp2 = False):
    
    """adapted the CURTAINS old code so it's now band-wise"""
    
    classifier_band_dir = os.path.join(classif_dir, f"{band}_{idd}")
    os.makedirs(classifier_band_dir, exist_ok=True)
    print("On band", band, "testing", idd, "...")
    
    scores_df = oldCC.assign_scores(transformed_sim_samples, dat_samples, classifier_band_dir, n_epochs=n_epochs, batch_size=batch_size, lr=lr, visualize = visualize, test_samp1 = test_samp1, test_samp2 = test_samp2)
    
    auc, fpr, tpr = oldCC.get_classification(scores_df)
    
    # save out the fpr, tpr
    np.save(os.path.join(classifier_band_dir, "fpr.npy"), fpr)
    np.save(os.path.join(classifier_band_dir, "tpr.npy"), tpr)
    
    if auc < 0.5:
        return 1.0 - auc
    
    return auc

    


def get_bootstrapped_indices(n):
    
    bootstrapped_indices = np.random.choice(n, size = n, replace = True)
    return(bootstrapped_indices)


def get_bootstrapped_AUC(auc_list, lower_percentile = 5, upper_percentile = 95, num_decimals = 6):
    
    """
    Returns mean, 5th percentile, 95th percentile rounded to 6 decimals
    """
    
    mean_AUC = np.round(np.mean(auc_list),num_decimals)
    lower_AUC = np.round(np.percentile(auc_list,lower_percentile),num_decimals)
    upper_AUC = np.round(np.percentile(auc_list,upper_percentile),num_decimals)
    
    return mean_AUC, lower_AUC, upper_AUC
    
    
