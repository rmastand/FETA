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

from tqdm import tqdm

from helpers.training import *
import helpers.CURTAINS_classifier as oldCC
from helpers.utils import EarlyStopping



def transform_sim_to_dat_2step(flow, data_input, device):
    
    """
    uses the learned flow_s2d to transform 
    make sure the data is preprocessed!!!
    """
    
    feat = data_input[:,:-1].float().to(device)
    cont = torch.reshape(data_input[:,-1], (-1, 1)).float().to(device)

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


def make_BD_samples_dict(bands_dict, n_features, dataset_sim, bd_flow, device):
    """
    returns 2 arrays
    sim_samples: SB1 + SB2 samples from SIM data (what was used to train the base density)
    BD_samples: samples from the base density, using the masses from sim_samples as context
    
    """
    # create the sim_samples
    sim_samples = dataset_sim.pull_from_mass_range([bands_dict["sb1"], bands_dict["sb2"]])
    sim_samples.minmaxscale()

    # extract the (preprocessd) mass
    context_masses = torch.reshape(sim_samples[:,-1], (-1, 1)).float().to(device)
    
    # sample from the base density
    BD_samples = bd_flow.sample(1, context=context_masses).detach().cpu().numpy()
    BD_samples = BD_samples.reshape(BD_samples.shape[0], n_features)
    
    BD_samples = np.hstack((BD_samples, np.reshape(sim_samples[:,-1], (-1, 1))))

    
    return sim_samples.data, BD_samples



def make_trans_samples_dict_2step(bands_to_transform, bands_dict, dataset_sim, dataset_dat, transform_flow, device):
    
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
        transformed_features_sim = transform_sim_to_dat_2step(transform_flow, eval_dataset_sim, device)

        # save to dicts
        sim_samples[band] = eval_dataset_sim[:,:-1].detach().cpu().numpy()
        transformed_sim_samples[band] = transformed_features_sim
        dat_samples[band] = eval_dataset_dat[:,:-1].detach().cpu().numpy()
        
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
    
    
def analyze_band_transform(band, transformed_sim_samples, dat_samples, n_features, n_epochs, batch_size, lr, patience, device, update_epochs = 1, early_stop = True, visualize = False):
    
    #torch.manual_seed(8)
    #np.random.seed(8)

    dense_net = NeuralNet(input_shape = n_features)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(dense_net.parameters(), lr=lr)

    dense_net.to(device)
    
    if early_stop:
        early_stopping = EarlyStopping(patience=patience)
    
    # transformed SIM has label 0, DAT has label 1
    # make the input and output data
    nn_train_data = np.concatenate((transformed_sim_samples, dat_samples))
    nn_train_labs = np.concatenate((torch.zeros((transformed_sim_samples.shape[0], 1)), torch.ones((dat_samples.shape[0],1))))

    # shuffle the data
    nn_train_data, nn_train_labs = shuffle(nn_train_data, nn_train_labs)
    
    # train-test split
    test_size = 0.1
    val_size = 0.3
    
    # train-val split
    X_train, X_test, y_train, y_test = train_test_split(nn_train_data, nn_train_labs, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)
    
    #print("Train data, labels shape:", X_train.shape, y_train.shape)
    #print("Val data, labels shape:", X_val.shape, y_val.shape)
    #print("Test data, labels  shape:", X_test.shape, y_test.shape)
    

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
            
            try:
                loss = criterion(dense_net(X_train[indices]), y_train[indices])
            except(RuntimeError):
                for pp in range(len(X_train[indices])):
                    print(X_train[indices][pp], dense_net(X_train[indices])[pp]  )             
                
                exit()
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
                    val_loss = criterion(dense_net(X_val[indices]), y_val[indices])

                    val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())

            epochs_val.append(epoch)
            losses_val.append(np.mean(val_losses_batch_per_e))
            
            if early_stop:
                early_stopping(np.mean(val_losses_batch_per_e))
                
        if early_stopping.early_stop:
            break

    if visualize:
        plt.figure()
        plt.plot(epochs, losses)
        plt.plot(epochs_val, losses_val, label = "val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(band)
        plt.show()

    # evaluate

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        outputs = dense_net(X_test).detach().cpu().numpy()
        predicted = np.round(outputs)

        # calculate auc 
        auc = roc_auc_score(y_test, outputs)
        fpr, tpr, _ = roc_curve(y_test, outputs)

    if visualize:
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.xlabel("TPR")
        plt.title("ROC: " + str(auc))
        plt.show()
        
    if auc < 0.5:
        auc = 1.0 - auc
    
    return auc


def analyze_band_transforms_CURTAILS_old(band, idd, transformed_sim_samples, dat_samples, classif_dir, n_epochs, batch_size, lr, visualize = False):
    
    """adapted the CURTAINS old code so it's now band-wise"""
    
    classifier_band_dir = os.path.join(classif_dir, f"{band}_{idd}")
    os.makedirs(classifier_band_dir, exist_ok=True)
    print("On band", band, "testing", idd, "...")
    
    scores_df = oldCC.assign_scores(transformed_sim_samples, dat_samples, classifier_band_dir, n_epochs=n_epochs, batch_size=batch_size, lr=lr, visualize = visualize)
    
    auc = oldCC.get_classification(scores_df)
    
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
    
    
