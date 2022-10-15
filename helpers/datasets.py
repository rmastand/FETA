import numpy as np
from torch.utils.data import Dataset

import pandas as pd
import os

from scipy.stats import skewnorm

import torch

"""
DATASETS
"""

def make_gaussian_dataset(n_points, n_features, mu, sig, context_endpoints):
    
    # make the array of contexts
    context_init = np.random.uniform(size=(n_points, 1)) 
    context_range = context_endpoints[1] - context_endpoints[0]
    context = context_init * context_range + context_endpoints[0]

    # make the features
    features = np.random.normal(mu, sig, size=(n_points, n_features))
    
    return np.hstack((features, context))

def make_triangle_dataset(n_points, n_features, l, m, r, context_endpoints):
    
    # make the array of contexts
    context_init = np.random.uniform(size=(n_points, 1)) 
    context_range = context_endpoints[1] - context_endpoints[0]
    context = context_init * context_range + context_endpoints[0]

    # make the features
    features = np.random.triangular(l, m, r, size=(n_points, n_features))

    return np.hstack((features, context))


def make_dataset_with_context(n_points, n_features, dataset_shape, context_endpoints, bias):
    
    # features now depend on the context
    
    # make the array of contexts
    context_init = np.random.uniform(size=(n_points, 1)) 
    context_range = context_endpoints[1] - context_endpoints[0]
    context_mean = 0.5*(context_endpoints[1] + context_endpoints[0])
    context = context_init * context_range + context_endpoints[0]
    
    features_list = []
    # make the features
    for cont in context:
        
        
        
        if dataset_shape == "triangle":
            end = bias*context_mean
            if bias >= 1: # DAT
                features_list.append(np.random.triangular(context_endpoints[0], (1.0/context_range)*((context_endpoints[1]  - end)*cont + context_endpoints[1]*(end - context_endpoints[0])), context_endpoints[1], size=(1, n_features)))
            else: # SIM
                
                features_list.append(np.random.triangular(context_endpoints[0], (1.0/context_range)*((end - context_endpoints[0])*cont + context_endpoints[0]*(context_endpoints[1] - end)), context_endpoints[1], size=(1, n_features)))
                
        elif dataset_shape == "skewnorm":
            features_list.append(skewnorm.rvs(bias*cont, size=(1, n_features)))
            
    features = np.vstack(features_list)
    return np.hstack((features, context))


"""
LHCO DATASETS (from Sam's code)
"""


def calculate_mass(four_vector):
    return (four_vector[:, 0] ** 2 - np.sum(four_vector[:, 1:4] ** 2, axis=1)) ** 0.5


def get_lhco(sim_type, sm='QCDjj_pT'):
    
   
    directory = '/clusterfs/ml4hep/rrmastandrea/LHCO/'
     
    if sim_type == 'pythia':
        lhco_filename = 'events_anomalydetection_v2.features.h5'
    elif sim_type == 'herwig':
        lhco_filename = 'events_anomalydetection_herwig_features.h5'
        
    print(lhco_filename)
    df = pd.read_hdf(f'{directory}/{lhco_filename}')
    # make_slim(df, directory, lhco_filename)

    # Reorder the features such that the jets are ordered according to their invariant masses
    jet_order_mask = df['mj1'] < df['mj2']
    inverted_keys = ['pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2', 'pxj1', 'pyj1', 'pzj1', 'mj1',
                     'tau1j1', 'tau2j1', 'tau3j1', 'label']
    proper_order = df.loc[jet_order_mask]
    improper_order = df.loc[~jet_order_mask]
    improper_order.columns = inverted_keys
    df = pd.concat((proper_order, improper_order))

    if sm == 'QCDjj_pT':
        df = df.loc[df['label'] == 0]
    else:
        df = df.loc[df['label'] == 1]

    for jet in ['j1', 'j2']:
        df[f'pt{jet}'] = np.sqrt(df[f'px{jet}'] ** 2 + df[f'py{jet}'] ** 2)
        df[f'eta{jet}'] = np.arcsinh(df[f'pz{jet}'] / df[f'pt{jet}'])
        df[f'phi{jet}'] = np.arctan2(df[f'py{jet}'], df[f'px{jet}'])
        df[f'p{jet}'] = np.sqrt(df[f'pz{jet}'] ** 2 + df[f'pt{jet}'] ** 2)
        df[f'e{jet}'] = np.sqrt(df[f'm{jet}'] ** 2 + df[f'p{jet}'] ** 2)

    data = df[['mj1', 'mj2']].copy()
    data['mj2-mj1'] = data['mj2'] - data['mj1']
    data[r'$\tau_{21}^{j_1}$'] = df['tau2j1'] / df['tau1j1']
    data[r'$\tau_{32}^{j_1}$'] = df['tau3j1'] / df['tau2j1']
    data[r'$\tau_{21}^{j_2}$'] = df['tau2j2'] / df['tau1j2']
    data[r'$\tau_{32}^{j_2}$'] = df['tau3j2'] / df['tau2j2']
    # data = pd.DataFrame()
    data[r'$p_t^{j_1}$'] = df['ptj1']
    data[r'$p_t^{j_2}$'] = df['ptj2']
    phi_1 = df['phij1']
    phi_2 = df['phij2']
    delPhi = np.arctan2(np.sin(phi_1 - phi_2), np.cos(phi_1 - phi_2))
    data[r'$dR_{jj}$'] = ((df['etaj1'] - df['etaj2']) ** 2 + delPhi ** 2) ** (0.5)

    data['delPhi'] = abs(delPhi)
    data['delEta'] = abs(df['etaj1'] - df['etaj2'])

    data['mjj'] = calculate_mass(
        np.sum([df[[f'ej{i}', f'pxj{i}', f'pyj{i}', f'pzj{i}']].to_numpy() for i in range(1, 3)], 0))
    return data.dropna()


def select_lhco_features(df, feature_type):
    # TODO properly label the data as true bkg and true signal!!
    
    if feature_type == 0:
        return df[['mj1', 'mj2-mj1', r'$\tau_{21}^{j_1}$', r'$\tau_{21}^{j_2}$', r'$dR_{jj}$', 'mjj']]
    elif feature_type == 1:
        return df[['mj1', 'mj2-mj1', r'$\tau_{21}^{j_1}$', r'$\tau_{21}^{j_2}$', 'mjj']]
    elif feature_type == 2:
        return df[[r'$dR_{jj}$', 'mjj']]



"""
CLASSES
"""

class BandedDataset(Dataset):
    
    """
    Instantiated from another dataset, just for training
    """
    def __init__(self, data, n_features):
        
        self.data = data
        self.n_features = n_features
        self.preprocess = False
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.tensor(item)
    

        

class ToyDataset(Dataset):
    
    """
    Reads in a toy dataset from a file
    """
    def __init__(self, data_dir, data_id):
        
        
        self.path_to_data = os.path.join(data_dir, data_id)
        self.data = np.load(self.path_to_data).astype(np.float32)

        self.n_features = self.data.shape[1] - 1 # 1 of the features is context (mass)
        self.preprocess = False
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.tensor(item)
    
    
    def pull_from_mass_range(self, mass_ranges):

        """
        mass_ranges is a list of lists [low_bound, high_bound]
        """
        if self.preprocess:
            print("Error: Data is preprocessed. Cannot pull from mass range")
            return False

        selected_data = []

        for mass_range in mass_ranges:
            loc = np.where((self.data[:, -1] >= mass_range[0]) & (self.data[:, -1] < mass_range[1]))[0]
            selected_data.append( self.data[loc, :] )
        selected_data = np.concatenate(selected_data)
        np.random.shuffle(selected_data)

        return BandedDataset(selected_data, self.n_features)



def make_train_val_split(dataset, val_split):

    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset


def minmaxscale(data, col_minmax, lower = -3.0, upper = 3.0, forward = True):
    if forward:    
        minmaxscaled_data = np.zeros(data.shape)
        for col in range(data.shape[1]):
            X_std = (data[:, col] - col_minmax[col][0]) / (col_minmax[col][1] - col_minmax[col][0])
            minmaxscaled_data[:, col] = X_std * (upper - lower) + lower      
        return minmaxscaled_data

    else:  
        reversescaled_data = np.zeros(data.shape)
        for col in range(data.shape[1]):
            X_std = (data[:, col] - lower) / (upper - lower)
            reversescaled_data[:, col] = X_std * (col_minmax[col][1] - col_minmax[col][0]) + col_minmax[col][0]
        return reversescaled_data

