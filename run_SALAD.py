# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

from helpers.composite_helpers import *
from helpers.datasets import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Lambda, Dense, Flatten
from keras import backend as K 

from scipy import interpolate
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
seed = 2
num_signal_to_inject = 1500
project_id = "wide"

np.random.seed(seed)
tf.random.set_seed(seed)



gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# directory setup



n_features = 5 # Note that the SALAD network will also use the mass feature


context_endpoints = (1500, 5500)

bands_dict = {"sb1": [1500, 3300],
              "sr" : [3300, 3700],
              "sb2": [3700, 5500]  }  

binning_scheme = np.linspace(-3.5, 3.5, 50)



feta_dir = "/global/home/users/rrmastandrea/FETA/"
dataset_config_string = f"LHCO_{num_signal_to_inject}sig_{project_id}"
path_to_minmax = f"{feta_dir}/LHCO_STS_{project_id}/data/col_minmax.npy"
col_minmax = np.load(path_to_minmax)

    
exp_dir = os.path.join(feta_dir, dataset_config_string)
#data_dir = os.path.join(exp_dir, "data")
data_dir = f"/global/ml4hep/spss/rrmastandrea/synthsamp_LHCOinput_{project_id}/nsig_{num_signal_to_inject}/data/"


"""
"
"
"
BEGIN HELPER FUNCTIONS
"
"
"
"""

def get_lhco_loc(filepath):
    
    print(filepath)
    df = pd.read_hdf(filepath)
    # make_slim(df, directory, lhco_filename)

    # Reorder the features such that the jets are ordered according to their invariant masses
    jet_order_mask = df['mj1'] < df['mj2']
    inverted_keys = ['pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2', 'pxj1', 'pyj1', 'pzj1', 'mj1',
                     'tau1j1', 'tau2j1', 'tau3j1']
    proper_order = df.loc[jet_order_mask]
    improper_order = df.loc[~jet_order_mask]
    improper_order.columns = inverted_keys
    df = pd.concat((proper_order, improper_order))

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


"""
"
"
"
END HELPER FUNCTIONS
"
"
"
"""


new_herwig_samps_path = "/clusterfs/ml4hep/rrmastandrea/LHCO/events_anomalydetection_DelphesHerwig_qcd_extra_inneronly_features.h5"

df_new_herwig = get_lhco_loc(new_herwig_samps_path)
data_herwig_sr = select_lhco_features(df_new_herwig, feature_type = 0).to_numpy()


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


dataset_train_sim = minmaxscale(dataset_train_sim.data, col_minmax, lower = 0, upper = 1, forward = True)
dataset_val_sim = minmaxscale(dataset_val_sim.data, col_minmax, lower = 0, upper = 1, forward = True)
dataset_train_dat = minmaxscale(dataset_train_dat.data, col_minmax, lower = 0, upper = 1, forward = True)
dataset_val_dat = minmaxscale(dataset_val_dat.data, col_minmax, lower = 0, upper = 1, forward = True)

dataset_sr_sim = minmaxscale(data_herwig_sr, col_minmax, lower = 0, upper = 1, forward = True)


# RUN SALAD


def get_weights(data, model):
    yhat = model.predict(data, batch_size=128)
    return np.squeeze(yhat/(1 - yhat))

stored_weights = {}

X_SALAD_sb_train = np.concatenate([dataset_train_sim, dataset_train_dat])
Y_SALAD_sb_train = np.concatenate([np.zeros(len(dataset_train_sim)), np.ones(len(dataset_train_dat))])

X_SALAD_val = np.concatenate([dataset_val_sim, dataset_val_dat])
Y_SALAD_val = np.concatenate([np.zeros(len(dataset_val_sim)),np.ones(len(dataset_val_dat))])

print('Training SALAD model...')

tf.keras.backend.clear_session()
model_SALAD_sb = Sequential()
model_SALAD_sb.add(Dense(100, input_dim=n_features+1, activation='relu')) 
model_SALAD_sb.add(Dense(100, activation='relu'))
model_SALAD_sb.add(Dense(100, activation='relu'))
model_SALAD_sb.add(Dense(1, activation='sigmoid'))
model_SALAD_sb.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 256

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, mode='min')
mcp_save = tf.keras.callbacks.ModelCheckpoint(f'SALAD_models_{project_id}/{num_signal_to_inject}.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE, verbose=1, epsilon=1e-7, mode='min')


hist_SALAD_sb = model_SALAD_sb.fit(
    X_SALAD_sb_train, Y_SALAD_sb_train, epochs=EPOCHS,
    batch_size=int(BATCH_SIZE), verbose=0, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_data=(X_SALAD_val, Y_SALAD_val)
)


print("Done training!")

# Load in the best weights

model_SALAD_sb.load_weights(f'SALAD_models_{project_id}/{num_signal_to_inject}.mdl_wts.hdf5')
print("Evaluating at best val loss epoch:", np.argmin(hist_SALAD_sb.history['val_loss']))

plot_weights = get_weights(dataset_sr_sim, model_SALAD_sb)


# save out

#scaled_data_dir = f"/global/home/users/rrmastandrea/scaled_data_{project_id}_seed_{seed}/"
scaled_data_dir = f"/global/ml4hep/spss/rrmastandrea/synth_SM_AD/scaled_data_{project_id}_seed_{seed}/"

np.save(f"{scaled_data_dir}/nsig_injected_{num_signal_to_inject}/salad", dataset_sr_sim)
np.save(f"{scaled_data_dir}/nsig_injected_{num_signal_to_inject}/salad_weights", plot_weights)
