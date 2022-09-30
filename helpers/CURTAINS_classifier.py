"""
Pulled the functions from the original CURTAINS repo to reproduce the classifier and ROC AUC calculator
"""

import os
from pathlib import Path
from pickle import dump, load

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tqdm import tqdm

import matplotlib.pyplot as plt



def activation(name):
    return {
        'none': nn.Identity(),
        'relu': F.relu,
        'elu': F.elu,
        'leaky_relu': F.leaky_relu,
        'sigmoid': F.sigmoid,
        'tanh': torch.tanh,
        'selu': nn.SELU()
    }[name]



def fit_classifier(classifier, train_data, valid_data, optimizer, batch_size, n_epochs, device, sv_dir, plot=True,
                   load_best=True, scheduler=None, fold=0, visualize = False):
    # Make an object to load training data
    n_workers = 0
    # TODO for some unknown reason the shuffling here will sometimes result in a radix sort error
    data_obj = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    data_valid = torch.utils.data.DataLoader(valid_data, batch_size=1000, shuffle=False, num_workers=n_workers)

    train_loss = np.zeros(n_epochs)
    valid_loss = np.zeros(n_epochs)

    
    scheduler_bool = scheduler is not None
    classifier_dir = os.path.join(sv_dir, f'classifier_{fold}')
    os.makedirs(classifier_dir, exist_ok=True)
    for epoch in tqdm(range(n_epochs), desc='Classifier training'):  # loop over the dataset multiple times
        running_loss = 0
        for i, data in enumerate(data_obj, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # Get the model loss
            data = [dt.to(device) for dt in data]
            loss = classifier.compute_loss(data)
            # Propogate the loss
            loss.backward()
            # Update the parameters
            optimizer.step()
            

            # Get statistics
            running_loss += loss.item()
        if scheduler_bool:
            scheduler.step()

        # Save loss info for the epoch
        train_loss[epoch] = running_loss / (i + 1)
        

        # Validate
        running_loss = 0
        classifier.eval()
        with torch.no_grad():
            for i, data in enumerate(data_valid, 0):
                # Get the model loss
                loss, pred = classifier.compute_loss(data, return_pred=True)
                running_loss += loss.item()
        valid_loss[epoch] = running_loss / (i + 1)
        classifier.save(f'{classifier_dir}/{epoch}')
        classifier.train()

    # Save the validation and training loss metrics
    np.save(f'{classifier_dir}/valid_loss.npy', valid_loss)
    np.save(f'{classifier_dir}/train_loss.npy', train_loss)
    
    if plot:
        # Plot loss development and sic development
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(train_loss, label='Train')
        ax.plot(valid_loss, label='Validation')
        ax.legend()
        ax.set_title('Classifier Training')
        ax.set_xlabel('epochs')
        if visualize:
            plt.show()
        else:
            fig.savefig(f'{classifier_dir}/training_{fold}.png')
            plt.close(fig)
    

    if load_best:
        best_epoch = np.argmin(valid_loss)
        # Index is counted from zero so add one to get the best epoch
        print(f'Best epoch: {best_epoch + 1} loaded')
        classifier.load(f'{classifier_dir}/{best_epoch}')

    classifier.eval()

    return train_loss, valid_loss



def get_optimizer(model, wd, lr):
    if (wd is None) or (wd == 0.):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.NAdam(classifier.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return optimizer


def get_scheduler(use_scheduler, max_step, optimizer):
    if use_scheduler:
        # TODO: pass this, set to one by default
        periods = 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_step / periods, 0)
    else:
        scheduler = None
    return scheduler



def norm_exception():
    raise Exception("Can't set both layer and batch normalization")


class base_network(nn.Module):

    def forward(self, data, context=None):
        NotImplementedError('Must implement a forward method')

    # TODO: make this take data and a batch_size arg so that you can automatically batch the data
    def batch_predict(self, data, context=None):
        store = []
        for data in data:
            store += [self(data, context)]
        return torch.cat(store)


class dense_net(base_network):
    def __init__(self, input_dim, latent_dim, islast=True, output_activ=nn.Identity(), layers=[64, 64, 64], drp=0,
                 batch_norm=False, layer_norm=False, int_activ=torch.relu, context_features=2):
        super(dense_net, self).__init__()
        layers = deepcopy(layers)
        # If adding additional layers to the encoder, don't compress directly to the latent dimension
        # Useful when expanind the capacity of these base models to compare with implicit approach

        self.latent_dim = latent_dim
        self.drp_p = drp
        self.inner_activ = int_activ
        # This is necessary for scaling the outputs to softmax when using splines
        self.hidden_features = layers[-1]

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, layers[0])

        self.functions = nn.ModuleList([nn.Linear(input_dim, layers[0])])

        if islast:
            layers += [latent_dim]

        self.functions.extend(nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]))

        # Change the initilization
        for function in self.functions:
            torch.nn.init.xavier_uniform_(function.weight)
            function.bias.data.fill_(0.0)
        self.output_activ = output_activ

        if batch_norm and layer_norm:
            norm_exception()

        self.norm = 0
        if batch_norm:
            self.norm = 1
            self.norm_func = nn.BatchNorm1d
        if layer_norm:
            self.norm = 1
            self.norm_func = nn.LayerNorm
        if self.norm:
            self.norm_funcs = nn.ModuleList([self.norm_func(layers[i]) for i in range(len(layers) - 1)])

    def forward(self, x, context=None):
        for i, function in enumerate(self.functions[:-1]):
            x = function(x)
            if (context is not None) and (i == 0):
                x += self.context_layer(context)
            if self.norm:
                x = self.norm_funcs[i](x)
            x = self.inner_activ(x)
            x = nn.Dropout(p=self.drp_p)(x)
        x = self.output_activ(self.functions[-1](x))
        return x



class Classifier(nn.Module):

    def __init__(self, base_model, loss_object=F.binary_cross_entropy,
                 loss_name='Classification Error', activation=nn.Identity()):
        super(Classifier, self).__init__()
        self.base_model = base_model
        self.loss_object = loss_object
        self.loss_name = loss_name
        self.activation = activation

    def forward(self, data):
        return self.predict(data)

    def get_scores(self, data):
        return self.base_model(data)

    def predict(self, data):
        return self.activation(self.get_scores(data))

    def device(self):
        return next(self.parameters()).device

    def weighted_loss(self, prediction, target, weight):
        wl = target * -prediction.log() * weight + (1 - target) * -(1 - prediction).log()
        return wl.mean()

    def compute_loss(self, data, return_pred=False):
        inputs, target, weight = data
        if inputs.isnan().any():
            raise Exception('Inputs are NaNs.')
        device = self.device()
        prediction = self.predict(inputs.to(device))
        if prediction.isnan().any():
            raise Exception('Classifier has diverged.')
        self.loss = self.loss_object(prediction, target.to(device), weight=weight.to(device))
        # self.loss = self.weighted_loss(prediction, target.to(device), 0.8)
        if return_pred:
            return self.loss, prediction
        else:
            return self.loss

    def save(self, path):
        torch.save(self.base_model.state_dict(), path)

    def load(self, path):
        self.base_model.load_state_dict(torch.load(path))

    def get_loss_state(self, nsf=10):
        return {self.loss_name: round_numpy(self.loss.item(), nsf)}
    
    
class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, preprocessor=None, weights=1):
        self.preprocessed = False
        self.preprocessor = preprocessor
        if preprocessor is not None:
            X = self.preprocessor.transform(X)
        if weights:
            self.weights = class_weight.compute_sample_weight('balanced', y=y.reshape(-1)).reshape(-1, 1)
        else:
            self.weights = torch.ones_like(y)
        self.X = torch.tensor(X, dtype=torch.float32).cpu()
        self.y = torch.tensor(y, dtype=torch.float32).cpu()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item], self.weights[item]


    
def kfold_gen(scikit_generator, X, y):
    """
    A generator that, given a scikit learn generator, will return groups of train, validation, test indicies
    """
    n = scikit_generator.get_n_splits()
    indicies = np.array([arr[1] for arr in scikit_generator.split(X, y)], dtype=object)
    count = 0
    while count < n:
        yield np.concatenate(indicies[0:3]).flatten().astype(np.int32), \
              indicies[3].astype(np.int32), \
              indicies[4].astype(np.int32)
        count += 1
        indicies = np.roll(indicies, 1, 0)


def assign_scores(true_samples, template_samples, results_dir, nfolds=5, use_weights=True, n_epochs=20, batch_size=128,
                  wd=None, lr=0.001, use_scheduler=True, batch_norm=False, layer_norm=False, width=32, depth=3, drp=0.0,
                  cf_activ='relu', event_id=None, truth_bkg=None, visualize = False):
    
    X = np.vstack((template_samples[:, :-1], true_samples[:, :-1]))
    masses = np.vstack((template_samples[:, -1:], true_samples[:, -1:]))
    print("shape without masses:", X.shape)
    n_templates = len(template_samples)
    y = torch.cat((torch.zeros(n_templates),
                   torch.ones(len(true_samples))), 0).view(-1, 1).cpu().numpy()
    if event_id is None:
        event_id = -torch.ones(len(true_samples))
    if truth_bkg is None:
        truth_bkg = -torch.ones(len(true_samples))
    event_id = torch.cat((-torch.ones(n_templates).to(event_id),
                          event_id)).view(-1, 1).cpu().numpy()
    truth_bkg = torch.cat((-torch.ones(n_templates).to(truth_bkg),
                           truth_bkg)).view(-1, 1).cpu().numpy()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=1)
    split_inds = kfold_gen(kfold, X, y)

    # Train the models
    losses = []
    for fold, (train_index, valid_index, _) in enumerate(split_inds):
        # Split the data and preprocess
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[valid_index], y[valid_index]
        preprocessor = StandardScaler()
        preprocessor.fit(X_train)
        scaler_name = f"{results_dir}/classifier_scaler_{fold}.pkl"
        dump(preprocessor, open(scaler_name, 'wb'))

        # Make the datasets
        train_data = SupervisedDataset(X_train, y_train, preprocessor=preprocessor, weights=use_weights)
        valid_data = SupervisedDataset(X_val, y_val, preprocessor=preprocessor, weights=use_weights)

        # Define a classifier object
        net = dense_net(X.shape[1], 1, layers=[width] * depth, batch_norm=batch_norm, layer_norm=layer_norm,
                        drp=drp, context_features=None, int_activ=activation(cf_activ))
        classifier = Classifier(net, activation=torch.sigmoid).to(device)

        # Set up the optimiser and scheduler
        optimizer = get_optimizer(classifier, wd, lr)
        max_step = int(n_epochs * np.ceil(len(train_data) / batch_size))
        scheduler = get_scheduler(use_scheduler, n_epochs, optimizer)

        losses += [
            fit_classifier(classifier, train_data, valid_data, optimizer, batch_size, n_epochs, device, results_dir,
                           fold=fold, scheduler=scheduler, visualize = visualize)
        ]

    # TODO pick the epoch to take
    epoch_to_load = n_epochs - 1

    # Load and evaluate the models
    preds = []
    split_inds = kfold_gen(kfold, X, y)
    for fold, (_, _, eval_index) in enumerate(split_inds):
        X_eval, y_eval = X[eval_index], y[eval_index]
        id_eval, bkg_truth_eval = event_id[eval_index], truth_bkg[eval_index]
        eval_masses = masses[eval_index]
        scaler_name = f"{results_dir}/classifier_scaler_{fold}.pkl"
        preprocessor = load(open(scaler_name, 'rb'))
        eval_data = SupervisedDataset(X_eval, y_eval, preprocessor=preprocessor, weights=use_weights)
        classifier_dir = Path(results_dir, f'classifier_{fold}')
        classifier.load(classifier_dir / f'{epoch_to_load}')
        with torch.no_grad():
            predictions = classifier.predict(eval_data.X.to(device)).cpu().numpy()
        
        features = [f"feat_{i}" for i in range(X_eval.shape[-1])]
        
        preds += [pd.DataFrame(
            np.concatenate((id_eval, eval_masses, preprocessor.inverse_transform(eval_data.X), y_eval, bkg_truth_eval, predictions), 1),
            columns=['EventID', 'mass']+features+['SoTLabel', 'istrueQCD', 'predictions']
        )]
        # sot = sample or true
        preds[-1]['fold'] = fold
    preds = pd.concat(preds)
    return preds


def get_classification(scores):
    
    fpr, tpr, _ = roc_curve(scores['SoTLabel'], scores['predictions'])
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr