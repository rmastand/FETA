import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from nflows.flows.base import Flow

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from tqdm import tqdm

from helpers.utils import EarlyStopping


def np_to_torch(array, device):
    
    return torch.tensor(array.astype(np.float32)).to(device)


def train_flow(flow, checkpoint_path, optimizer, scheduler, cos_anneal_sched, val_sched, train_dataset, val_dataset, device, n_epochs, batch_size, seed, early_stop = True, patience = 5):
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    update_epochs = 1
    
    epochs, epochs_val = [], []
    losses, losses_val = [], []
    
    print("Training flow for", n_epochs, "epochs ...")
    print("Training:", len(train_dataset), "objects")
    print("Validation:", len(val_dataset), "objects")
    print()
        
    if early_stop:
        early_stopping = EarlyStopping(patience = patience)
        
    # save the best model
    val_losses_to_beat = [10]
    best_epoch = -1
    
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = True)
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = True)
    
 
    for epoch in tqdm(range(n_epochs)):
          
    
        losses_batch_per_e = []
                
        for batch_ndx, data in enumerate(train_data):
            
            
            data = data.to(device)
            feats = data[:,:-1].float()
            cont = torch.reshape(data[:,-1], (-1, 1)).float()
            loss = -flow.log_prob(inputs=feats, context = cont).mean()  
            losses_batch_per_e.append(loss.detach().cpu().numpy())
            optimizer.zero_grad()       
            loss.backward()
            optimizer.step()  
            
        if cos_anneal_sched:
            scheduler.step()
            
        # store the loss
        epochs.append(epoch)
        losses.append(np.mean(losses_batch_per_e))
        
        if epoch % update_epochs == 0: # validation loss
            with torch.no_grad():
                
                val_losses_batch_per_e = []
                
                for batch_ndx, data in enumerate(val_data):
                    data = data.to(device)
                    optimizer.zero_grad()       
                    feats = data[:,:-1].float()
                    cont = torch.reshape(data[:,-1], (-1, 1)).float()
                    val_loss = -flow.log_prob(inputs=feats, context = cont).mean()  
                    val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())
               
                # store the loss
                epochs_val.append(epoch)
                mean_val_loss = np.mean(val_losses_batch_per_e)
                losses_val.append(mean_val_loss)
                
                # see if the model has the best val loss
                for ii in range(len(val_losses_to_beat)):
                    if mean_val_loss < val_losses_to_beat[ii]:
                        val_losses_to_beat[ii] = mean_val_loss
                        # save the model
                        model_path = f"{checkpoint_path}_best_model.pt"
                        torch.save(flow, model_path)
                        best_epoch = epoch
                        break
                
                if early_stop:
                    early_stopping(mean_val_loss)
                if val_sched:
                    scheduler.step(mean_val_loss)
         
        if early_stop:
            if early_stopping.early_stop:
                break
                
    print("Done training!")
    
    with open(f"{checkpoint_path}_best_val_loss.txt", "w") as info_file:
        info_file.write(f"Best validation loss of {val_losses_to_beat[0]} at epoch {best_epoch}")
                                   
    print("Saving the final base density model ...")

    model_path = f"{checkpoint_path}_final_model.pt"
    torch.save(flow, model_path)
                
        
    return epochs, losses, epochs_val, losses_val, best_epoch


def with_L2_loss(tflow, inputs, context, alpha):
    # alpha is the l2 loss param
    
    embedded_context = tflow._embedding_net(context)
    # generic loss for SIM -> DAT transform
    SIM_attempt, logabsdet_s2d = tflow._transform(inputs, context=embedded_context)
    log_prob_s2d = tflow._distribution.log_prob(SIM_attempt, context=embedded_context)
    
    distance_mat = inputs - SIM_attempt
    distances = torch.linalg.norm(distance_mat, axis = 1)
    
    # the regular NF loss is -1 * log prob
    # we're going to call -(this) the loss, so we subtract the distance cost
    
    return log_prob_s2d  + logabsdet_s2d  - alpha*distances

def train_flow_L2_loss(flow, checkpoint_path, optimizer, scheduler, cos_anneal_sched, val_sched, train_dataset, val_dataset, device, n_epochs, batch_size, seed, alpha, early_stop = True, patience = 5):
    
    print("Training flow with L2 loss term ...")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    update_epochs = 1
    
    epochs, epochs_val = [], []
    losses, losses_val = [], []
    
    print("Training flow for", n_epochs, "epochs ...")
    print("Training:", len(train_dataset), "objects")
    print("Validation:", len(val_dataset), "objects")
    print()
        
    if early_stop:
        early_stopping = EarlyStopping(patience = patience)
        
    # save the best model
    val_losses_to_beat = [10]
    best_epoch = -1
    
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = True)
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = True)
    
    # to visualize the probability distributions 
  
    for epoch in tqdm(range(n_epochs)):
          
          
                
        losses_batch_per_e = []
                
        for batch_ndx, data in enumerate(train_data):
            
            
            data = data.to(device)
            feats = data[:,:-1].float()
            cont = torch.reshape(data[:,-1], (-1, 1)).float()
            loss = -with_L2_loss(flow, feats, cont, alpha).mean()
            losses_batch_per_e.append(loss.detach().cpu().numpy())
            optimizer.zero_grad()       
            loss.backward()
            optimizer.step()  
            
        if cos_anneal_sched:
            scheduler.step()
            
        # store the loss
        epochs.append(epoch)
        losses.append(np.mean(losses_batch_per_e))
        
        if epoch % update_epochs == 0: # validation loss
            with torch.no_grad():
                
                val_losses_batch_per_e = []
                
                for batch_ndx, data in enumerate(val_data):
                    data = data.to(device)
                    optimizer.zero_grad()       
                    feats = data[:,:-1].float()
                    cont = torch.reshape(data[:,-1], (-1, 1)).float()
                    val_loss = -with_L2_loss(flow, feats, cont, alpha).mean()
                    val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())
               
                # store the loss
                epochs_val.append(epoch)
                mean_val_loss = np.mean(val_losses_batch_per_e)
                losses_val.append(mean_val_loss)
                
                # see if the model has the best val loss
                for ii in range(len(val_losses_to_beat)):
                    if mean_val_loss < val_losses_to_beat[ii]:
                        val_losses_to_beat[ii] = mean_val_loss
                        # save the model
                        model_path = f"{checkpoint_path}_best_model.pt"
                        torch.save(flow, model_path)
                        best_epoch = epoch
                        break
                
                if early_stop:
                    early_stopping(mean_val_loss)
                if val_sched:
                    scheduler.step(mean_val_loss)
         
        if early_stop:
            if early_stopping.early_stop:
                break
                
    print("Done training!")
    
    with open(f"{checkpoint_path}_best_val_loss.txt", "w") as info_file:
        info_file.write(f"Best validation loss of {val_losses_to_beat[0]} at epoch {best_epoch}")
                                   
    print("Saving the final base density model ...")

    model_path = f"{checkpoint_path}_final_model.pt"
    torch.save(flow, model_path)
                
        
    return epochs, losses, epochs_val, losses_val, best_epoch
    

    



def s2d_loss(tflow, SIM_BD, DAT_input, context):
    embedded_context = tflow._embedding_net(context)
    # generic loss for SIM -> DAT transform
    SIM_attempt, logabsdet_s2d = tflow._transform(DAT_input, context=embedded_context)
    log_prob_s2d = SIM_BD.log_prob(SIM_attempt, context=embedded_context)
    
    return log_prob_s2d  + logabsdet_s2d 


def d2s_loss(tflow, DAT_BD, SIM_input, context):
    embedded_context = tflow._embedding_net(context)

    # modified loss for DAT -> SIM transform
    DAT_attempt, logabsdet_d2s = tflow._transform.inverse(SIM_input, context=embedded_context)
    log_prob_d2s = DAT_BD.log_prob(DAT_attempt, context=embedded_context)
    
    return log_prob_d2s  + logabsdet_d2s


def train_flow_bidirectional(flow, SIM_BD, DAT_BD, checkpoint_path, optimizer, scheduler, cos_anneal_sched, val_sched, sim_train_dataset, sim_val_dataset, dat_train_dataset, dat_val_dataset, device, n_epochs, batch_size, update_epochs = 1, early_stop = True, patience = 10):
    
    torch.manual_seed(8)
    np.random.seed(8)

    epochs, epochs_val = [], []
    losses, losses_val = [], []
    
    print("Training flow for", n_epochs, "epochs ...")
    print("Training:", len(sim_train_dataset), "SIM objects,", len(dat_train_dataset), "DAT objects")
    print("Validation:", len(sim_val_dataset), "SIM objects,", len(dat_val_dataset), "DAT objects")
    print()
        
    if early_stop:
        early_stopping = EarlyStopping(patience = patience)
        
    # save the best model
    val_losses_to_beat = [10]
    best_epoch = -1
    
    sim_train_data = torch.utils.data.DataLoader(sim_train_dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = True)
    sim_val_data = torch.utils.data.DataLoader(sim_val_dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = True)
    dat_train_data = torch.utils.data.DataLoader(dat_train_dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = True)
    dat_val_data = torch.utils.data.DataLoader(dat_val_dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = True)
    
    # to visualize the probability distributions 
  

    for epoch in tqdm(range(n_epochs)):
               
        losses_batch_per_e = []
                
        # first do SIM -> DAT
        for batch_ndx, data in enumerate(dat_train_data):     
            data = data.to(device)
            feats = data[:,:-1].float()
            cont = torch.reshape(data[:,-1], (-1, 1)).float()
            loss = -s2d_loss(flow, SIM_BD, feats, cont).mean()
            losses_batch_per_e.append(loss.detach().cpu().numpy())
            optimizer.zero_grad()       
            loss.backward()
            optimizer.step()  
            
        # first do DAT -> SIM
        for batch_ndx, data in enumerate(sim_train_data):     
            data = data.to(device)
            feats = data[:,:-1].float()
            cont = torch.reshape(data[:,-1], (-1, 1)).float()
            loss = -d2s_loss(flow, DAT_BD, feats, cont).mean()
            losses_batch_per_e.append(loss.detach().cpu().numpy())
            optimizer.zero_grad()       
            loss.backward()
            optimizer.step()  
            
        if cos_anneal_sched:
            scheduler.step()
            
        # store the loss
        epochs.append(epoch)
        losses.append(np.mean(losses_batch_per_e))
        
        if epoch % update_epochs == 0: # validation loss
            with torch.no_grad():
                
                val_losses_batch_per_e = []
                
                for batch_ndx, data in enumerate(dat_val_data):
                    data = data.to(device)
                    optimizer.zero_grad()       
                    feats = data[:,:-1].float()
                    cont = torch.reshape(data[:,-1], (-1, 1)).float()
                    val_loss = -s2d_loss(flow, SIM_BD, feats, cont).mean()
                    val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())
                    
                for batch_ndx, data in enumerate(sim_val_data):
                    data = data.to(device)
                    optimizer.zero_grad()       
                    feats = data[:,:-1].float()
                    cont = torch.reshape(data[:,-1], (-1, 1)).float()
                    val_loss = -d2s_loss(flow, DAT_BD, feats, cont).mean()
                    val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())
               
                # store the loss
                epochs_val.append(epoch)
                mean_val_loss = np.mean(val_losses_batch_per_e)
                losses_val.append(mean_val_loss)
                
                # see if the model has the best val loss
                for ii in range(len(val_losses_to_beat)):
                    if mean_val_loss < val_losses_to_beat[ii]:
                        val_losses_to_beat[ii] = mean_val_loss
                        # save the model
                        model_path = f"{checkpoint_path}_best_model.pt"
                        torch.save(flow, model_path)
                        best_epoch = epoch
                        break
                
                if early_stop:
                    early_stopping(mean_val_loss)
                if val_sched:
                    scheduler.step(mean_val_loss)
         
        if early_stop:
            if early_stopping.early_stop:
                break
                
    print("Done training!")
    
    with open(f"{checkpoint_path}_best_val_loss.txt", "w") as info_file:
        info_file.write(f"Best validation loss of {val_losses_to_beat[0]} at epoch {best_epoch}")
                                   
    print("Saving the final base density model ...")

    model_path = f"{checkpoint_path}_final_model.pt"
    torch.save(flow, model_path)
                
        
    return epochs, losses, epochs_val, losses_val, best_epoch

    

    
