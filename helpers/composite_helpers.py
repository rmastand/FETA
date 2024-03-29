from helpers.datasets import *
from helpers.make_flows import *
from helpers.training import *
from helpers.evaluation import *
from helpers.plotting import *


def create_and_train_flow(keyword, flow_training_dir, transforms, base_dist, hyperparameters_dict, device, train_dataset, val_dataset, early_stop = False, seed = 2515, L2 = False):
    
    """
    keyword should be BDSIM, BDDAT, or TRANS
    Hard-coded to do cosine annealing LR scheduler
    """
    
    """
    "
    "
    CREATE FLOW
    "
    "
    """
    
    print("Making models directories ...")
    print()

    os.makedirs(flow_training_dir, exist_ok=True)
    
    # Define the flow
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)


    model_parameters = filter(lambda p: p.requires_grad, flow.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters]) 

    print(f"Model has {params} trainable params.")   
    # Get the number of parameters in the flow
    num_params = 0
    for param in flow.parameters():
        num_params += 1

    # Write out a flow architecture
    architecture_file = f"model_architecture_{keyword}.txt"

    with open(flow_training_dir+architecture_file, "w") as arch_file:
        arch_file.write(f"Num flow params: {num_params}\n")
        arch_file.write("Base density:\n")
        arch_file.write(str(base_dist))
        arch_file.write(3*"\n")
        arch_file.write("Transforms:\n")
        arch_file.write(str(transforms))
        
    """
    "
    "
    TRAIN FLOW
    "
    "
    """
    
    print("Learning the distribution...")
    print()
    
    n_epochs = hyperparameters_dict["n_epochs"]
    lr = hyperparameters_dict["lr"]
    weight_decay = hyperparameters_dict["weight_decay"]
    batch_size = hyperparameters_dict["batch_size"]
    
    config_string = f"epochs{n_epochs}_lr{lr}_wd{weight_decay}_bs{batch_size}"

    checkpoint_path = os.path.join(flow_training_dir, f"{keyword}_{config_string}")
    loss_img_path = os.path.join(flow_training_dir, f"{keyword}_loss_{config_string}.png")

    # send network to device
    flow.to(device)

    optimizer = optim.AdamW(flow.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min = 0)
    cos_anneal_sched = True
    val_sched = False
    
    if not L2:
        epochs_learn, losses_learn, epochs_val_learn, losses_val_learn, best_epoch = train_flow(flow, checkpoint_path, optimizer, scheduler, cos_anneal_sched, val_sched, train_dataset, val_dataset, device, n_epochs, batch_size, seed, early_stop = early_stop)

    else:
        alpha = hyperparameters_dict["alpha"]
        epochs_learn, losses_learn, epochs_val_learn, losses_val_learn, best_epoch = train_flow_L2_loss(flow, checkpoint_path, optimizer, scheduler, cos_anneal_sched, val_sched, train_dataset, val_dataset, device, n_epochs, batch_size, seed, alpha, early_stop = early_stop)
        
        
    # Plot the losses
    make_loss_png(epochs_learn, losses_learn, epochs_val_learn, losses_val_learn, loss_img_path)
    
def continue_to_train_flow(keyword, trained_flow_checkpoint_path, flow_training_dir_2, hyperparameters_dict, device, train_dataset_2, val_dataset_2, early_stop = False, seed = 2515, L2 = False):
    
    """
    keyword should be BDSIM, BDDAT, or TRANS
    Hard-coded to do cosine annealing LR scheduler
    
    """
    
    print("Making models directories ...")
    
    print()
    os.makedirs(flow_training_dir_2, exist_ok=True)
    
    """
    "
    "
    LOAD IN TRAINED FLOW
    "
    "
    """
    print(f"Loading the {keyword} base density model ...")
    print()
    flow_trained = torch.load(f"{trained_flow_checkpoint_path}_best_model.pt")
    
    # send network to device
    flow_trained.to(device)
        
    """
    "
    "
    TRAIN FLOW
    "
    "
    """
    
    print("Learning the distribution...")
    print()
    
    n_epochs = hyperparameters_dict["n_epochs"]
    lr = hyperparameters_dict["lr"]
    weight_decay = hyperparameters_dict["weight_decay"]
    batch_size = hyperparameters_dict["batch_size"]
    
    config_string_2 = f"epochs{n_epochs}_lr{lr}_wd{weight_decay}_bs{batch_size}"

    checkpoint_path_2 = os.path.join(flow_training_dir_2, f"{keyword}_{config_string_2}")
    loss_img_path_2 = os.path.join(flow_training_dir_2, f"{keyword}_loss_{config_string_2}.png")

    
    cos_anneal_sched = True
    val_sched = False
    
   

    # 2nd train
    optimizer = optim.AdamW(flow_trained.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min = 0)
    print("2nd train starting")
    print()
    
    if not L2:
        epochs_learn, losses_learn, epochs_val_learn, losses_val_learn, best_epoch = train_flow(flow_trained, checkpoint_path_2, optimizer, scheduler, cos_anneal_sched, val_sched, train_dataset_2, val_dataset_2, device, n_epochs, batch_size, seed, early_stop = early_stop)
    else:
        alpha = hyperparameters_dict["alpha"]
        epochs_learn, losses_learn, epochs_val_learn, losses_val_learn, best_epoch = train_flow_L2_loss(flow_trained, checkpoint_path_2, optimizer, scheduler, cos_anneal_sched, val_sched, train_dataset_2, val_dataset_2, device, n_epochs, batch_size, seed, alpha, early_stop = early_stop)
        
    # Plot the losses
    make_loss_png(epochs_learn, losses_learn, epochs_val_learn, losses_val_learn, loss_img_path_2)
    



def make_base_density_samples(hyperparameters_dict_BD, keyword, flow_training_dir, flow_samples_dir, device, bands_dict, n_features, dataset_sim, binning_scheme, col_minmax):
    
    # Assumes keyword starts with BD
    
    os.makedirs(flow_samples_dir, exist_ok = True)
   
    # Load in BD model
    config_string_BD = "epochs{0}_lr{1}_wd{2}_bs{3}".format(hyperparameters_dict_BD["n_epochs"], hyperparameters_dict_BD["lr"], hyperparameters_dict_BD["weight_decay"], hyperparameters_dict_BD["batch_size"])
    checkpoint_path_BD = os.path.join(flow_training_dir, f"{keyword}_{config_string_BD}")
    
    print(f"Loading the {keyword} base density model ...")
    print()
    flow_BD = torch.load(f"{checkpoint_path_BD}_best_model.pt")
    
    # send network to device
    flow_BD.to(device)

    # freeze the trained model
    for param in flow_BD.parameters():
        param.requires_grad = False
    flow_BD.eval()

    SB_sim_samples, SB_BD_samples = make_BD_samples_dict(["sb1", "sb2"], bands_dict, n_features, dataset_sim, col_minmax, flow_BD, device)
    
    SB_sim_samples_1, SB_BD_samples_1 = make_BD_samples_dict(["sb1"], bands_dict, n_features, dataset_sim, col_minmax, flow_BD, device)
    make_sim_BD_sample_plots(flow_training_dir, SB_sim_samples_1, SB_BD_samples_1, binning_scheme, "SB1")
    SB_sim_samples_2, SB_BD_samples_2 = make_BD_samples_dict(["sb2"], bands_dict, n_features, dataset_sim, col_minmax, flow_BD, device)
    make_sim_BD_sample_plots(flow_training_dir, SB_sim_samples_2, SB_BD_samples_2, binning_scheme, "SB2")
    
    # Plot the samples
    make_sim_BD_sample_plots(flow_training_dir, SB_sim_samples, SB_BD_samples, binning_scheme, keyword[2:])
    
    # save out to npy
    np.save(os.path.join(flow_samples_dir, f"SIM_SB.npy"), SB_sim_samples)
    np.save(os.path.join(flow_samples_dir, f"BD_SB.npy"), SB_BD_samples)


def evaluate_base_density(flow_samples_dir, hyperparameters_dict_BD, keyword, flow_training_dir, device, bands_dict, n_features, hyperparameters_dict_eval):
    
    print("Evaluating base density ...")
    print()
    
    # Assumes keyword starts with BD
    
    epochs_NN = hyperparameters_dict_eval["n_epochs"]
    batch_size_NN = hyperparameters_dict_eval["batch_size"]
    lr_NN = hyperparameters_dict_eval["lr"]
    patience_NN = hyperparameters_dict_eval["patience"]
    
    SB_sim_samples = np.load(os.path.join(flow_samples_dir, f"SIM_SB.npy"))
    SB_BD_samples = np.load(os.path.join(flow_samples_dir, f"BD_SB.npy"))
   
    
    # Use a NN to check how well the base density has been learned for SB1 + SB2  
    sim_BD_rocs = []
    sim_sim_rocs = []
    BD_BD_rocs = []
       

     # sim - BD
    roc = analyze_band_transform(flow_samples_dir, "SB_sim_v_BD", SB_sim_samples[:,:-1], SB_BD_samples[:,:-1], None, None, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
    sim_BD_rocs.append(roc)
   
    roc = analyze_band_transform(flow_samples_dir, "SB_sim_v_sim", SB_sim_samples[:,:-1], SB_sim_samples[:,:-1], None, None, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
    sim_sim_rocs.append(roc)

    roc = analyze_band_transform(flow_samples_dir, "SB_BD_v_BD", SB_BD_samples[:,:-1], SB_BD_samples[:,:-1], None, None, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
    BD_BD_rocs.append(roc)


    results_file = f"{keyword}_results.txt"

    with open(flow_training_dir+results_file, "w") as results:
        
        mean_AUC, lower_AUC, upper_AUC = get_bootstrapped_AUC(sim_BD_rocs)
        results.write(f"Discrim. power for SIM from BD in SB1 + SB2: {mean_AUC}; (5%, 95%) = ({lower_AUC}, {upper_AUC})\n")
        mean_AUC, lower_AUC, upper_AUC = get_bootstrapped_AUC(sim_sim_rocs)
        results.write(f"Discrim. power for SIM from SIM in SB1 + SB2: {mean_AUC}; (5%, 95%) = ({lower_AUC}, {upper_AUC})\n")
        mean_AUC, lower_AUC, upper_AUC = get_bootstrapped_AUC(BD_BD_rocs)
        results.write(f"Discrim. power for BD from BD in SB1 + SB2: {mean_AUC}; (5%, 95%) = ({lower_AUC}, {upper_AUC})\n")

        

def make_s2d_samples(bands_to_sample, hyperparameters_dict_BD_sim, hyperparameters_dict_dat_props, BD_sim_training_dir, dat_props_dir, s2d_dir, device, bands_dict, n_features, dataset_sim, dataset_dat, binning_scheme, col_minmax, direct, oversample = 1):
    
    """
    2_step process: set direct = False
                    dat_dir is the training dir for sim BD -> dat
                    
    direct process: dat_props_dir ("data properties") is the training dir to learn DAT_BD
    
    s2d_dir stores the final transformed samples and plots
    
    """
    
    # Load in BD and s2d models
    config_string_BD_sim = "epochs{0}_lr{1}_wd{2}_bs{3}".format(hyperparameters_dict_BD_sim["n_epochs"], hyperparameters_dict_BD_sim["lr"], hyperparameters_dict_BD_sim["weight_decay"], hyperparameters_dict_BD_sim["batch_size"])
    config_string_dat_props = "epochs{0}_lr{1}_wd{2}_bs{3}".format(hyperparameters_dict_dat_props["n_epochs"], hyperparameters_dict_dat_props["lr"], hyperparameters_dict_dat_props["weight_decay"], hyperparameters_dict_dat_props["batch_size"])

    checkpoint_path_BD_sim = os.path.join(BD_sim_training_dir, f"BDSIM_{config_string_BD_sim}")
    if not direct:
        checkpoint_path_dat_props = os.path.join(dat_props_dir, f"TRANS_{config_string_dat_props}")
    else:
        checkpoint_path_dat_props = os.path.join(dat_props_dir, f"BDDAT_{config_string_dat_props}")
        
    # make a directory to store the npy samples
    samples_dir = os.path.join(s2d_dir, f"npy_samples/")
    os.makedirs(samples_dir, exist_ok = True)

    print("Loading the models ...")
    print()
    flow_BD_sim = torch.load(f"{checkpoint_path_BD_sim}_best_model.pt")
    flow_dat_props = torch.load(f"{checkpoint_path_dat_props}_best_model.pt")
    
    # send networks to device
    flow_BD_sim.to(device)
    flow_dat_props.to(device)

    # freeze the trained models
    for param in flow_BD_sim.parameters():
        param.requires_grad = False
    flow_BD_sim.eval()
    
    for param in flow_dat_props.parameters():
        param.requires_grad = False
    flow_dat_props.eval()
    
    # Apply the flow to data
    if not direct:
        sim_samples, transformed_sim_samples, dat_samples = make_trans_samples_dict_2step(bands_to_sample, bands_dict, dataset_sim, dataset_dat, col_minmax, flow_dat_props, device)
    else:
        sim_samples, transformed_sim_samples, dat_samples = make_trans_samples_dict_direct(bands_to_sample, bands_dict, dataset_sim, dataset_dat, flow_BD_sim, flow_dat_props, device)
        
        
    # Plot all the samples
    make_SIM_transDAT_plots(s2d_dir, sim_samples, transformed_sim_samples, dat_samples, binning_scheme, bands_to_sample)
    
    # Apply the flow to BD
    SB_sim_samples, SB_BD_samples = make_BD_samples_dict(bands_to_sample, bands_dict, n_features, dataset_sim, col_minmax, flow_BD_sim, device, oversample)
    if not direct:
        SB_trans_BD_samples = transform_sim_to_dat_2step(flow_dat_props, SB_BD_samples, device)
    else:
        SB_trans_BD_samples = transform_sim_to_dat_direct(flow_BD_sim, flow_dat_props, torch.tensor(SB_BD_samples), device)
    SB_trans_BD_samples = np.hstack((SB_trans_BD_samples, np.reshape(SB_BD_samples[:,-1], (-1, 1))))

    bands_edges = [bands_dict[band] for band in bands_to_sample]
    SB_dat_samples = dataset_dat.pull_from_mass_range(bands_edges)
    SB_dat_samples = minmaxscale(SB_dat_samples.data, col_minmax, lower = -3, upper = 3, forward = True)
    make_BD_transBD_plots(s2d_dir, SB_BD_samples, SB_trans_BD_samples, SB_dat_samples, binning_scheme, str(bands_to_sample))
    
    # save the npy samples out
    for band in bands_to_sample:
        # sim
        np.save(os.path.join(samples_dir, f"{band}_SIM.npy"), sim_samples[band])
        # trans sim
        np.save(os.path.join(samples_dir, f"{band}_transSIM.npy"), transformed_sim_samples[band])
        # dat
        np.save(os.path.join(samples_dir, f"{band}_DAT.npy"), dat_samples[band])
        
                
    # BD
    np.save(os.path.join(samples_dir, f"BD.npy"), SB_BD_samples)
    
    # transBD
    np.save(os.path.join(samples_dir, f"transBD.npy"), SB_trans_BD_samples)
    
    # dat_sideband
    np.save(os.path.join(samples_dir, f"DAT_SB.npy"), SB_dat_samples)
    
    

def evaluate_s2d(bands_to_sample, s2d_samples_dir, s2d_dir, hyperparameters_dict_eval, device, bands_dict, n_features):
    
    epochs_NN = hyperparameters_dict_eval["n_epochs"]
    batch_size_NN = hyperparameters_dict_eval["batch_size"]
    lr_NN = hyperparameters_dict_eval["lr"]
    patience_NN = hyperparameters_dict_eval["patience"]

    # Load in npy samples
    sim_samples = {}
    transformed_sim_samples = {}
    dat_samples = {}
    
    
    for band in bands_to_sample:
        # sim
        sim_samples[band] = np.load(os.path.join(s2d_samples_dir, f"{band}_SIM.npy"))
        # trans sim
        transformed_sim_samples[band] = np.load(os.path.join(s2d_samples_dir, f"{band}_transSIM.npy"))
        # dat
        dat_samples[band] = np.load(os.path.join(s2d_samples_dir, f"{band}_DAT.npy")) 

    # transBD
    SB_trans_BD_samples = np.load(os.path.join(s2d_samples_dir, f"transBD.npy"))
    
    # dat_sideband
    SB_dat_samples = np.load(os.path.join(s2d_samples_dir, f"DAT_SB.npy"))
  
    # Now attempt to discriminate transformed SIM from DAT

    band_rocs = {band:[] for band in bands_to_sample}
    untransfomed_band_rocs = {band:[] for band in bands_to_sample}
    BD_band_rocs = []
    
    for band in bands_to_sample:
        
        roc = analyze_band_transform(s2d_samples_dir, f"{band}_transSIM_v_DAT", transformed_sim_samples[band][:,:-1], dat_samples[band][:,:-1], None, None, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
        band_rocs[band].append(roc)

        roc = analyze_band_transform(s2d_samples_dir, f"{band}_SIM_v_DAT", sim_samples[band][:,:-1], dat_samples[band][:,:-1], None, None, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
        untransfomed_band_rocs[band].append(roc)
    
    roc = analyze_band_transform(s2d_samples_dir, "SB_transBD_v_DAT", SB_trans_BD_samples[:,:-1], SB_dat_samples[:,:-1], None, None, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
    BD_band_rocs.append(roc)

    
    
   
    print()
            
    results_file = f"SIM_transSIM_DAT_results_{str(bands_to_sample)}.txt"

    with open(s2d_dir+results_file, "w") as results:

        for band in bands_to_sample:
            results.write(f"Discrim. power for trans. SIM from DAT in band {band}: {band_rocs[band][0]}\n")
        results.write(3*"\n")
        
        for band in bands_to_sample:
            results.write(f"Discrim. power for SIM from DAT in band {band}: {untransfomed_band_rocs[band][0]}\n")
        results.write(3*"\n")
        
        results.write(f"Discrim. power for trans BD from DAT in SB1 + SB2: {BD_band_rocs[0]}\n")

    

    
