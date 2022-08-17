from helpers.datasets import *
from helpers.make_flows import *
from helpers.training import *
from helpers.evaluation import *
from helpers.plotting import *


def create_and_train_flow(keyword, exp_dir, exp_id, transforms, base_dist, hyperparameters_dict, device, train_dataset, val_dataset, early_stop = False):
    
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

    flow_training_dir = os.path.join(exp_dir, f"saved_models_{exp_id}/")
    os.makedirs(flow_training_dir, exist_ok=True)
    
    # Define the flow
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    
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
    
    epochs_learn, losses_learn, epochs_val_learn, losses_val_learn, best_epoch = train_flow(flow, checkpoint_path, optimizer, scheduler, cos_anneal_sched, val_sched, train_dataset, val_dataset, device, n_epochs, batch_size, early_stop = early_stop)

    # Plot the losses
    make_loss_png(epochs_learn, losses_learn, epochs_val_learn, losses_val_learn, loss_img_path)
    
    
    
def create_and_train_flow_bidirectional(exp_dir, exp_id, transforms_s2d, flow_BD_sim, flow_BD_dat, hyperparameters_dict_s2d, device, sim_train_dataset, sim_val_dataset, dat_train_dataset, dat_val_dataset, early_stop = False):
    
    
    """
    Hard-coded to do cosine annealing LR scheduler
    """
    
    """
    "
    "
    CREATE FLOW
    "
    "
    """
    
    keyword = "TRANS"
    
    print("Making models directories ...")
    print()

    flow_training_dir = os.path.join(exp_dir, f"saved_models_{exp_id}/")
    os.makedirs(flow_training_dir, exist_ok=True)
    
    # Define the flow
    transform = CompositeTransform(transforms_s2d)
    flow = Flow(transform, flow_BD_sim) # the DAT base density is used manually in the loss function
    
    # Get the number of parameters in the flow
    num_params = 0
    for param in flow.parameters():
        num_params += 1

    # Write out a flow architecture
    architecture_file = f"model_architecture_{keyword}.txt"

    with open(flow_training_dir+architecture_file, "w") as arch_file:
        arch_file.write(f"Num flow params: {num_params}\n")
        arch_file.write(f"Using SIM base density\n")
        arch_file.write(3*"\n")
        arch_file.write("Transforms:\n")
        arch_file.write(str(transforms_s2d))
        
    """
    "
    "
    TRAIN FLOW
    "
    "
    """
    
    print("Learning the distribution...")
    print()
    
    n_epochs = hyperparameters_dict_s2d["n_epochs"]
    lr = hyperparameters_dict_s2d["lr"]
    weight_decay = hyperparameters_dict_s2d["weight_decay"]
    batch_size = hyperparameters_dict_s2d["batch_size"]
    
    config_string = f"epochs{n_epochs}_lr{lr}_wd{weight_decay}_bs{batch_size}"

    checkpoint_path = os.path.join(flow_training_dir, f"{keyword}_{config_string}")
    loss_img_path = os.path.join(flow_training_dir, f"{keyword}_loss_{config_string}.png")

    # send network to device
    flow.to(device)

    optimizer = optim.AdamW(flow.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min = 0)
    cos_anneal_sched = True
    val_sched = False
    
    epochs_learn, losses_learn, epochs_val_learn, losses_val_learn, best_epoch = train_flow_bidirectional(flow, flow_BD_sim, flow_BD_dat, checkpoint_path, optimizer, scheduler, cos_anneal_sched, val_sched, sim_train_dataset, sim_val_dataset, dat_train_dataset, dat_val_dataset, device, n_epochs, batch_size, early_stop = early_stop)

    # Plot the losses
    make_loss_png(epochs_learn, losses_learn, epochs_val_learn, losses_val_learn, loss_img_path)


def make_base_density_samples(hyperparameters_dict_BD, keyword, exp_dir, exp_id, device, bands_dict, n_features, dataset_sim, binning_scheme):
    
    # Assumes keyword starts with BD
   
    # Load in BD model
    config_string_BD = "epochs{0}_lr{1}_wd{2}_bs{3}".format(hyperparameters_dict_BD["n_epochs"], hyperparameters_dict_BD["lr"], hyperparameters_dict_BD["weight_decay"], hyperparameters_dict_BD["batch_size"])

    
    BD_dir = os.path.join(exp_dir, f"saved_models_{exp_id}/")
    checkpoint_path_BD = os.path.join(BD_dir, f"{keyword}_{config_string_BD}")
    print(f"Loading the {keyword} base density model ...")
    flow_BD = torch.load(f"{checkpoint_path_BD}_best_model.pt")
    
    # make a directory to store the npy samples
    samples_dir = os.path.join(BD_dir, f"npy_samples/")
    os.makedirs(samples_dir, exist_ok = True)
  
    # send network to device
    flow_BD.to(device)

    # freeze the trained model
    for param in flow_BD.parameters():
        param.requires_grad = False
    flow_BD.eval()

    SB_sim_samples, SB_BD_samples = make_BD_samples_dict(bands_dict, n_features, dataset_sim, flow_BD, device)
    
    # Plot the samples
    make_sim_BD_sample_plots(BD_dir, SB_sim_samples, SB_BD_samples, binning_scheme, keyword[2:])
    
    # save out to npy
    np.save(os.path.join(samples_dir, f"SIM_SB.npy"), SB_sim_samples)
    np.save(os.path.join(samples_dir, f"BD_SB.npy"), SB_BD_samples)


  
    
def evaluate_base_density(samples_dir, hyperparameters_dict_BD, keyword, exp_dir, exp_id, device, bands_dict, n_features, dataset_sim, binning_scheme, hyperparameters_dict_eval, use_old_CC = False):
    
    # Assumes keyword starts with BD
    
    epochs_NN = hyperparameters_dict_eval["n_epochs"]
    batch_size_NN = hyperparameters_dict_eval["batch_size"]
    lr_NN = hyperparameters_dict_eval["lr"]
    patience_NN = hyperparameters_dict_eval["patience"]
    num_bootstrap = hyperparameters_dict_eval["num_bootstrap"]
    
   
    # Load in BD model
    config_string_BD = "epochs{0}_lr{1}_wd{2}_bs{3}".format(hyperparameters_dict_BD["n_epochs"], hyperparameters_dict_BD["lr"], hyperparameters_dict_BD["weight_decay"], hyperparameters_dict_BD["batch_size"])

    
    BD_dir = os.path.join(exp_dir, f"saved_models_{exp_id}/")
    checkpoint_path_BD = os.path.join(BD_dir, f"{keyword}_{config_string_BD}")
   
    
    if use_old_CC:
        classifs_results_dir = os.path.join(BD_dir, "oldCC_results/")
        os.makedirs(classifs_results_dir, exist_ok=True)
    
    SB_sim_samples = np.load(os.path.join(samples_dir, f"SIM_SB.npy"))
    SB_BD_samples = np.load(os.path.join(samples_dir, f"BD_SB.npy"))
   
    
    # Use a NN to check how well the base density has been learned for SB1 + SB2  
    sim_BD_rocs = []
    sim_sim_rocs = []
    BD_BD_rocs = []
    
    if num_bootstrap > 1:
        for bs in range(num_bootstrap): # repeat, bootstrapping each time

            if bs % 5 == 0:
                print("On bootstrap", bs, "of", num_bootstrap, "...")

            # get sim indices to bootstrap
            sim_indices_bs = get_bootstrapped_indices(SB_sim_samples.shape[0])

            SB_sim_samples_bs = SB_sim_samples[sim_indices_bs]
            SB_BD_samples_bs = SB_BD_samples[sim_indices_bs]

            # sim - BD
            if not use_old_CC:
                roc = analyze_band_transform("SB1 + SB2", SB_sim_samples_bs[:,:-1], SB_BD_samples_bs[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
            else:
                roc = analyze_band_transforms_CURTAILS_old("SB1 + SB2", SB_sim_samples_bs[:,:-1], SB_BD_samples_bs[:,:-1], classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
            sim_BD_rocs.append(roc)

            # sim - sim
            if not use_old_CC:
                roc = analyze_band_transform("SB1 + SB2", SB_sim_samples_bs[:,:-1], SB_sim_samples_bs[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
            else:
                roc = analyze_band_transforms_CURTAILS_old("SB1 + SB2", SB_sim_samples_bs[:,:-1], SB_sim_samples_bs[:,:-1], classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
            sim_sim_rocs.append(roc)

            # BD - BD
            if not use_old_CC:
                roc = analyze_band_transform("SB1 + SB2", SB_BD_samples_bs[:,:-1], SB_BD_samples_bs[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
            else:
                roc = analyze_band_transforms_CURTAILS_old("SB1 + SB2", SB_BD_samples_bs[:,:-1], SB_BD_samples_bs[:,:-1], classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
            BD_BD_rocs.append(roc)
    
    else:
     # sim - BD
        if not use_old_CC:
            roc = analyze_band_transform("SB1 + SB2", SB_sim_samples[:,:-1], SB_BD_samples[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
        else:
            roc = analyze_band_transforms_CURTAILS_old("SB1 + SB2", SB_sim_samples[:,:-1], SB_BD_samples[:,:-1], classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
        sim_BD_rocs.append(roc)

        # sim - sim
        if not use_old_CC:
            roc = analyze_band_transform("SB1 + SB2", SB_sim_samples[:,:-1], SB_sim_samples[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
        else:
            roc = analyze_band_transforms_CURTAILS_old("SB1 + SB2", SB_sim_samples[:,:-1], SB_sim_samples[:,:-1], classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
        sim_sim_rocs.append(roc)

        # BD - BD
        if not use_old_CC:
            roc = analyze_band_transform("SB1 + SB2", SB_BD_samples[:,:-1], SB_BD_samples[:,:-1], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
        else:
            roc = analyze_band_transforms_CURTAILS_old("SB1 + SB2", SB_BD_samples[:,:-1], SB_BD_samples[:,:-1], classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
        BD_BD_rocs.append(roc)
        
        
    results_file = "BD_SIM_results.txt"

    with open(BD_dir+results_file, "w") as results:
        
        mean_AUC, lower_AUC, upper_AUC = get_bootstrapped_AUC(sim_BD_rocs)
        results.write(f"Discrim. power for SIM from BD in SB1 + SB2: {mean_AUC}; (5%, 95%) = ({lower_AUC}, {upper_AUC})\n")
        mean_AUC, lower_AUC, upper_AUC = get_bootstrapped_AUC(sim_sim_rocs)
        results.write(f"Discrim. power for SIM from SIM in SB1 + SB2: {mean_AUC}; (5%, 95%) = ({lower_AUC}, {upper_AUC})\n")
        mean_AUC, lower_AUC, upper_AUC = get_bootstrapped_AUC(BD_BD_rocs)
        results.write(f"Discrim. power for BD from BD in SB1 + SB2: {mean_AUC}; (5%, 95%) = ({lower_AUC}, {upper_AUC})\n")

        
def make_s2d_samples(hyperparameters_dict_BD, hyperparameters_dict_s2d, exp_dir, exp_id_BD, exp_id_s2d, device, bands_dict, n_features, dataset_sim, dataset_dat, binning_scheme, bidirectional = False):
    
    
    # Load in BD and s2d models
    config_string_BD = "epochs{0}_lr{1}_wd{2}_bs{3}".format(hyperparameters_dict_BD["n_epochs"], hyperparameters_dict_BD["lr"], hyperparameters_dict_BD["weight_decay"], hyperparameters_dict_BD["batch_size"])
    config_string_s2d = "epochs{0}_lr{1}_wd{2}_bs{3}".format(hyperparameters_dict_s2d["n_epochs"], hyperparameters_dict_s2d["lr"], hyperparameters_dict_s2d["weight_decay"], hyperparameters_dict_s2d["batch_size"])

    
    BD_dir = os.path.join(exp_dir, f"saved_models_{exp_id_BD}/")
    checkpoint_path_BD = os.path.join(BD_dir, f"BDSIM_{config_string_BD}")
    
    if bidirectional:
        s2d_dir = os.path.join(exp_dir, f"saved_models_{exp_id_s2d}/")
    else: 
        s2d_dir = os.path.join(BD_dir, f"saved_models_{exp_id_s2d}/")
    checkpoint_path_s2d = os.path.join(s2d_dir, f"TRANS_{config_string_s2d}")
        
    # make a directory to store the npy samples
    samples_dir = os.path.join(s2d_dir, f"npy_samples/")
    os.makedirs(samples_dir, exist_ok = True)

    print("Loading the models ...")
    print()
    flow_BD = torch.load(f"{checkpoint_path_BD}_best_model.pt")
    flow_s2d = torch.load(f"{checkpoint_path_s2d}_best_model.pt")
    
    # send networks to device
    flow_BD.to(device)
    flow_s2d.to(device)

    # freeze the trained models
    for param in flow_BD.parameters():
        param.requires_grad = False
    flow_BD.eval()
    
    for param in flow_s2d.parameters():
        param.requires_grad = False
    flow_s2d.eval()
    
    # Apply the flow to data
    sim_samples, transformed_sim_samples, dat_samples = make_trans_samples_dict(list(bands_dict.keys()), bands_dict,dataset_sim, dataset_dat, flow_s2d, device)
    # Plot all the samples
    make_SIM_transDAT_plots(s2d_dir, sim_samples, transformed_sim_samples, dat_samples, binning_scheme, list(bands_dict.keys()))
    
    # Apply the flow to BD
    SB_sim_samples, SB_BD_samples = make_BD_samples_dict(bands_dict, n_features, dataset_sim, flow_BD, device)
    SB_trans_BD_samples = transform_sim_to_dat(flow_s2d, torch.tensor(SB_BD_samples), device)
    SB_dat_samples = dataset_dat.pull_from_mass_range([bands_dict["sb1"], bands_dict["sb2"]])
    SB_dat_samples.minmaxscale()
    
    SB_BD_samples = SB_BD_samples[:,:-1]
    SB_dat_samples = SB_dat_samples.data[:,:-1]
    make_BD_transBD_plots(s2d_dir, SB_BD_samples, SB_trans_BD_samples, SB_dat_samples, binning_scheme)
    
    # save the npy samples out
    for band in list(bands_dict.keys()):
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
    


    
def evaluate_s2d_density(samples_dir, hyperparameters_dict_BD, hyperparameters_dict_s2d, exp_dir, exp_id_BD, exp_id_s2d, device, bands_dict, n_features, dataset_sim, dataset_dat, binning_scheme, hyperparameters_dict_eval, bidirectional = False, use_old_CC = False):
    
    epochs_NN = hyperparameters_dict_eval["n_epochs"]
    batch_size_NN = hyperparameters_dict_eval["batch_size"]
    lr_NN = hyperparameters_dict_eval["lr"]
    patience_NN = hyperparameters_dict_eval["patience"]
    num_bootstrap = hyperparameters_dict_eval["num_bootstrap"]
    
    BD_dir = os.path.join(exp_dir, f"saved_models_{exp_id_BD}/")
    if bidirectional:
        s2d_dir = os.path.join(exp_dir, f"saved_models_{exp_id_s2d}/")
    else: 
        s2d_dir = os.path.join(BD_dir, f"saved_models_{exp_id_s2d}/")
    
    # Load in npy samples
    sim_samples = {}
    transformed_sim_samples = {}
    dat_samples = {}
    
    for band in list(bands_dict.keys()):
        # sim
        sim_samples[band] = np.load(os.path.join(samples_dir, f"{band}_SIM.npy"))
        # trans sim
        transformed_sim_samples[band] = np.load(os.path.join(samples_dir, f"{band}_transSIM.npy"))
        # dat
        dat_samples[band] = np.load(os.path.join(samples_dir, f"{band}_DAT.npy"))
                
    # BD
    SB_BD_samples = np.load(os.path.join(samples_dir, f"BD.npy"))
    
    # transBD
    SB_trans_BD_samples = np.load(os.path.join(samples_dir, f"transBD.npy"))
    
    # dat_sideband
    SB_dat_samples = np.load(os.path.join(samples_dir, f"DAT_SB.npy"))
    
  
    # Now attempt to discriminate transformed SIM from DAT

    band_rocs = {band:[] for band in bands_dict.keys()}
    untransfomed_band_rocs = {band:[] for band in bands_dict.keys()}
    BD_band_rocs = []

    for band in bands_dict.keys():
        print("Analyzing band", band, "...")
        
        if num_bootstrap > 1:

            for bs in range(num_bootstrap): # repeat, bootstrapping each time

                if bs % 5 == 0:
                    print("On bootstrap", bs, "of", num_bootstrap, "...")

                # get sim, dat indices to bootstrap
                sim_indices_bs = get_bootstrapped_indices(sim_samples[band].shape[0])
                dat_indices_bs = get_bootstrapped_indices(dat_samples[band].shape[0])

                sim_sample_bs = sim_samples[band][sim_indices_bs]
                trans_sim_sample_bs = transformed_sim_samples[band][sim_indices_bs]
                dat_sample_bs = dat_samples[band][dat_indices_bs]

                if not use_old_CC:
                    roc = analyze_band_transform(band, trans_sim_sample_bs, dat_sample_bs, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
                else: 
                    roc = analyze_band_transforms_CURTAILS_old(band, trans_sim_sample_bs, dat_sample_bs, classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
                band_rocs[band].append(roc)

                # Sanity check: attempt to discriminate UNTRANSFORMED SIM from DAT
                if not use_old_CC:
                    ut_roc = analyze_band_transform(band, sim_sample_bs, dat_sample_bs, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
                else: 
                    roc = analyze_band_transforms_CURTAILS_old(band, sim_sample_bs, dat_sample_bs, classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
                untransfomed_band_rocs[band].append(ut_roc) 

                if band == "ob1": # only once do we need to test the transBD vs DAT
                    SB_BD_indices_bs = get_bootstrapped_indices(SB_BD_samples.shape[0])
                    SB_dat_indices_bs = get_bootstrapped_indices(SB_dat_samples.shape[0])

                    SB_trans_BD_samples_bs = SB_trans_BD_samples[SB_BD_indices_bs]
                    SB_dat_samples_bs = SB_dat_samples[SB_dat_indices_bs]
                    if not use_old_CC:
                        roc = analyze_band_transform("SB1 + SB2", SB_trans_BD_samples_bs, SB_dat_samples_bs, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
                    else: 
                        roc = analyze_band_transforms_CURTAILS_old("SB1 + SB2", SB_trans_BD_samples_bs, SB_dat_samples_bs, classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
                    BD_band_rocs.append(roc)
                    
        else:
            
            # Sanity check: attempt to discriminate UNTRANSFORMED SIM from DAT
            if not use_old_CC:
                ut_roc = analyze_band_transform(band, sim_samples[band], dat_samples[band], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
            else:
                ut_roc = analyze_band_transforms_CURTAILS_old(band, "SIM_2_DAT", sim_samples[band], dat_samples[band], classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
            untransfomed_band_rocs[band].append(ut_roc) 


            
            if not use_old_CC:
                roc = analyze_band_transform(band, transformed_sim_samples[band], dat_samples[band], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
            else:
                 roc = analyze_band_transforms_CURTAILS_old(band, "transSIM_2_DAT", transformed_sim_samples[band], dat_samples[band], classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
            band_rocs[band].append(roc)

            
            if band == "ob1": # only once do we need to test the transBD vs DAT
                if not use_old_CC:
                    roc = analyze_band_transform("SB1 + SB2", SB_trans_BD_samples, SB_dat_samples, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
                else:
                    analyze_band_transforms_CURTAILS_old("SB1 + SB2", "BD_2_DAT", SB_trans_BD_samples, SB_dat_samples, classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
                BD_band_rocs.append(roc)
                
        print()


    results_file = "SIM_transSIM_DAT_results.txt"

    with open(s2d_dir+results_file, "w") as results:

        for band in bands_dict.keys():
            
            mean_AUC, lower_AUC, upper_AUC = get_bootstrapped_AUC(band_rocs[band])
            results.write(f"Discrim. power for trans. SIM from DAT in band {band}: {mean_AUC}; (5%, 95%) = ({lower_AUC}, {upper_AUC})\n")

        results.write(3*"\n")

        for band in bands_dict.keys():
            
            mean_AUC, lower_AUC, upper_AUC = get_bootstrapped_AUC(untransfomed_band_rocs[band])
            results.write(f"Discrim. power for SIM from DAT in band {band}: {mean_AUC}; (5%, 95%) = ({lower_AUC}, {upper_AUC})\n")
            
        results.write(3*"\n")
        mean_AUC, lower_AUC, upper_AUC = get_bootstrapped_AUC(BD_band_rocs)
        results.write(f"Discrim. power for trans BD from DAT in SB1 + SB2: {mean_AUC}; (5%, 95%) = ({lower_AUC}, {upper_AUC})\n")

    
    
def evaluate_simBD_to_datBD(hyperparameters_dict_BD_sim, hyperparameters_dict_BD_dat, exp_dir, exp_id_BD_sim, exp_id_BD_dat, device, bands_dict, n_features, dataset_sim, dataset_dat, binning_scheme, hyperparameters_dict_eval, use_old_CC = False):
    
    epochs_NN = hyperparameters_dict_eval["n_epochs"]
    batch_size_NN = hyperparameters_dict_eval["batch_size"]
    lr_NN = hyperparameters_dict_eval["lr"]
    patience_NN = hyperparameters_dict_eval["patience"]
    num_bootstrap = hyperparameters_dict_eval["num_bootstrap"]
    
    # Load in BD and s2d models
    config_string_BD_sim = "epochs{0}_lr{1}_wd{2}_bs{3}".format(hyperparameters_dict_BD_sim["n_epochs"], hyperparameters_dict_BD_sim["lr"], hyperparameters_dict_BD_sim["weight_decay"], hyperparameters_dict_BD_sim["batch_size"])
    config_string_BD_dat = "epochs{0}_lr{1}_wd{2}_bs{3}".format(hyperparameters_dict_BD_dat["n_epochs"], hyperparameters_dict_BD_dat["lr"], hyperparameters_dict_BD_dat["weight_decay"], hyperparameters_dict_BD_dat["batch_size"])

    
    BD_sim_dir = os.path.join(exp_dir, f"saved_models_{exp_id_BD_sim}/")
    checkpoint_path_BD_sim = os.path.join(BD_sim_dir, f"BDSIM_{config_string_BD_sim}")
    
    BD_dat_dir = os.path.join(exp_dir, f"saved_models_{exp_id_BD_dat}/")
    checkpoint_path_BD_dat = os.path.join(BD_dat_dir, f"BDDAT_{config_string_BD_dat}")
    
    s2d_dir = os.path.join(exp_dir, f"simBD_to_datBD_transform/")
    os.makedirs(s2d_dir, exist_ok=True)
        
    if use_old_CC:
        classifs_results_dir = os.path.join(s2d_dir, "oldCC_results/")
        os.makedirs(classifs_results_dir, exist_ok=True)
    
    
    print("Loading the models ...")
    print()
    flow_BD_sim = torch.load(f"{checkpoint_path_BD_sim}_best_model.pt")
    flow_BD_dat = torch.load(f"{checkpoint_path_BD_dat}_best_model.pt")

    # send network to device
    flow_BD_sim.to(device)
    flow_BD_dat.to(device)
    # freeze the trained model
    for param in flow_BD_sim.parameters():
        param.requires_grad = False
    flow_BD_sim.eval()
    for param in flow_BD_dat.parameters():
        param.requires_grad = False
    flow_BD_dat.eval()
    

    # Apply the flow to data
    sim_samples, transformed_sim_samples, dat_samples = make_trans_samples_dict_BD_to_BD(list(bands_dict.keys()), bands_dict, dataset_sim, dataset_dat, flow_BD_sim, flow_BD_dat, device)
    # Plot all the samples
    make_SIM_transDAT_plots(s2d_dir, sim_samples, transformed_sim_samples, dat_samples, binning_scheme, list(bands_dict.keys()))
    
    # Apply the flow to BD
    SB_sim_samples, SB_BD_samples = make_BD_samples_dict(bands_dict, n_features, dataset_sim, flow_BD_sim, device)
    SB_trans_BD_samples = transform_simBD_to_datBD(flow_BD_sim, flow_BD_dat, torch.tensor(SB_BD_samples), device)
    SB_dat_samples = dataset_dat.pull_from_mass_range([bands_dict["sb1"], bands_dict["sb2"]])
    SB_dat_samples.minmaxscale()
    
    SB_BD_samples = SB_BD_samples[:,:-1]
    SB_dat_samples = SB_dat_samples.data[:,:-1]
    make_BD_transBD_plots(s2d_dir, SB_BD_samples, SB_trans_BD_samples, SB_dat_samples, binning_scheme)
    
  
    # Now attempt to discriminate transformed SIM from DAT

    band_rocs = {band:[] for band in bands_dict.keys()}
    untransfomed_band_rocs = {band:[] for band in bands_dict.keys()}
    BD_band_rocs = []

    for band in bands_dict.keys():
        print("Analyzing band", band, "...")
        
        if num_bootstrap > 1:

            for bs in range(num_bootstrap): # repeat, bootstrapping each time

                if bs % 5 == 0:
                    print("On bootstrap", bs, "of", num_bootstrap, "...")

                # get sim, dat indices to bootstrap
                sim_indices_bs = get_bootstrapped_indices(sim_samples[band].shape[0])
                dat_indices_bs = get_bootstrapped_indices(dat_samples[band].shape[0])

                sim_sample_bs = sim_samples[band][sim_indices_bs]
                trans_sim_sample_bs = transformed_sim_samples[band][sim_indices_bs]
                dat_sample_bs = dat_samples[band][dat_indices_bs]

                if not use_old_CC:
                    roc = analyze_band_transform(band, trans_sim_sample_bs, dat_sample_bs, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
                else: 
                    roc = analyze_band_transforms_CURTAILS_old(band, trans_sim_sample_bs, dat_sample_bs, classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
                band_rocs[band].append(roc)

                # Sanity check: attempt to discriminate UNTRANSFORMED SIM from DAT
                if not use_old_CC:
                    ut_roc = analyze_band_transform(band, sim_sample_bs, dat_sample_bs, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
                else: 
                    roc = analyze_band_transforms_CURTAILS_old(band, sim_sample_bs, dat_sample_bs, classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
                untransfomed_band_rocs[band].append(ut_roc) 

                if band == "ob1": # only once do we need to test the transBD vs DAT
                    SB_BD_indices_bs = get_bootstrapped_indices(SB_BD_samples.shape[0])
                    SB_dat_indices_bs = get_bootstrapped_indices(SB_dat_samples.shape[0])

                    SB_trans_BD_samples_bs = SB_trans_BD_samples[SB_BD_indices_bs]
                    SB_dat_samples_bs = SB_dat_samples[SB_dat_indices_bs]
                    if not use_old_CC:
                        roc = analyze_band_transform("SB1 + SB2", SB_trans_BD_samples_bs, SB_dat_samples_bs, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
                    else: 
                        roc = analyze_band_transforms_CURTAILS_old("SB1 + SB2", SB_trans_BD_samples_bs, SB_dat_samples_bs, classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
                    BD_band_rocs.append(roc)
                    
        else:
            if not use_old_CC:
                roc = analyze_band_transform(band, transformed_sim_samples[band], dat_samples[band], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
            else:
                 roc = analyze_band_transforms_CURTAILS_old(band, "transSIM_2_DAT", transformed_sim_samples[band], dat_samples[band], classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
            band_rocs[band].append(roc)

            # Sanity check: attempt to discriminate UNTRANSFORMED SIM from DAT
            if not use_old_CC:
                ut_roc = analyze_band_transform(band, sim_samples[band], dat_samples[band], n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
            else:
                roc = analyze_band_transforms_CURTAILS_old(band, "SIM_2_DAT", sim_samples[band], dat_samples[band], classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
            untransfomed_band_rocs[band].append(ut_roc) 

            if band == "ob1": # only once do we need to test the transBD vs DAT
                if not use_old_CC:
                    roc = analyze_band_transform("SB1 + SB2", SB_trans_BD_samples, SB_dat_samples, n_features, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)
                else:
                    analyze_band_transforms_CURTAILS_old("SB1 + SB2", "BD_2_DAT", SB_trans_BD_samples, SB_dat_samples, classifs_results_dir, epochs_NN, batch_size_NN, lr_NN)
                BD_band_rocs.append(roc)
                
        print()


    results_file = "SIM_transSIM_DAT_results.txt"

    with open(s2d_dir+results_file, "w") as results:

        for band in bands_dict.keys():
            
            mean_AUC, lower_AUC, upper_AUC = get_bootstrapped_AUC(band_rocs[band])
            results.write(f"Discrim. power for trans. SIM from DAT in band {band}: {mean_AUC}; (5%, 95%) = ({lower_AUC}, {upper_AUC})\n")

        results.write(3*"\n")

        for band in bands_dict.keys():
            
            mean_AUC, lower_AUC, upper_AUC = get_bootstrapped_AUC(untransfomed_band_rocs[band])
            results.write(f"Discrim. power for SIM from DAT in band {band}: {mean_AUC}; (5%, 95%) = ({lower_AUC}, {upper_AUC})\n")
            
        results.write(3*"\n")
        mean_AUC, lower_AUC, upper_AUC = get_bootstrapped_AUC(BD_band_rocs)
        results.write(f"Discrim. power for trans BD from DAT in SB1 + SB2: {mean_AUC}; (5%, 95%) = ({lower_AUC}, {upper_AUC})\n")

    

