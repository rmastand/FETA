from helpers.datasets import *
from helpers.make_flows import *
from helpers.training import *
from helpers.evaluation import *
from helpers.plotting import *


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