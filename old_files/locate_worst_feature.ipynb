{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e4dba465",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "import numpy as np\n",
    "\n",
    "import torch\n",
    "import os\n",
    "\n",
    "from helpers.composite_helpers import *\n",
    "\n",
    "\n",
    "from numba import cuda \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e5e1f6dd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Using device: cuda\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "\"\"\"\n",
    "\"\"\"\n",
    "\"\"\"\n",
    "COMPUTING PARAMETERS\n",
    "\"\"\"\n",
    "\"\"\"\n",
    "\"\"\"\n",
    "\n",
    "os.environ[\"CUDA_VISIBLE_DEVICES\"]=\"0\"\n",
    "device = cuda.get_current_device()\n",
    "device.reset()\n",
    "\n",
    "# set the number of threads that pytorch will use\n",
    "torch.set_num_threads(2)\n",
    "\n",
    "# set gpu device\n",
    "device = torch.device( \"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print( \"Using device: \" + str( device ), flush=True)\n",
    "\n",
    "torch.manual_seed(8)\n",
    "np.random.seed(8)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3289ab46",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Making results directory at /global/home/users/rrmastandrea/curtains/fresh/LHCO_minmax_new/ ...\n"
     ]
    }
   ],
   "source": [
    "\"\"\"\n",
    "\"\"\"\n",
    "\"\"\"\n",
    "RUN PARAMETERS\n",
    "\"\"\"\n",
    "\"\"\"\n",
    "\"\"\"\n",
    "\n",
    "# directories\n",
    "\n",
    "curtains_dir = \"/global/home/users/rrmastandrea/curtains/fresh/\"\n",
    "\n",
    "n_points = 100000\n",
    "n_features = 5\n",
    "dataset_config_string = f\"LHCO_minmax_new/\"\n",
    "\n",
    "exp_dir = os.path.join(curtains_dir, dataset_config_string)\n",
    "data_dir = os.path.join(curtains_dir, dataset_config_string, \"data\")\n",
    "\n",
    "print(\"Making results directory at\", exp_dir, \"...\")\n",
    "os.makedirs(exp_dir, exist_ok=True)\n",
    "\n",
    "data_dir = os.path.join(exp_dir, \"data\")\n",
    "\n",
    "\n",
    "# dataset generation parameters\n",
    "context_endpoints = (3000, 4000)\n",
    "\n",
    "\n",
    "bands_dict = {\"ob1\": [3000, 3200],\n",
    "              \"sb1\": [3200, 3400],\n",
    "              \"sr\" : [3400, 3600],\n",
    "              \"sb2\": [3600, 3800],\n",
    "              \"ob2\": [3800, 4000]}\n",
    "\n",
    "binning_scheme = np.linspace(-4, 4, 50)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "66d9e9a7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Num SIM events: 999986\n",
      "Num DAT events: 999881\n",
      "\n",
      "SIM sb1 + sb2 shape: 124815\n",
      "DAT sb1 + sb2 shape: 124562\n",
      "\n",
      "Preproccessing data...\n",
      "\n"
     ]
    }
   ],
   "source": [
    "\"\"\"\n",
    "\"\"\"\n",
    "\"\"\"\n",
    "LOAD IN THE DATASETS AND PROCESS\n",
    "\"\"\"\n",
    "\"\"\"\n",
    "\"\"\"\n",
    "\n",
    "dataset_sim = ToyDataset(data_dir, \"data_sim.npy\")\n",
    "dataset_dat = ToyDataset(data_dir, \"data_dat.npy\")\n",
    "               \n",
    "\n",
    "print(\"Num SIM events:\", len(dataset_sim))\n",
    "print(\"Num DAT events:\", len(dataset_dat))\n",
    "print()\n",
    "\n",
    "# Pull data from SB1 + SB2\n",
    "training_dataset_sim = dataset_sim.pull_from_mass_range([bands_dict[\"sb1\"], bands_dict[\"sb2\"]])\n",
    "training_dataset_dat = dataset_dat.pull_from_mass_range([bands_dict[\"sb1\"], bands_dict[\"sb2\"]])\n",
    "\n",
    "print(\"SIM sb1 + sb2 shape:\", len(training_dataset_sim))\n",
    "print(\"DAT sb1 + sb2 shape:\", len(training_dataset_dat))\n",
    "print()\n",
    "\n",
    "# Preprocess the data\n",
    "print(\"Preproccessing data...\")\n",
    "print()\n",
    "training_dataset_sim.minmaxscale()\n",
    "training_dataset_dat.minmaxscale()\n",
    "\n",
    "\n",
    "# split into train val\n",
    "val_split = 0.2\n",
    "\n",
    "sim_val_size = int(val_split * len(training_dataset_sim))\n",
    "dat_val_size = int(val_split * len(training_dataset_dat))\n",
    "\n",
    "sim_train_size = len(training_dataset_sim) - sim_val_size\n",
    "dat_train_size = len(training_dataset_dat) - dat_val_size\n",
    "\n",
    "\n",
    "sim_train_dataset, sim_val_dataset = torch.utils.data.random_split(training_dataset_sim, [sim_train_size, sim_val_size])\n",
    "dat_train_dataset, dat_val_dataset = torch.utils.data.random_split(training_dataset_dat, [dat_train_size, dat_val_size])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "675e6084",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\"\"\"\n",
    "\"\n",
    "\"\n",
    "LOAD BASE DENSITY\n",
    "\"\n",
    "\"\n",
    "\"\"\"\n",
    "\n",
    "# Training the BD\n",
    "# This will be the upper subdirectory in saved_models/\n",
    "\n",
    "num_layers_BD = 8\n",
    "num_hidden_features_BD = 64\n",
    "hyperparameters_dict_BD = {\"n_epochs\": 50,\n",
    "                          \"batch_size\": 128,\n",
    "                          \"lr\": 0.0001,\n",
    "                          \"weight_decay\": 0.001}\n",
    "\n",
    "exp_id_BD = f\"Masked_PRQ_AR_{num_layers_BD}layers_{num_hidden_features_BD}hidden_LRCos\" # to distingush the saved models\n",
    "\n",
    "# Define a flow architecture\n",
    "transforms_BD = make_masked_AR_flow(num_layers_BD, n_features, num_hidden_features_BD)\n",
    "base_dist = ConditionalDiagonalNormal(shape=[n_features], context_encoder=nn.Linear(1, 2*n_features))\n",
    "\n",
    "\n",
    "\n",
    "\"\"\"\n",
    "\"\n",
    "\"\n",
    "LOAD SIM -> DAT\n",
    "\"\n",
    "\"\n",
    "\"\"\"\n",
    "\n",
    "# Training s2d\n",
    "# This will be another (of many) subdirectory in saved_models/\n",
    "\n",
    "num_layers_s2d = 2\n",
    "num_blocks_s2d = 8\n",
    "num_nodes_s2d = 32\n",
    "exp_id_s2d = f\"PRQ_Coupling_{num_layers_s2d}layers_{num_blocks_s2d}blocks_{num_nodes_s2d}nodes_LRCos\" # to distingush the saved models\n",
    "transforms_s2d = make_coupling_flow(num_layers_s2d, n_features, num_nodes_s2d, num_blocks = num_blocks_s2d)\n",
    "\n",
    "\n",
    "hyperparameters_dict_s2d = {\"n_epochs\": 50,\n",
    "                          \"batch_size\": 256,\n",
    "                          \"lr\": 0.0002,\n",
    "                          \"weight_decay\": 0.0001}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "197cf867",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading the models ...\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Flow(\n",
       "  (_transform): CompositeTransform(\n",
       "    (_transforms): ModuleList(\n",
       "      (0): PiecewiseRationalQuadraticCouplingTransform(\n",
       "        (transform_net): dense_net(\n",
       "          (context_layer): Linear(in_features=1, out_features=32, bias=True)\n",
       "          (functions): ModuleList(\n",
       "            (0): Linear(in_features=2, out_features=32, bias=True)\n",
       "            (1): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (2): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (3): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (4): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (5): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (6): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (7): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (8): Linear(in_features=32, out_features=87, bias=True)\n",
       "          )\n",
       "          (output_activ): Identity()\n",
       "        )\n",
       "      )\n",
       "      (1): ReversePermutation()\n",
       "      (2): PiecewiseRationalQuadraticCouplingTransform(\n",
       "        (transform_net): dense_net(\n",
       "          (context_layer): Linear(in_features=1, out_features=32, bias=True)\n",
       "          (functions): ModuleList(\n",
       "            (0): Linear(in_features=2, out_features=32, bias=True)\n",
       "            (1): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (2): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (3): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (4): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (5): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (6): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (7): Linear(in_features=32, out_features=32, bias=True)\n",
       "            (8): Linear(in_features=32, out_features=87, bias=True)\n",
       "          )\n",
       "          (output_activ): Identity()\n",
       "        )\n",
       "      )\n",
       "      (3): ReversePermutation()\n",
       "    )\n",
       "  )\n",
       "  (_distribution): Flow(\n",
       "    (_transform): CompositeTransform(\n",
       "      (_transforms): ModuleList(\n",
       "        (0): MaskedPiecewiseRationalQuadraticAutoregressiveTransform(\n",
       "          (autoregressive_net): MADE(\n",
       "            (initial_layer): MaskedLinear(in_features=5, out_features=64, bias=True)\n",
       "            (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "            (blocks): ModuleList(\n",
       "              (0): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (1): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (2): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (3): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (4): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (5): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (6): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (7): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "            )\n",
       "            (final_layer): MaskedLinear(in_features=64, out_features=145, bias=True)\n",
       "          )\n",
       "        )\n",
       "        (1): ReversePermutation()\n",
       "        (2): MaskedPiecewiseRationalQuadraticAutoregressiveTransform(\n",
       "          (autoregressive_net): MADE(\n",
       "            (initial_layer): MaskedLinear(in_features=5, out_features=64, bias=True)\n",
       "            (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "            (blocks): ModuleList(\n",
       "              (0): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (1): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (2): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (3): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (4): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (5): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (6): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (7): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "            )\n",
       "            (final_layer): MaskedLinear(in_features=64, out_features=145, bias=True)\n",
       "          )\n",
       "        )\n",
       "        (3): ReversePermutation()\n",
       "        (4): MaskedPiecewiseRationalQuadraticAutoregressiveTransform(\n",
       "          (autoregressive_net): MADE(\n",
       "            (initial_layer): MaskedLinear(in_features=5, out_features=64, bias=True)\n",
       "            (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "            (blocks): ModuleList(\n",
       "              (0): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (1): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (2): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (3): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (4): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (5): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (6): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (7): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "            )\n",
       "            (final_layer): MaskedLinear(in_features=64, out_features=145, bias=True)\n",
       "          )\n",
       "        )\n",
       "        (5): ReversePermutation()\n",
       "        (6): MaskedPiecewiseRationalQuadraticAutoregressiveTransform(\n",
       "          (autoregressive_net): MADE(\n",
       "            (initial_layer): MaskedLinear(in_features=5, out_features=64, bias=True)\n",
       "            (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "            (blocks): ModuleList(\n",
       "              (0): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (1): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (2): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (3): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (4): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (5): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (6): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (7): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "            )\n",
       "            (final_layer): MaskedLinear(in_features=64, out_features=145, bias=True)\n",
       "          )\n",
       "        )\n",
       "        (7): ReversePermutation()\n",
       "        (8): MaskedPiecewiseRationalQuadraticAutoregressiveTransform(\n",
       "          (autoregressive_net): MADE(\n",
       "            (initial_layer): MaskedLinear(in_features=5, out_features=64, bias=True)\n",
       "            (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "            (blocks): ModuleList(\n",
       "              (0): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (1): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (2): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (3): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (4): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (5): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (6): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (7): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "            )\n",
       "            (final_layer): MaskedLinear(in_features=64, out_features=145, bias=True)\n",
       "          )\n",
       "        )\n",
       "        (9): ReversePermutation()\n",
       "        (10): MaskedPiecewiseRationalQuadraticAutoregressiveTransform(\n",
       "          (autoregressive_net): MADE(\n",
       "            (initial_layer): MaskedLinear(in_features=5, out_features=64, bias=True)\n",
       "            (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "            (blocks): ModuleList(\n",
       "              (0): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (1): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (2): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (3): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (4): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (5): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (6): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (7): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "            )\n",
       "            (final_layer): MaskedLinear(in_features=64, out_features=145, bias=True)\n",
       "          )\n",
       "        )\n",
       "        (11): ReversePermutation()\n",
       "        (12): MaskedPiecewiseRationalQuadraticAutoregressiveTransform(\n",
       "          (autoregressive_net): MADE(\n",
       "            (initial_layer): MaskedLinear(in_features=5, out_features=64, bias=True)\n",
       "            (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "            (blocks): ModuleList(\n",
       "              (0): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (1): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (2): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (3): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (4): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (5): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (6): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (7): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "            )\n",
       "            (final_layer): MaskedLinear(in_features=64, out_features=145, bias=True)\n",
       "          )\n",
       "        )\n",
       "        (13): ReversePermutation()\n",
       "        (14): MaskedPiecewiseRationalQuadraticAutoregressiveTransform(\n",
       "          (autoregressive_net): MADE(\n",
       "            (initial_layer): MaskedLinear(in_features=5, out_features=64, bias=True)\n",
       "            (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "            (blocks): ModuleList(\n",
       "              (0): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (1): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (2): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (3): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (4): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (5): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (6): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "              (7): MaskedResidualBlock(\n",
       "                (context_layer): Linear(in_features=1, out_features=64, bias=True)\n",
       "                (linear_layers): ModuleList(\n",
       "                  (0): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                  (1): MaskedLinear(in_features=64, out_features=64, bias=True)\n",
       "                )\n",
       "                (dropout): Dropout(p=0.05, inplace=False)\n",
       "              )\n",
       "            )\n",
       "            (final_layer): MaskedLinear(in_features=64, out_features=145, bias=True)\n",
       "          )\n",
       "        )\n",
       "        (15): ReversePermutation()\n",
       "      )\n",
       "    )\n",
       "    (_distribution): ConditionalDiagonalNormal(\n",
       "      (_context_encoder): Linear(in_features=1, out_features=10, bias=True)\n",
       "    )\n",
       "    (_embedding_net): Identity()\n",
       "  )\n",
       "  (_embedding_net): Identity()\n",
       ")"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# Load in BD and s2d models\n",
    "config_string_BD = \"epochs{0}_lr{1}_wd{2}_bs{3}\".format(hyperparameters_dict_BD[\"n_epochs\"], hyperparameters_dict_BD[\"lr\"], hyperparameters_dict_BD[\"weight_decay\"], hyperparameters_dict_BD[\"batch_size\"])\n",
    "config_string_s2d = \"epochs{0}_lr{1}_wd{2}_bs{3}\".format(hyperparameters_dict_s2d[\"n_epochs\"], hyperparameters_dict_s2d[\"lr\"], hyperparameters_dict_s2d[\"weight_decay\"], hyperparameters_dict_s2d[\"batch_size\"])\n",
    "\n",
    "\n",
    "BD_dir = os.path.join(exp_dir, f\"saved_models_{exp_id_BD}/\")\n",
    "checkpoint_path_BD = os.path.join(BD_dir, f\"BDSIM_{config_string_BD}\")\n",
    "\n",
    "\n",
    "s2d_dir = os.path.join(BD_dir, f\"saved_models_{exp_id_s2d}/\")\n",
    "\n",
    "checkpoint_path_s2d = os.path.join(s2d_dir, f\"TRANS_{config_string_s2d}\")\n",
    "\n",
    "print(\"Loading the models ...\")\n",
    "print()\n",
    "flow_BD = torch.load(f\"{checkpoint_path_BD}_best_model.pt\")\n",
    "flow_s2d = torch.load(f\"{checkpoint_path_s2d}_best_model.pt\")\n",
    "\n",
    "# send networks to device\n",
    "flow_BD.to(device)\n",
    "flow_s2d.to(device)\n",
    "\n",
    "# freeze the trained models\n",
    "for param in flow_BD.parameters():\n",
    "    param.requires_grad = False\n",
    "flow_BD.eval()\n",
    "\n",
    "for param in flow_s2d.parameters():\n",
    "    param.requires_grad = False\n",
    "flow_s2d.eval()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a9b6a1a4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Making samples for band ob1 ...\n",
      "Making samples for band sb1 ...\n",
      "Making samples for band sr ...\n",
      "Making samples for band sb2 ...\n",
      "Making samples for band ob2 ...\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x288 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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/M43vOEN/NVf/RWrV6/mXe96F1/72tdmje+KK65g69atPPHEE9x9992ce27b+6E37Q0zItZGxK6I2LVv375WxDpvx02MHfywpM5ibgINFGEz83uFzedAC8ZOl8zXgtSZeiY3x8c5WPWJODhKP7O68l7xtn79Iv5g6gZXAlsycxnVVan/LCIO+Szbide06kw9c55cgDmLQBFxE3Ab8OKIGI+It0TEW2vD3KH6wXIPsBv4JNC+SXXqO+eccw4rV648uP3Rj370YAV27969PPDAAwA861nP4pd/+ZcBOPvssxmtjaU///zzufrqq/nkJz/J008/fcjPX7ZsGffffz//6T/9J4444ghe8YpX8Hd/93czxnLmmWcyOjrKTTfdxKte1bG96Rp6w8zMTZk5lJlDJ510UtuDnDQ58md4JNg/MFhaHJq/PszNRoqwRMTbIuJBqiOB3l54amWtueXfR8RLZ/sl3Xgx24evhYOKoxQiHJmgztKVufn001PD4rI5dfTJqZmTt+UxPm1EUaXSlF+j+WlkZslbgJsBMvM2YClwYv0P6pRrWnWnrjxPLsCcE90y87DjjjIzqfY8kFquOGdyx44d/O3f/i233XYbz372s7ngggt44oknADjqqKOIWkOFJUuWcODAAQA+/vGP89WvfpUvfvGLnH322dxxxx2ccMIJ037H0UcfzSWXXMIll1zCv/t3/44vfOELvOIVr5gxnksvvZR3v/vd7Nix42BluI0afcNcA9U3zIiYfMN8pC0RzpOjfrqXuTmzzLwRuDEirqLao+tNVBdOWJGZj0bE2cAXIuK0upFDk8d3VS896O/XQn0vwYU2uWx2c0wJ+js3i+p7cX1rJKZdf4z0QK+TLrQTWBURK6ley14BXFW3z7eAVwBbIuIUqkWg7vh2RF2jX86TbW0M3ekOWQ2MUeyhW55jjz2W73//+7M+PzExwfHHH8+zn/1s/uVf/oXbb799zp/54IMPcu6553LuuefypS99ib17905LzDvvvJOf+Zmf4fnPfz7PPPMMd999N2eeeeasP+/Nb34zxx13HGeccQY7duyY17+vCXzDVCnMTWD+/fC2An8MkJlPAk/W7t9RGyn0ImBXKwJtJV8LC1dsFH24JtHSQpib6iaZeSAi1gG3UF1sYXNm3hMR1wG7MnMb8C7gkxHxf1OdXn11bTCCtCD9fJ60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\n",
      "text/plain": [
       "<Figure size 1440x288 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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+1gEUlc566yzWLVq1dTxxz72salq+J49e7jnnnsAOOKII/jlX/5lAM4880xGa+t6zz33XC677DI++clPsn///gO+/vLly7n77rv5L//lv3DYYYdx/vnn89WvfnXWWM444wxGR0e57rrreMUrXtHkf2lDdgKrI2JVRBwBrAfqf/N/DzgfICJOploEeqitUaovmJua5HNB6kxdkZu33w67dk1flixp0r9+2t6BQYZHYuqycXOl6d9DWoiuyE31FHsCqasU96/u2LGDv/u7v+Mb3/gGz3zmM1m7di2PP/44AM94xjOI2gbvJUuWsG/fPgA+/vGPc/PNN/OlL32JM888k1tuuYXjjjtuxvc48sgjufDCC7nwwgv5d//u3/HFL36R888/f9Z4LrroIt75zneyY8eOqSp9u2Tmvoi4HLiR6vj3azPzjoi4CtiVmduBdwCfjIj/SLVJ9GWZndthcXhkelP+3gFnvXaTfszNiFgHfJRq/n0qM6+uu/8twFuB/cCPgQ2ZeWdEVIC7gLtrp96UmW9pSZAl6MfngtQNuiI3n3wShoamT7rrrmb986fUD34pvveQytAVuQmc8P1dU9dv58CeQuoeFoHUsY455hh+9KMfzXn/xMQExx57LM985jP553/+Z2666aZ5v+Z9993H2Wefzdlnn82Xv/xl9uzZM+OX5K233srP/dzPccIJJ/D0009z++23c8YZZ8z59d70pjexdOlSTj/9dHbs2LGgf18zZOYNVMe+F2+7snD9TuDcdsd1qKamgamjmZsQEUuAa4CXUm3KvjMittdybtJnMvPjtfMvAj4CrKvdd19mrml6YG3mc0HqTL2am+2YuFdsFG2TaDVbN+fm908oFGh/eGgF2gr3M1aY+uvkzHJYBFLD2r1X87jjjuPcc8/ltNNO48ILL+SVr3zljPvXrVvHxz/+cU4++WSe//znc84558z7Nd/1rndxzz33kJmcf/75vOAFL5hx/4MPPshv/dZv8cQTTwDV5ZmXX375nF9v+fLlvO1tbzuEf53UPObmgdqQm2cB92bmboCI2AZcDEwVgTLzscL5z6K6Gq+lfC4cyN/T6gTm5oEOJTcnJqp9s1qpWPQJFwn1PHPzQK183Ryj0vIc1vyirJ0hQ0NDuWvXrvlPbKOIqWEewIG/FEZGZt7f6+666y5OPvnkssPoerP9HCPilswc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\n",
      "text/plain": [
       "<Figure size 1440x288 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x288 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x288 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Apply the flow to data\n",
    "sim_samples, transformed_sim_samples, dat_samples = make_trans_samples_dict(list(bands_dict.keys()), bands_dict,dataset_sim, dataset_dat, flow_s2d, device)\n",
    "# Plot all the samples\n",
    "   \n",
    "bands_to_plot = list(bands_dict.keys())\n",
    "\n",
    "n_features = sim_samples[bands_to_plot[0]].shape[-1]\n",
    "\n",
    "alpha = 0.5\n",
    "    \n",
    "for band in bands_to_plot:\n",
    "\n",
    "    fig, ax = plt.subplots(1, n_features, figsize = (4*n_features, 4))\n",
    "    for i in range(n_features):\n",
    "\n",
    "        ax[i].hist(sim_samples[band][:,i], bins = binning_scheme, label = \"SIM\", histtype = \"step\", color = \"red\", density = True)\n",
    "        ax[i].hist(transformed_sim_samples[band][:,i], bins = binning_scheme, label = \"trans SIM\", alpha = alpha, color = \"blue\", density = True)\n",
    "        ax[i].hist(dat_samples[band][:,i], bins = binning_scheme, label = \"DAT\", histtype = \"step\", color = \"blue\", density = True)\n",
    "        ax[i].legend()\n",
    "        ax[i].set_title(f\"f{i}\")\n",
    "    fig.suptitle(band)      \n",
    "\n",
    "    fig.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "456811d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "hyperparameters_dict_eval = {\"n_epochs\": 200,\n",
    "                          \"batch_size\": 128,\n",
    "                          \"lr\": 0.0005,\n",
    "                          \"num_bootstrap\": 10,\n",
    "                            \"patience\": 10}\n",
    "\n",
    "epochs_NN = hyperparameters_dict_eval[\"n_epochs\"]\n",
    "batch_size_NN = hyperparameters_dict_eval[\"batch_size\"]\n",
    "lr_NN = hyperparameters_dict_eval[\"lr\"]\n",
    "patience_NN = hyperparameters_dict_eval[\"patience\"]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "d5b77208",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band ob1 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 10%|█         | 20/200 [00:33<05:05,  1.70s/it]\n",
      " 12%|█▏        | 24/200 [00:40<04:56,  1.68s/it]\n",
      "  7%|▋         | 14/200 [00:24<05:24,  1.75s/it]\n",
      " 11%|█         | 22/200 [00:35<04:49,  1.62s/it]\n",
      " 12%|█▏        | 24/200 [00:38<04:44,  1.62s/it]\n",
      " 12%|█▎        | 25/200 [00:41<04:50,  1.66s/it]\n",
      " 18%|█▊        | 37/200 [01:00<04:24,  1.62s/it]\n",
      " 13%|█▎        | 26/200 [00:41<04:40,  1.61s/it]\n",
      " 18%|█▊        | 35/200 [00:58<04:34,  1.66s/it]\n",
      " 18%|█▊        | 36/200 [00:57<04:22,  1.60s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sb1 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 12%|█▏        | 23/200 [00:28<03:38,  1.23s/it]\n",
      " 12%|█▎        | 25/200 [00:31<03:39,  1.26s/it]\n",
      "  8%|▊         | 16/200 [00:21<04:02,  1.32s/it]\n",
      " 16%|█▋        | 33/200 [00:40<03:22,  1.21s/it]\n",
      " 12%|█▏        | 24/200 [00:29<03:36,  1.23s/it]\n",
      " 16%|█▌        | 32/200 [00:38<03:24,  1.22s/it]\n",
      " 12%|█▏        | 24/200 [00:29<03:35,  1.23s/it]\n",
      "  8%|▊         | 15/200 [00:18<03:52,  1.26s/it]\n",
      " 10%|▉         | 19/200 [00:23<03:44,  1.24s/it]\n",
      " 12%|█▎        | 25/200 [00:30<03:36,  1.24s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sr ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 12%|█▏        | 23/200 [00:23<03:02,  1.03s/it]\n",
      " 10%|█         | 21/200 [00:19<02:47,  1.07it/s]\n",
      "  6%|▋         | 13/200 [00:12<02:59,  1.04it/s]\n",
      "  6%|▌         | 12/200 [00:11<03:01,  1.04it/s]\n",
      " 12%|█▏        | 24/200 [00:22<02:43,  1.08it/s]\n",
      " 10%|▉         | 19/200 [00:17<02:49,  1.07it/s]\n",
      " 10%|▉         | 19/200 [00:17<02:49,  1.07it/s]\n",
      "  8%|▊         | 16/200 [00:16<03:13,  1.05s/it]\n",
      "  7%|▋         | 14/200 [00:13<02:57,  1.05it/s]\n",
      " 12%|█▏        | 24/200 [00:22<02:43,  1.08it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sb2 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 14%|█▎        | 27/200 [00:18<01:58,  1.46it/s]\n",
      "  6%|▋         | 13/200 [00:09<02:12,  1.41it/s]\n",
      " 12%|█▏        | 23/200 [00:15<02:01,  1.45it/s]\n",
      "  7%|▋         | 14/200 [00:09<02:11,  1.42it/s]\n",
      " 10%|█         | 21/200 [00:15<02:11,  1.36it/s]\n",
      "  6%|▌         | 11/200 [00:07<02:15,  1.39it/s]\n",
      " 13%|█▎        | 26/200 [00:17<01:59,  1.46it/s]\n",
      "  8%|▊         | 17/200 [00:12<02:14,  1.36it/s]\n",
      "  8%|▊         | 16/200 [00:12<02:24,  1.28it/s]\n",
      "  8%|▊         | 15/200 [00:10<02:13,  1.38it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band ob2 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 16%|█▋        | 33/200 [00:17<01:28,  1.89it/s]\n",
      " 14%|█▍        | 29/200 [00:14<01:27,  1.96it/s]\n",
      " 18%|█▊        | 36/200 [00:18<01:23,  1.97it/s]\n",
      "  8%|▊         | 16/200 [00:08<01:36,  1.91it/s]\n",
      " 22%|██▎       | 45/200 [00:23<01:21,  1.90it/s]\n",
      "  9%|▉         | 18/200 [00:10<01:48,  1.67it/s]\n",
      " 14%|█▎        | 27/200 [00:14<01:34,  1.83it/s]\n",
      " 10%|█         | 20/200 [00:13<01:57,  1.53it/s]\n",
      " 14%|█▎        | 27/200 [00:14<01:35,  1.81it/s]\n",
      " 14%|█▍        | 28/200 [00:14<01:27,  1.98it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ob1 roc: 0.521201 0.515726 0.525737\n",
      "sb1 roc: 0.50402 0.500169 0.510374\n",
      "sr roc: 0.511295 0.502971 0.519255\n",
      "sb2 roc: 0.507545 0.501247 0.519499\n",
      "ob2 roc: 0.520554 0.513876 0.529415\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "n_repeat = 10\n",
    "\n",
    "\n",
    "band_rocs = {band:[] for band in list(bands_dict.keys())}\n",
    "for band in bands_dict.keys():\n",
    "    print(\"Analyzing band\", band, \"...\")\n",
    "    for j in range(n_repeat):\n",
    "        roc = analyze_band_transform(band, transformed_sim_samples[band], dat_samples[band], 5, epochs_NN, batch_size_NN, lr_NN, patience_NN, device)\n",
    "        band_rocs[band].append(roc)\n",
    "for band in bands_dict.keys():\n",
    "    mean, lower, upper = get_bootstrapped_AUC(band_rocs[band], lower_percentile = 5, upper_percentile = 95, num_decimals = 6)\n",
    "    print(band, \"roc:\", mean, lower, upper)\n",
    "print()\n",
    "        "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6080adac",
   "metadata": {},
   "source": [
    "## Analyze 1 feature at a time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "d70582d6",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing features [0] only...\n",
      "Analyzing band ob1 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 12%|█▏        | 23/200 [00:41<05:15,  1.78s/it]\n",
      "  6%|▋         | 13/200 [00:26<06:22,  2.05s/it]\n",
      " 10%|█         | 20/200 [00:35<05:21,  1.78s/it]\n",
      " 16%|█▌        | 32/200 [00:54<04:43,  1.69s/it]\n",
      "  6%|▌         | 12/200 [00:23<06:00,  1.92s/it]\n",
      " 14%|█▎        | 27/200 [00:47<05:06,  1.77s/it]\n",
      "  8%|▊         | 15/200 [00:24<05:05,  1.65s/it]\n",
      " 10%|█         | 20/200 [00:33<05:02,  1.68s/it]\n",
      " 17%|█▋        | 34/200 [00:55<04:33,  1.65s/it]\n",
      " 14%|█▍        | 28/200 [00:49<05:02,  1.76s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sb1 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 16%|█▌        | 31/200 [00:38<03:31,  1.25s/it]\n",
      " 10%|▉         | 19/200 [00:25<04:02,  1.34s/it]\n",
      " 10%|▉         | 19/200 [00:23<03:48,  1.26s/it]\n",
      " 10%|█         | 21/200 [00:25<03:36,  1.21s/it]\n",
      "  7%|▋         | 14/200 [00:18<04:02,  1.30s/it]\n",
      " 17%|█▋        | 34/200 [00:41<03:21,  1.22s/it]\n",
      " 14%|█▎        | 27/200 [00:33<03:33,  1.23s/it]\n",
      " 12%|█▏        | 24/200 [00:33<04:02,  1.38s/it]\n",
      " 16%|█▌        | 31/200 [00:37<03:24,  1.21s/it]\n",
      "  8%|▊         | 16/200 [00:19<03:47,  1.23s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sr ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 12%|█▎        | 25/200 [00:23<02:44,  1.07it/s]\n",
      " 10%|▉         | 19/200 [00:17<02:46,  1.09it/s]\n",
      " 10%|▉         | 19/200 [00:17<02:48,  1.08it/s]\n",
      " 10%|█         | 21/200 [00:19<02:45,  1.08it/s]\n",
      " 15%|█▌        | 30/200 [00:27<02:35,  1.10it/s]\n",
      " 10%|█         | 21/200 [00:19<02:45,  1.08it/s]\n",
      "  6%|▌         | 12/200 [00:11<03:00,  1.04it/s]\n",
      " 18%|█▊        | 37/200 [00:34<02:29,  1.09it/s]\n",
      " 13%|█▎        | 26/200 [00:23<02:39,  1.09it/s]\n",
      "  9%|▉         | 18/200 [00:16<02:49,  1.07it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sb2 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  6%|▌         | 11/200 [00:07<02:14,  1.40it/s]\n",
      " 11%|█         | 22/200 [00:15<02:01,  1.46it/s]\n",
      "  8%|▊         | 17/200 [00:11<02:07,  1.44it/s]\n",
      " 10%|▉         | 19/200 [00:13<02:05,  1.44it/s]\n",
      "  6%|▌         | 12/200 [00:08<02:14,  1.40it/s]\n",
      "  6%|▌         | 12/200 [00:08<02:14,  1.40it/s]\n",
      " 16%|█▋        | 33/200 [00:22<01:53,  1.47it/s]\n",
      " 12%|█▏        | 23/200 [00:16<02:10,  1.36it/s]\n",
      "  8%|▊         | 17/200 [00:11<02:07,  1.43it/s]\n",
      " 14%|█▎        | 27/200 [00:18<01:58,  1.46it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band ob2 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  6%|▌         | 12/200 [00:06<01:40,  1.87it/s]\n",
      " 11%|█         | 22/200 [00:11<01:31,  1.94it/s]\n",
      "  5%|▌         | 10/200 [00:05<01:43,  1.84it/s]\n",
      " 13%|█▎        | 26/200 [00:14<01:36,  1.80it/s]\n",
      " 20%|█▉        | 39/200 [00:20<01:24,  1.90it/s]\n",
      "  9%|▉         | 18/200 [00:09<01:39,  1.82it/s]\n",
      "  8%|▊         | 17/200 [00:08<01:36,  1.90it/s]\n",
      " 14%|█▍        | 28/200 [00:14<01:27,  1.95it/s]\n",
      " 19%|█▉        | 38/200 [00:19<01:22,  1.97it/s]\n",
      "  8%|▊         | 15/200 [00:07<01:37,  1.90it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ob1 roc: 0.508719 0.5019 0.51384\n",
      "sb1 roc: 0.503421 0.5001 0.507077\n",
      "sr roc: 0.509888 0.50541 0.512542\n",
      "sb2 roc: 0.504949 0.5013 0.511582\n",
      "ob2 roc: 0.509189 0.502609 0.520402\n",
      "\n",
      "Analyzing features [1] only...\n",
      "Analyzing band ob1 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 10%|█         | 20/200 [00:32<04:53,  1.63s/it]\n",
      " 15%|█▌        | 30/200 [00:48<04:33,  1.61s/it]\n",
      "  6%|▋         | 13/200 [00:23<05:40,  1.82s/it]\n",
      "  6%|▌         | 11/200 [00:18<05:20,  1.70s/it]\n",
      " 12%|█▏        | 24/200 [00:41<05:06,  1.74s/it]\n",
      " 14%|█▍        | 29/200 [00:47<04:42,  1.65s/it]\n",
      " 11%|█         | 22/200 [00:36<04:54,  1.65s/it]\n",
      "  8%|▊         | 17/200 [00:30<05:30,  1.81s/it]\n",
      " 10%|█         | 21/200 [00:34<04:51,  1.63s/it]\n",
      " 14%|█▎        | 27/200 [00:43<04:40,  1.62s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sb1 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  6%|▋         | 13/200 [00:16<03:57,  1.27s/it]\n",
      " 13%|█▎        | 26/200 [00:31<03:33,  1.23s/it]\n",
      "  8%|▊         | 17/200 [00:21<03:48,  1.25s/it]\n",
      " 10%|█         | 20/200 [00:24<03:43,  1.24s/it]\n",
      "  8%|▊         | 17/200 [00:21<03:48,  1.25s/it]\n",
      " 14%|█▍        | 28/200 [00:33<03:28,  1.21s/it]\n",
      "  5%|▌         | 10/200 [00:15<04:48,  1.52s/it]\n",
      " 14%|█▍        | 29/200 [00:35<03:29,  1.22s/it]\n",
      " 10%|▉         | 19/200 [00:23<03:44,  1.24s/it]\n",
      " 12%|█▏        | 24/200 [00:29<03:36,  1.23s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sr ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 13%|█▎        | 26/200 [00:24<02:40,  1.08it/s]\n",
      "  8%|▊         | 16/200 [00:15<02:53,  1.06it/s]\n",
      "  8%|▊         | 16/200 [00:15<02:53,  1.06it/s]\n",
      "  8%|▊         | 15/200 [00:14<03:01,  1.02it/s]\n",
      "  8%|▊         | 17/200 [00:16<02:52,  1.06it/s]\n",
      " 10%|▉         | 19/200 [00:18<02:51,  1.05it/s]\n",
      " 13%|█▎        | 26/200 [00:25<02:52,  1.01it/s]\n",
      " 12%|█▎        | 25/200 [00:23<02:43,  1.07it/s]\n",
      " 14%|█▍        | 29/200 [00:27<02:42,  1.05it/s]\n",
      " 12%|█▏        | 24/200 [00:22<02:43,  1.08it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sb2 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 10%|█         | 21/200 [00:14<02:03,  1.45it/s]\n",
      " 12%|█▎        | 25/200 [00:17<02:01,  1.44it/s]\n",
      " 15%|█▌        | 30/200 [00:21<02:04,  1.37it/s]\n",
      " 11%|█         | 22/200 [00:15<02:06,  1.41it/s]\n",
      "  9%|▉         | 18/200 [00:12<02:07,  1.43it/s]\n",
      "  6%|▋         | 13/200 [00:09<02:10,  1.43it/s]\n",
      "  7%|▋         | 14/200 [00:09<02:07,  1.46it/s]\n",
      " 18%|█▊        | 37/200 [00:24<01:50,  1.48it/s]\n",
      " 16%|█▌        | 31/200 [00:21<01:59,  1.41it/s]\n",
      "  6%|▌         | 11/200 [00:07<02:16,  1.39it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band ob2 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 13%|█▎        | 26/200 [00:13<01:29,  1.95it/s]\n",
      " 10%|█         | 21/200 [00:10<01:32,  1.93it/s]\n",
      " 10%|█         | 20/200 [00:10<01:33,  1.93it/s]\n",
      "  7%|▋         | 14/200 [00:07<01:38,  1.89it/s]\n",
      " 18%|█▊        | 37/200 [00:18<01:22,  1.97it/s]\n",
      "  6%|▌         | 11/200 [00:05<01:41,  1.85it/s]\n",
      " 10%|█         | 21/200 [00:10<01:32,  1.93it/s]\n",
      "  9%|▉         | 18/200 [00:09<01:34,  1.92it/s]\n",
      " 18%|█▊        | 37/200 [00:19<01:24,  1.93it/s]\n",
      " 12%|█▎        | 25/200 [00:14<01:39,  1.76it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ob1 roc: 0.509916 0.50418 0.515242\n",
      "sb1 roc: 0.503487 0.500524 0.507966\n",
      "sr roc: 0.504294 0.501113 0.508562\n",
      "sb2 roc: 0.507043 0.503924 0.511684\n",
      "ob2 roc: 0.505472 0.500868 0.510973\n",
      "\n",
      "Analyzing features [2] only...\n",
      "Analyzing band ob1 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 17%|█▋        | 34/200 [00:54<04:25,  1.60s/it]\n",
      " 12%|█▏        | 24/200 [00:38<04:42,  1.60s/it]\n",
      " 10%|█         | 20/200 [00:32<04:50,  1.61s/it]\n",
      " 16%|█▌        | 31/200 [00:49<04:28,  1.59s/it]\n",
      " 14%|█▍        | 29/200 [00:46<04:32,  1.59s/it]\n",
      " 14%|█▍        | 29/200 [00:48<04:44,  1.66s/it]\n",
      "  7%|▋         | 14/200 [00:23<05:05,  1.64s/it]\n",
      " 12%|█▏        | 24/200 [00:40<04:56,  1.68s/it]\n",
      " 12%|█▏        | 23/200 [00:37<04:50,  1.64s/it]\n",
      " 14%|█▍        | 28/200 [00:50<05:10,  1.81s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sb1 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 12%|█▏        | 24/200 [00:33<04:06,  1.40s/it]\n",
      "  8%|▊         | 16/200 [00:21<04:03,  1.32s/it]\n",
      "  8%|▊         | 17/200 [00:21<03:50,  1.26s/it]\n",
      " 22%|██▎       | 45/200 [00:58<03:20,  1.29s/it]\n",
      "  6%|▌         | 11/200 [00:16<04:37,  1.47s/it]\n",
      " 10%|█         | 20/200 [00:24<03:42,  1.24s/it]\n",
      " 14%|█▍        | 28/200 [00:34<03:30,  1.23s/it]\n",
      " 12%|█▏        | 24/200 [00:29<03:36,  1.23s/it]\n",
      " 17%|█▋        | 34/200 [00:43<03:33,  1.28s/it]\n",
      " 14%|█▍        | 29/200 [00:36<03:38,  1.28s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sr ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  5%|▌         | 10/200 [00:09<03:04,  1.03it/s]\n",
      "  8%|▊         | 17/200 [00:15<02:51,  1.07it/s]\n",
      " 16%|█▌        | 32/200 [00:30<02:38,  1.06it/s]\n",
      "  7%|▋         | 14/200 [00:13<02:58,  1.04it/s]\n",
      " 26%|██▋       | 53/200 [00:49<02:18,  1.06it/s]\n",
      " 15%|█▌        | 30/200 [00:29<02:47,  1.02it/s]\n",
      " 12%|█▏        | 24/200 [00:26<03:11,  1.09s/it]\n",
      " 14%|█▍        | 28/200 [00:27<02:51,  1.00it/s]\n",
      "  8%|▊         | 16/200 [00:16<03:13,  1.05s/it]\n",
      " 14%|█▍        | 28/200 [00:27<02:45,  1.04it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sb2 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 12%|█▎        | 25/200 [00:18<02:11,  1.33it/s]\n",
      " 14%|█▎        | 27/200 [00:18<01:58,  1.45it/s]\n",
      " 10%|▉         | 19/200 [00:15<02:27,  1.23it/s]\n",
      "  8%|▊         | 17/200 [00:11<02:07,  1.43it/s]\n",
      " 13%|█▎        | 26/200 [00:17<01:59,  1.46it/s]\n",
      "  7%|▋         | 14/200 [00:09<02:10,  1.42it/s]\n",
      "  7%|▋         | 14/200 [00:09<02:12,  1.41it/s]\n",
      "  5%|▌         | 10/200 [00:07<02:25,  1.30it/s]\n",
      "  9%|▉         | 18/200 [00:13<02:19,  1.31it/s]\n",
      "  6%|▋         | 13/200 [00:09<02:20,  1.33it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band ob2 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  5%|▌         | 10/200 [00:05<01:46,  1.78it/s]\n",
      " 15%|█▌        | 30/200 [00:15<01:28,  1.91it/s]\n",
      "  8%|▊         | 17/200 [00:08<01:34,  1.93it/s]\n",
      "  8%|▊         | 17/200 [00:08<01:34,  1.93it/s]\n",
      "  9%|▉         | 18/200 [00:09<01:34,  1.93it/s]\n",
      " 11%|█         | 22/200 [00:11<01:31,  1.95it/s]\n",
      "  8%|▊         | 16/200 [00:08<01:35,  1.92it/s]\n",
      "  6%|▌         | 12/200 [00:06<01:39,  1.88it/s]\n",
      "  7%|▋         | 14/200 [00:08<01:50,  1.69it/s]\n",
      " 18%|█▊        | 37/200 [00:18<01:21,  2.00it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ob1 roc: 0.508869 0.503444 0.514112\n",
      "sb1 roc: 0.502606 0.500036 0.507053\n",
      "sr roc: 0.503714 0.501093 0.508489\n",
      "sb2 roc: 0.506131 0.500634 0.511964\n",
      "ob2 roc: 0.505676 0.502393 0.509834\n",
      "\n",
      "Analyzing features [3] only...\n",
      "Analyzing band ob1 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 10%|█         | 21/200 [00:39<05:40,  1.90s/it]\n",
      " 22%|██▎       | 45/200 [01:20<04:37,  1.79s/it]\n",
      "  6%|▋         | 13/200 [00:22<05:17,  1.70s/it]\n",
      "  6%|▌         | 12/200 [00:23<06:14,  1.99s/it]\n",
      " 10%|▉         | 19/200 [00:31<05:03,  1.68s/it]\n",
      " 10%|█         | 20/200 [00:36<05:29,  1.83s/it]\n",
      " 12%|█▏        | 24/200 [00:42<05:08,  1.75s/it]\n",
      " 10%|█         | 21/200 [00:37<05:18,  1.78s/it]\n",
      " 10%|█         | 20/200 [00:33<04:59,  1.66s/it]\n",
      "  8%|▊         | 16/200 [00:26<05:08,  1.67s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sb1 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  9%|▉         | 18/200 [00:22<03:50,  1.27s/it]\n",
      " 10%|▉         | 19/200 [00:26<04:16,  1.42s/it]\n",
      " 18%|█▊        | 35/200 [00:43<03:26,  1.25s/it]\n",
      " 10%|▉         | 19/200 [00:24<03:54,  1.29s/it]\n",
      " 16%|█▋        | 33/200 [00:40<03:24,  1.22s/it]\n",
      "  8%|▊         | 16/200 [00:19<03:48,  1.24s/it]\n",
      " 20%|██        | 40/200 [00:51<03:25,  1.28s/it]\n",
      " 13%|█▎        | 26/200 [00:33<03:47,  1.31s/it]\n",
      "  8%|▊         | 15/200 [00:22<04:37,  1.50s/it]\n",
      "  8%|▊         | 15/200 [00:25<05:11,  1.69s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sr ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 12%|█▏        | 23/200 [00:28<03:42,  1.26s/it]\n",
      " 14%|█▍        | 28/200 [00:35<03:35,  1.25s/it]\n",
      "  8%|▊         | 17/200 [00:21<03:53,  1.28s/it]\n",
      "  6%|▌         | 11/200 [00:14<04:02,  1.28s/it]\n",
      " 10%|█         | 21/200 [00:26<03:42,  1.24s/it]\n",
      "  6%|▋         | 13/200 [00:16<04:00,  1.29s/it]\n",
      "  8%|▊         | 16/200 [00:20<03:54,  1.27s/it]\n",
      "  6%|▋         | 13/200 [00:17<04:08,  1.33s/it]\n",
      "  7%|▋         | 14/200 [00:17<03:54,  1.26s/it]\n",
      " 10%|█         | 21/200 [00:26<03:44,  1.26s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sb2 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 14%|█▍        | 28/200 [00:26<02:42,  1.06it/s]\n",
      "  8%|▊         | 15/200 [00:15<03:15,  1.06s/it]\n",
      "  8%|▊         | 16/200 [00:15<02:59,  1.02it/s]\n",
      "  7%|▋         | 14/200 [00:13<03:05,  1.00it/s]\n",
      " 13%|█▎        | 26/200 [00:25<02:48,  1.03it/s]\n",
      "  6%|▋         | 13/200 [00:12<03:01,  1.03it/s]\n",
      " 12%|█▏        | 23/200 [00:21<02:48,  1.05it/s]\n",
      "  9%|▉         | 18/200 [00:18<03:03,  1.01s/it]\n",
      " 14%|█▍        | 28/200 [00:28<02:54,  1.01s/it]\n",
      " 12%|█▎        | 25/200 [00:24<02:51,  1.02it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band ob2 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 10%|█         | 20/200 [00:14<02:09,  1.39it/s]\n",
      " 14%|█▎        | 27/200 [00:19<02:06,  1.37it/s]\n",
      " 10%|█         | 21/200 [00:15<02:13,  1.34it/s]\n",
      "  8%|▊         | 17/200 [00:11<02:08,  1.42it/s]\n",
      "  9%|▉         | 18/200 [00:12<02:05,  1.45it/s]\n",
      " 10%|█         | 20/200 [00:14<02:11,  1.37it/s]\n",
      "  6%|▋         | 13/200 [00:09<02:09,  1.44it/s]\n",
      " 24%|██▍       | 48/200 [00:33<01:45,  1.44it/s]\n",
      "  5%|▌         | 10/200 [00:07<02:26,  1.30it/s]\n",
      "  6%|▌         | 12/200 [00:08<02:13,  1.41it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ob1 roc: 0.504344 0.500896 0.511542\n",
      "sb1 roc: 0.503165 0.500939 0.505907\n",
      "sr roc: 0.503742 0.500821 0.507194\n",
      "sb2 roc: 0.506715 0.502042 0.513225\n",
      "ob2 roc: 0.506774 0.500595 0.513528\n",
      "\n",
      "Analyzing features [4] only...\n",
      "Analyzing band ob1 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 13%|█▎        | 26/200 [00:55<06:08,  2.12s/it]\n",
      "  6%|▌         | 12/200 [00:26<07:02,  2.25s/it]\n",
      " 11%|█         | 22/200 [00:50<06:49,  2.30s/it]\n",
      " 11%|█         | 22/200 [00:50<06:45,  2.28s/it]\n",
      " 10%|█         | 21/200 [00:47<06:46,  2.27s/it]\n",
      "  7%|▋         | 14/200 [00:31<06:52,  2.22s/it]\n",
      " 14%|█▎        | 27/200 [01:00<06:28,  2.25s/it]\n",
      " 12%|█▏        | 24/200 [00:56<06:52,  2.35s/it]\n",
      " 12%|█▏        | 24/200 [00:52<06:23,  2.18s/it]\n",
      "  9%|▉         | 18/200 [00:40<06:54,  2.28s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sb1 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 10%|█         | 20/200 [00:34<05:09,  1.72s/it]\n",
      " 13%|█▎        | 26/200 [00:44<05:00,  1.73s/it]\n",
      "  6%|▌         | 12/200 [00:21<05:40,  1.81s/it]\n",
      " 14%|█▍        | 29/200 [00:48<04:48,  1.69s/it]\n",
      "  8%|▊         | 15/200 [00:27<05:35,  1.81s/it]\n",
      " 18%|█▊        | 35/200 [01:03<04:59,  1.82s/it]\n",
      "  6%|▌         | 12/200 [00:23<06:11,  1.97s/it]\n",
      "  6%|▌         | 12/200 [00:23<06:05,  1.95s/it]\n",
      " 16%|█▌        | 31/200 [00:54<04:58,  1.77s/it]\n",
      " 12%|█▏        | 24/200 [00:46<05:44,  1.96s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sr ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 10%|█         | 20/200 [00:29<04:29,  1.50s/it]\n",
      "  8%|▊         | 17/200 [00:24<04:27,  1.46s/it]\n",
      " 14%|█▍        | 28/200 [00:38<03:57,  1.38s/it]\n",
      "  8%|▊         | 15/200 [00:21<04:19,  1.40s/it]\n",
      "  8%|▊         | 17/200 [00:23<04:17,  1.41s/it]\n",
      " 10%|▉         | 19/200 [00:26<04:10,  1.38s/it]\n",
      " 10%|▉         | 19/200 [00:27<04:20,  1.44s/it]\n",
      "  6%|▌         | 12/200 [00:17<04:34,  1.46s/it]\n",
      " 12%|█▎        | 25/200 [00:35<04:09,  1.43s/it]\n",
      "  6%|▌         | 12/200 [00:17<04:41,  1.50s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band sb2 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 20%|██        | 40/200 [00:40<02:43,  1.02s/it]\n",
      "  6%|▋         | 13/200 [00:13<03:18,  1.06s/it]\n",
      " 11%|█         | 22/200 [00:22<02:58,  1.00s/it]\n",
      " 16%|█▌        | 32/200 [00:29<02:35,  1.08it/s]\n",
      "  7%|▋         | 14/200 [00:13<02:53,  1.07it/s]\n",
      " 12%|█▏        | 24/200 [00:22<02:47,  1.05it/s]\n",
      "  8%|▊         | 16/200 [00:15<02:59,  1.02it/s]\n",
      "  6%|▋         | 13/200 [00:12<02:54,  1.07it/s]\n",
      "  7%|▋         | 14/200 [00:14<03:06,  1.00s/it]\n",
      "  7%|▋         | 14/200 [00:14<03:07,  1.01s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing band ob2 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  8%|▊         | 15/200 [00:11<02:16,  1.35it/s]\n",
      "  6%|▋         | 13/200 [00:10<02:27,  1.27it/s]\n",
      " 16%|█▌        | 31/200 [00:23<02:07,  1.33it/s]\n",
      "  6%|▌         | 11/200 [00:08<02:23,  1.31it/s]\n",
      " 10%|█         | 21/200 [00:14<02:03,  1.45it/s]\n",
      " 18%|█▊        | 37/200 [00:25<01:51,  1.46it/s]\n",
      " 27%|██▋       | 54/200 [00:35<01:36,  1.51it/s]\n",
      " 16%|█▋        | 33/200 [00:22<01:55,  1.45it/s]\n",
      " 20%|██        | 41/200 [00:27<01:46,  1.49it/s]\n",
      "  6%|▋         | 13/200 [00:09<02:13,  1.40it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ob1 roc: 0.506267 0.502895 0.509271\n",
      "sb1 roc: 0.503531 0.501513 0.506467\n",
      "sr roc: 0.50383 0.500592 0.507653\n",
      "sb2 roc: 0.502924 0.500122 0.507925\n",
      "ob2 roc: 0.507943 0.501226 0.514276\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "n_repeat = 10\n",
    "\n",
    "\n",
    "\n",
    "for i in range(n_features):\n",
    "    feats_to_choose = [i]\n",
    "    print(\"Analyzing features\", feats_to_choose, \"only...\")\n",
    "    band_rocs = {band:[] for band in list(bands_dict.keys())}\n",
    "    for band in bands_dict.keys():\n",
    "        print(\"Analyzing band\", band, \"...\")\n",
    "        for j in range(n_repeat):\n",
    "            roc = analyze_band_transform(band, transformed_sim_samples[band][:,feats_to_choose], dat_samples[band][:,feats_to_choose], len(feats_to_choose), epochs_NN, batch_size_NN, lr_NN, patience_NN, device)\n",
    "            band_rocs[band].append(roc)\n",
    "    for band in bands_dict.keys():\n",
    "        mean, lower, upper = get_bootstrapped_AUC(band_rocs[band], lower_percentile = 5, upper_percentile = 95, num_decimals = 6)\n",
    "        print(band, \"roc:\", mean, lower, upper)\n",
    "    print()\n",
    "        \n",
    "        "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b9081168",
   "metadata": {},
   "source": [
    "## Analyze all but 1 feature"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ed8ea80a",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Analyzing features [1, 2, 3, 4] only...\n",
      "Analyzing band ob1 ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 14%|█▍        | 29/200 [01:00<05:57,  2.09s/it]"
     ]
    }
   ],
   "source": [
    "n_repeat = 10\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "for i in range(n_features):\n",
    "    feats_to_choose = [0,1,2,3,4]\n",
    "    feats_to_choose.remove(i)\n",
    "    print(\"Analyzing features\", feats_to_choose, \"only...\")\n",
    "    band_rocs = {band:[] for band in list(bands_dict.keys())}\n",
    "    for band in bands_dict.keys():\n",
    "        print(\"Analyzing band\", band, \"...\")\n",
    "        for j in range(n_repeat):\n",
    "            roc = analyze_band_transform(band, transformed_sim_samples[band][:,feats_to_choose], dat_samples[band][:,feats_to_choose], len(feats_to_choose), epochs_NN, batch_size_NN, lr_NN, patience_NN, device)\n",
    "            band_rocs[band].append(roc)\n",
    "    for band in bands_dict.keys():\n",
    "        mean, lower, upper = get_bootstrapped_AUC(band_rocs[band], lower_percentile = 5, upper_percentile = 95, num_decimals = 6)\n",
    "        print(band, \"roc:\", mean, lower, upper)\n",
    "    print()\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9b62e565",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "95225cfb",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "50c6b5e9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "66594a94",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
