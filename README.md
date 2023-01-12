# FETA
This is the codebase for the FETA (Flow-Enhanced Transportation for Anomaly Detection) method for background construction. 

Accompanying paper: https://arxiv.org/abs/2212.11285 (authors: Tobias Golling, Samuel Klein, Radha Mastandrea, Benjamin Nachman)

For questions/comments about the code contact: rmastand@berkeley.edu

## Pipeline 

### Prepare and preprocess the data

Use the notebook ```generate_dataset_and_preprocess.ipynb```.

### Generate the flow and train it to map between simulation and data in the SB.

Use the script ```run_full_cycle_2step.py```. The script also applies the trained flow to data in the SR to generate the background template samples.

### Compare the FETA-generate background template samples with SR data.

Use the script ```final_eval_SR.py```.

### Generate all results plots

Use the notebook ```make_sig_rej_plots.ipynb```.

## Other comments

The folder ```full_cycle_scripts``` contains alternate versions of the ```run_full_cycle_2step.py``` script that were used in the supplementary optimal transport studies (https://arxiv.org/abs/2212.06155). The notebook ```OT_studies.ipynb``` contains the code to make all of the plots shown in that paper.

The notebook ```paper_plots.ipynb``` contains code to generate the "prettier" schematics. 
