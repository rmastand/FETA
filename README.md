# FETA: Flow-Enhanced Transportation for Anomaly Detection
[![DOI](https://zenodo.org/badge/525910028.svg)](https://zenodo.org/badge/latestdoi/525910028)

This is the codebase for the FETA (Flow-Enhanced Transportation for Anomaly Detection) method for background construction. 

Accompanying paper: https://arxiv.org/abs/2212.11285 (authors: Tobias Golling, Samuel Klein, Radha Mastandrea, Benjamin Nachman)

For questions/comments about the code contact: rmastand@berkeley.edu

## Pipeline 

### Prepare and preprocess the data.

Use the notebook ```generate_dataset_and_preprocess.ipynb```.

### Generate the flow and train it to map between simulation and data in the SB.

Use the script ```run_full_cycle_2step.py```. The script also applies the trained flow to data in the SR to generate the background template samples.

### Compare the FETA-generated background template samples with SR data.

Use the script ```final_eval_SR.py```.

### Generate all results plots.

Use the notebook ```make_sig_rej_plots.ipynb```.

## Other comments

The notebook ```paper_plots.ipynb``` contains code to generate the "prettier" schematics. 

To make the scatterplots of classifier scores for the different background construction methods (Figs 10 and 11 in the main paper), run ```run_scatter_plot.py``` to process the data, then ```analyze_scatterplot.ipynb``` to make the visualizations.

# Optimal Transport studies

Accompanying report: https://arxiv.org/abs/2212.06155 (authors: Radha Mastandrea, Benjamin Nachman)

The folder ```full_cycle_scripts``` contains alternate versions of the ```run_full_cycle_2step.py``` script that were used in the supplementary optimal transport studies. The notebook ```OT_studies.ipynb``` contains the code to make all of the visualizations (from the data generated from ```run_full_cycle_xx.py```) shown in that paper.

