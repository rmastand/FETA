import numpy as np
import matplotlib.pyplot as plt
import os 

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib


"""
Preprocessing visualizers
"""


def plot_feature_histograms(data_sim, data_dat, bins, n_features, feature_labels, sim_color = "#990000", dat_color = "#1c4587", n_plot = -1, linewidth = 2, third_plot = None):
    
    fig, ax = plt.subplots(1, n_features, figsize = (4*n_features, 5))

    for i in range(n_features):
        ax[i].hist(data_sim[:n_plot,i], bins = bins, density = True, label = "SIM", histtype = "step", color = sim_color, linewidth = linewidth)
        ax[i].hist(data_dat[:n_plot,i], bins = bins, density = True, label = "DAT", histtype = "step", color = dat_color, linewidth = linewidth)
        
        if third_plot is not None:
            ax[i].hist(third_plot[:n_plot,i], bins = bins, density = True, label = "SIG", histtype = "step", color = "green", linewidth = linewidth)
        

        ax[i].set_xlabel(feature_labels[i], fontsize = 30)
        ax[i].set_yticks([])

    ax[0].set_ylabel("Normalized Density")  
    ax[0].legend(fontsize = 20, loc = "lower center")  
    plt.tight_layout()
    
    return fig


def scatterplot_features(data_sim, data_dat, bins, n_features, feature_labels, n_plot = -1, pointsize = 1):
    
    cm = matplotlib.cm.get_cmap('Greens')

    # Scatterplot
    fig, ax = plt.subplots(1, 2, figsize = (12, 4))
    fig.subplots_adjust(right=0.8, left = 0.2)


    sc_sim = ax[0].scatter(data_sim[:n_plot,0], data_sim[:n_plot,1], c = data_sim[:n_plot,-1], cmap=cm, s = 1)
    ax[0].set_xlabel("F0")
    ax[0].set_ylabel("F1")
    ax[0].set_title("SIM")

    sc_dat = ax[1].scatter(data_dat[:n_plot,0], data_dat[:n_plot,1], c = data_dat[:n_plot,-1], cmap=cm, s = 1)
    ax[1].set_xlabel("F0")
    ax[1].set_ylabel("F1")
    ax[1].set_title("DAT")

    cbar_ax_sim = fig.add_axes([0, 0.15, 0.05, 0.7])
    cbar_ax_dat = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sc_sim, cax=cbar_ax_sim)
    fig.colorbar(sc_dat, cax=cbar_ax_dat)

    return fig





def make_loss_png(epochs, losses, val_epochs, val_losses, loss_img_path):
    
    plt.figure()
    plt.plot(epochs, losses)
    plt.plot(val_epochs, val_losses, label = "val")
    plt.xlabel("Epochs")
    plt.legend()
    plt.ylabel("Loss (learn base density)")
    plt.title(f"Last val loss: {val_losses[-1]}")
    plt.savefig(loss_img_path)
    plt.show()
    
    
def make_sim_BD_sample_plots(img_dir, sim_samples, BD_samples, binning_scheme, plot_label):
    
    pp = PdfPages(os.path.join(img_dir, "BD_samples.pdf"))

    alpha = 0.4
    n_features = sim_samples.shape[-1]

    fig, ax = plt.subplots(2, n_features, figsize = (4*(n_features), 2*4))
    for i in range(n_features):
        
        # Plots hists
        ax[0, i].hist(sim_samples[:,i], bins = binning_scheme, label = plot_label, histtype = "step", color = "purple", density = True)
        ax[0, i].hist(BD_samples[:,i], bins = binning_scheme, label = "BD", color = "purple", density = True, alpha = alpha)
        ax[0, i].set_title(f"f{i}")
        ax[0, i].legend()

        # Plot ratios
        sim_hist, _ = np.histogram(sim_samples[:,i], bins = binning_scheme, density = True)
        BD_hist, _ = np.histogram(BD_samples[:,i], bins = binning_scheme, density = True)

        BD_sim_ratio = np.divide(BD_hist, sim_hist, out=np.zeros_like(BD_hist), where=sim_hist!=0)

        bin_centers = 0.5*(binning_scheme[1:] + binning_scheme[:-1])
        width = bin_centers[1] - bin_centers[0]

        ax[1, i].step(bin_centers, BD_sim_ratio, color = "purple", where = "mid", label = "BD/SIM") 
        ax[1, i].plot(bin_centers, np.full(bin_centers.shape, 1), color = "black")
        ax[1, i].legend()
        

        
        
    fig.suptitle("SB1 + SB2")      

    #fig.show()
    pp.savefig()
    pp.close()
    
def make_SIM_transDAT_plots(img_dir, sim_samples, transformed_sim_samples, dat_samples, binning_scheme, bands_to_plot):
    
    pp = PdfPages(os.path.join(img_dir, F"SIM_transSIM_DAT_samples_{str(bands_to_plot)}.pdf"))

    alpha = 0.5
    n_features = sim_samples[bands_to_plot[0]].shape[-1]
    
    for band in bands_to_plot:

        fig, ax = plt.subplots(2, n_features, figsize = (4*n_features, 5), gridspec_kw={'height_ratios': [4, 1]})
        for i in range(n_features):

            # Plots hists
            ax[0, i].hist(sim_samples[band][:,i], bins = binning_scheme, label = "SIM", histtype = "step", color = "red", density = True)
            ax[0, i].hist(transformed_sim_samples[band][:,i], bins = binning_scheme, label = "trans(SIM)", alpha = alpha, color = "blue", density = True)
            ax[0, i].hist(dat_samples[band][:,i], bins = binning_scheme, label = "DAT", histtype = "step", color = "blue", density = True)
            ax[0, i].legend()
            ax[0, i].set_title(f"f{i}")
            
            # Plot ratios
            sim_hist, _ = np.histogram(sim_samples[band][:,i], bins = binning_scheme, density = True)
            trans_sim_hist, _ = np.histogram(transformed_sim_samples[band][:,i], bins = binning_scheme, density = True)
            dat_hist, _ = np.histogram(dat_samples[band][:,i], bins = binning_scheme, density = True)
            
            sim_ratio = np.divide(sim_hist, dat_hist, out=np.zeros_like(sim_hist), where=dat_hist!=0)
            trans_sim_ratio = np.divide(trans_sim_hist, dat_hist, out=np.zeros_like(trans_sim_hist), where=dat_hist!=0)
            
            bin_centers = 0.5*(binning_scheme[1:] + binning_scheme[:-1])
            width = bin_centers[1] - bin_centers[0]
            
            ax[1, i].step(bin_centers, sim_ratio, color = "red", where = "mid", label = "SIM/DAT") 
            ax[1, i].step(bin_centers, trans_sim_ratio, color = "blue", where = "mid", alpha = alpha)
            ax[1, i].fill_between(bin_centers, trans_sim_ratio, step="mid", alpha = alpha, label = "(trans SIM)/DAT")
            ax[1, i].plot(bin_centers, np.full(bin_centers.shape, 1), color = "black")
            ax[1, i].legend()
            
            
        fig.suptitle(band)      

        #fig.show()
        pp.savefig()

    pp.close()
    
    
def make_BD_transBD_plots(img_dir, BD_samples, transformed_BD_samples, dat_samples, binning_scheme, bands_to_plot):
    
    pp = PdfPages(os.path.join(img_dir, f"BD_transBD_DAT_samples_{str(bands_to_plot)}.pdf"))

    alpha = 0.5
    n_features = BD_samples.shape[-1]

    fig, ax = plt.subplots(2, n_features, figsize = (4*n_features, 5), gridspec_kw={'height_ratios': [4, 1]})
    for i in range(n_features):

        # Plots hists
        ax[0, i].hist(BD_samples[:,i], bins = binning_scheme, label = "BD", histtype = "step", color = "red", density = True)
        ax[0, i].hist(transformed_BD_samples[:,i], bins = binning_scheme, label = "trans BD", alpha = alpha, color = "blue", density = True)
        ax[0, i].hist(dat_samples[:,i], bins = binning_scheme, label = "DAT", histtype = "step", color = "blue", density = True)
        ax[0, i].legend()
        ax[0, i].set_title(f"f{i}")

        # Plot ratios
        BD_hist, _ = np.histogram(BD_samples[:,i], bins = binning_scheme, density = True)
        trans_BD_hist, _ = np.histogram(transformed_BD_samples[:,i], bins = binning_scheme, density = True)
        dat_hist, _ = np.histogram(dat_samples[:,i], bins = binning_scheme, density = True)

        BD_ratio = np.divide(BD_hist, dat_hist, out=np.zeros_like(BD_hist), where=dat_hist!=0)
        trans_BD_ratio = np.divide(trans_BD_hist, dat_hist, out=np.zeros_like(trans_BD_hist), where=dat_hist!=0)

        bin_centers = 0.5*(binning_scheme[1:] + binning_scheme[:-1])
        width = bin_centers[1] - bin_centers[0]

        ax[1, i].step(bin_centers, BD_ratio, color = "red", where = "mid", label = "BD/DAT") 
        ax[1, i].step(bin_centers, trans_BD_ratio, color = "blue", where = "mid", alpha = alpha)
        ax[1, i].fill_between(bin_centers, trans_BD_ratio, step="mid", alpha = alpha, label = "(trans BD)/DAT")
        ax[1, i].plot(bin_centers, np.full(bin_centers.shape, 1), color = "black")
        ax[1, i].legend()

  
        
    fig.suptitle(str(bands_to_plot))      

    #fig.show()
    pp.savefig()

    pp.close()
    
    
def make_SIM_transDAT_plots_singleton(img_dir, sim_samples, transformed_sim_samples, dat_samples, binning_scheme, bands_to_plot):
    
    pp = PdfPages(os.path.join(img_dir, "SIM_transSIM_DAT_samples.pdf"))

    alpha = 0.5
    n_features = sim_samples[bands_to_plot[0]].shape[-1]
    
    

    for band in bands_to_plot:

        fig, ax = plt.subplots(1, n_features, figsize = (4*n_features, 4), squeeze=False)
        for i in range(n_features):

            plt.hist(sim_samples[band][:,i], bins = binning_scheme, label = "SIM", histtype = "step", color = "red", density = True)
            plt.hist(transformed_sim_samples[band][:,i], bins = binning_scheme, label = "trans SIM", alpha = alpha, color = "blue", density = True)
            plt.hist(dat_samples[band][:,i], bins = binning_scheme, label = "DAT", histtype = "step", color = "blue", density = True)
            plt.legend()
            plt.title(f"f{i}")
        fig.suptitle(band)      

        #fig.show()
        pp.savefig()

    pp.close()
    
    
def make_BD_transBD_singleton(img_dir, BD_samples, transformed_BD_samples, dat_samples, binning_scheme):
    
    pp = PdfPages(os.path.join(img_dir, "BD_transBD_DAT_samples.pdf"))

    alpha = 0.5
    n_features = BD_samples.shape[-1]

    fig, ax = plt.subplots(1, n_features, figsize = (4*n_features, 4))
    for i in range(n_features):

        plt.hist(BD_samples[:,i], bins = binning_scheme, label = "BD", histtype = "step", color = "red", density = True)
        plt.hist(transformed_BD_samples[:,i], bins = binning_scheme, label = "trans BD", alpha = alpha, color = "blue", density = True)
        plt.hist(dat_samples[:,i], bins = binning_scheme, label = "DAT", histtype = "step", color = "blue", density = True)
        plt.legend()
        plt.title(f"f{i}")
    fig.suptitle("SB1 + SB2")      

    #fig.show()
    pp.savefig()

    pp.close()
