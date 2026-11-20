import emcee 
import corner
import numpy as np 
from astro_functions import wv_to_vel
import matplotlib.pyplot as plt 
import pandas as pd
import os 

## global constants 

ha_line = 0.6564614 # microns
hb_line = 0.4862721 # microns
o3_line = 0.5008239 # microns 

plt.rcParams.update({
"text.usetex": False,
'axes.labelsize' : 16,
'xtick.labelsize' : 14,
'ytick.labelsize' : 14,
'xtick.direction' : "in",
'ytick.direction' : "in",
'xtick.top' : True,
'ytick.right' : True,
"xtick.minor.visible" : False,
"ytick.minor.visible" : False,
"font.family" : "serif"})

def o3_log_prior(theta, amp_bounds = (0, 20), cont_bounds = (0,2)):

    """
    Parameters
    -----------
    theta: 
        vector that contains the parameters used to generate the model
    vel: np.array
        velocity array of data 
    amp_bounds: float
        prior on the emission line amplitude 
    cont_bounds: float
        prior on the continuum amplitude 

    Returns
    ---------
    0 if priors are satisified or -np.inf is theta parameters are outside pripr 
    """
    center, amp_nar, sig_nar, cont = theta 
    
    cen_bounds = (-500, 500)
    sig_nar_bounds = (0, (700/2.35))
    
    if ((cen_bounds[0] < center < cen_bounds[1]) and 
        (amp_bounds[0] < amp_nar < amp_bounds[1]) and 
        (sig_nar_bounds[0] < sig_nar < sig_nar_bounds[1]) and 
        (cont_bounds[0] < cont < cont_bounds[1])):
            return 0.0
    return -np.inf


def o3_log_likelihood(theta, velocity, flux, flux_err):

    """
        Determines how well the theta values fit to the data 
    
        Parameters
        ----------
        theta:
            vector that contains the parameters used to generate the model
        vel: np.array
            velocity array of data (km/s)
        flux: np.array
            flux array of data (flam)
        flux_err: np.array
            flux error array of data (flam)
    
    
        Returns
        ----------
        the log likelihood of the given theta parameters 
        
        """
    
    center, amp_nar, sig_nar, cont = theta 

    ## find the distance in velocity space between the 5008 line and 4960 line
    o3_5008_line = .5008 ## microns
    o3_4960_line = .4960 ## microns
    line_diff = o3_4960_line - o3_5008_line
    o3_vel_change = (3e5) * (line_diff / o3_5008_line )

    narrow_5008 = amp_nar * np.exp(-0.5 * np.power(velocity - center,2) / sig_nar**2)
    narrow_4960 = (1/2.98) * amp_nar * np.exp(-0.5 * np.power(velocity - (center + o3_vel_change) ,2) / sig_nar**2)
  
    model = narrow_5008 + narrow_4960 + cont
    
    return -0.5 * np.nansum((flux - model)**2 / (flux_err)**2)

def o3_log_probability(theta, velocity, flux, flux_err): 
    lp = o3_log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + o3_log_likelihood(theta, velocity, flux, flux_err)


def plot_narrow_emission(vel, flux, flux_err, flat_samples):
    """
    Parameters
    -----------
    vel: np.array
            velocity array of data (km/s)
    flux: np.array
        flux array of data (flam)
    flux_err: np.array
        flux error array of data (flam)
    flat_samples: np.array
        flattened mcmc samples   

    Returns
    -----------
    fig:  matplolib figure object 
    """

    ## grab fit parameters from mcmc chains
    cen_16, cen_50, cen_84 = np.percentile(flat_samples[:,0], [16, 50, 84]) 
    n_amp_16, n_amp_50, n_amp_84 = np.percentile(flat_samples[:,1], [16, 50, 84]) 
    n_sig_16, n_sig_50, n_sig_84 = np.percentile(flat_samples[:,2], [16, 50, 84])
    cont_16, cont_50, cont_84 = np.percentile(flat_samples[:,3], [16, 50, 84])

    ## find the distance in velocity space between the 5008 line and 4960 line
    o3_5008_line = .5008 ## microns
    o3_4960_line = .4960 ## microns
    line_diff = o3_4960_line - o3_5008_line
    o3_vel_change = (3e5) * (line_diff / o3_5008_line )
    
    ## make our narrow gaussian to plot 
    narrow_model_5008 = n_amp_50 * np.exp(-0.5 * ((vel - cen_50) / n_sig_50)**2) 
    narrow_model_4960 = (1/2.98) * n_amp_50 * np.exp(-0.5 * np.power(vel - (cen_50 + o3_vel_change) ,2) / n_sig_50**2)
      
    total_model = narrow_model_5008 + narrow_model_4960 + cont_50

    fig, ax = plt.subplots(figsize = [8,5])
    ax.plot(vel, flux, drawstyle = "steps-mid", color = "black")
    ax.fill_between(vel, flux - flux_err, flux + flux_err, color = "lightgray", step='mid')
    ax.plot(vel, total_model, color = "blue", label = r" $\mathrm{H\alpha}$ Narrow", linestyle = "--")
    ax.set_xlabel(r"Velocity [km/s]", fontsize = 16)
    ax.set_ylabel(r'$f_{\lambda}$ [10$^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]', fontsize = 16)
    ax.legend(frameon = False, fontsize = 14)
    ax.set_xlim(-5000,5000)


    return fig


def narrow_fit_o3(wv, flux, flux_err, z, spectra_name, save_direc, 
                  *, scale_factor = 1e18, fit_window=5000, plot_chains = False, plot_spec = False):
    
    print()
    print("fitting the [OIII] line...")
    print("---------------------------")

    labels = ["center", "amp nar", r"$\sigma_{nar}$", "cont"]

    ## convert wavelength to velocity
    vel = wv_to_vel(wv, z, ha_line)

    ## limit vel, flux, and flux_err to +- fit_window around line 
    bounds = (vel > -1*fit_window) & (vel < 1*fit_window)
    vel = vel[bounds]
    flux = flux[bounds] * scale_factor
    flux_err = flux_err[bounds] * scale_factor

    ## check if the velocity array is empty, or if the flux is all nans and return all nans for parameter values
    if len(vel) == 0 or np.all(np.isnan(flux)):
        print("Velocity array is empty or flux is all NaNs. Returning NaNs.")
        
        full_path = f"{save_direc}/{spectra_name}_narrow_o3_fit.csv"
        data_to_save = pd.DataFrame({   "Center": [np.nan],
                                        "Narrow Amp" : [np.nan],
                                        "Narrow Sigma" : [np.nan], 
                                        "Continuum" : [np.nan] })
        data_to_save.to_csv(full_path, index = False)

        

    else:
        ## getting rid of units so the fitter doesn't get mad at us 
        flux = flux.value
        flux_err = flux_err.value
        
        pos = [0, .5*np.nanmax(flux), 150, 0] + 1e-10 * np.random.rand(16, 4)
        
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
                nwalkers, ndim, o3_log_probability, args=(vel, flux, flux_err)
            )
        sampler.run_mcmc(pos, 20000, progress = True, store = True)
        
        fig, axes = plt.subplots(4, figsize=(10, 4), sharex=True)
        samples = sampler.get_chain()

        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k.", alpha=0.3, markersize = 0.2)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            #ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        
        flat_samples_o3 = sampler.get_chain(discard= 10000, thin=15, flat=True)

        if plot_chains == True:
                fig, axes = plt.subplots(len(labels), figsize=(10, len(labels)), sharex=True)
    
                for i in range(ndim):
                    ax = axes[i]
                    ax.plot(samples[:, :, i], "k.", alpha=0.3, markersize = 0.2)
                    ax.set_xlim(0, len(samples))
                    ax.set_ylabel(labels[i])
                    #ax.yaxis.set_label_coords(-0.1, 0.5)
    
                    axes[-1].set_xlabel('step number');
                    fig = corner.corner(
                        flat_samples_o3, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles = True, plot_datapoints=False
                        )
        
                if plot_spec == True:
                    print(f"plotting fit...")
                    fig = plot_narrow_emission(vel, flux, flux_err, flat_samples_o3)
                    fig_path = f"{save_direc}/{spectra_name}_broad_ha_fit.png"
                    fig.savefig(fig_path, bbox_inches = "tight", dpi=300)
        
        fig = corner.corner(
            flat_samples_o3, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles = True, plot_datapoints=False
            )
        
        plot_narrow_emission(vel, flux, flux_err, flat_samples_o3, spectra_name)
        

        folder_path = save_direc
        file_name = f"{spectra_name}_narrow_o3_fit.csv"
        full_path = os.path.join(folder_path, file_name)
        
        data_to_save = pd.DataFrame({"Center": flat_samples_o3[:,0],
                                    "Narrow Amp" : flat_samples_o3[:,1], 
                                    "Narrow Sigma": flat_samples_o3[:,2], 
                                    "Continuum" : flat_samples_o3[:,3]})
        data_to_save.to_csv(full_path)

        return flat_samples_o3