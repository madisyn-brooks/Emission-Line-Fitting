import emcee 
import corner
import numpy as np 
from astro_functions import wv_to_vel
import matplotlib.pyplot as plt 
import pandas as pd

## global constants 

ha_line = 0.6564614 # microns
hb_line = 0.4862721 # microns
o3_line = 0.5008239 # microns 

def narrow_ha_log_prior(theta, vel,
                 amp_bounds = (0, 20), cont_bounds = (0, 2)):

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
    
    center, amp_nar, sigma_nar, cont = theta 

    cen_bounds = (-500, 500)
    sigma_nar_bounds = (0, (700/2.35))
    
    if ((cen_bounds[0] < center < cen_bounds[1]) 
        and (amp_bounds[0] < amp_nar < amp_bounds[1]) and 
        (sigma_nar_bounds[0] < sigma_nar < sigma_nar_bounds[1]) and 
        (cont_bounds[0] < cont < cont_bounds[1])):
        return 0.0
    return -np.inf


def narrow_ha_log_likelihood(theta, vel, flux, flux_err):
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
        
    narrow = amp_nar * np.exp(-0.5 * np.power(vel - center,2) / sig_nar**2)
    
    model = narrow + cont
    
    return -0.5 * np.nansum((flux - model)**2 / (flux_err)**2)

def narrow_ha_log_probability(theta, vel, flux, flux_err): 
    """
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
    ---------
    prior output + likelihood output 
    
    """

    lp = narrow_ha_log_prior(theta, vel)
    if not np.isfinite(lp):
        return -np.inf
    return lp + narrow_ha_log_likelihood(theta, vel, flux, flux_err,)

def narrow_fit_ha(wv, flux, flux_err, z, spectra_name, save_direc, 
                  *, scale_factor = 1e18, fit_window=5000, plot = False):
    
    print("fitting the halpha line...")
    print("---------------------------")

    ## convert wavelength to velocity
    vel = wv_to_vel(wv, z, ha_line)

    ## limit vel, flux, and flux_err to +- fit_window around line 
    bounds = (vel > -1*fit_window) & (vel < 1*fit_window)
    vel = vel[bounds]
    flux = flux[bounds] * scale_factor
    flux_err = flux_err[bounds] * scale_factor

    ## get initial sigma_nar as 2 * velocity pixel spacing
    ## so initial sigma_nar will change with redshift 

    sigma_nar = (vel[1] - vel[0]) * 2
    
    pos = [0, .7*np.nanmax(flux),  sigma_nar, 0]  \
    + 1e-10 * np.random.rand(16, len(labels))

    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
            nwalkers, ndim, narrow_ha_log_probability, args=(vel, flux, flux_err
                                                     )
        )
    sampler.run_mcmc(pos, 20000, progress = True, store = True)

    labels = ["center", "amp nar", r"$\sigma_{nar}$", "cont"]

    ## get the chains from the emcee fit 
    samples = sampler.get_chain()
    flat_samples_ha = sampler.get_chain(discard= 10000, thin=15, flat=True)

    if plot == True:
        fig, axes = plt.subplots(len(labels), figsize=(10, len(labels)), sharex=True)

        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k.", alpha=0.3, markersize = 0.2)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            #ax.yaxis.set_label_coords(-0.1, 0.5)

            axes[-1].set_xlabel('step number');
            fig = corner.corner(
                flat_samples_ha, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles = True, plot_datapoints=False
                )

    ## save halpha fit

    data_to_save = pd.DataFrame({"Center": flat_samples_ha[:,0], 
                                    "Narrow Amp" : flat_samples_ha[:,1],
                                    "Narrow Sigma": flat_samples_ha[:,2], 
                                    "Continuum" : flat_samples_ha[:,3]})
    data_to_save.to_csv(full_path)
            