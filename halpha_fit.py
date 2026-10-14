import emcee 
import corner
import numpy as np 
from astro_functions import wv_to_vel
import matplotlib.pyplot

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

def narrow_fit_hb(wv, flux, flux_err, z, spectra_name, *, scale_factor = 1e18, fit_window=5000):
    
    print("fitting the halpha line...")
    print("---------------------------")

    ## convert wavelength to velocity
    vel = wv_to_vel(wv, z, ha_line)

    ## limit vel, flux, and flux_err to +- fit_window around line 
    bounds = (vel > -1*fit_window) & (vel < 1*fit_window)
    vel = vel[bounds]
    flux = flux[bounds] * scale_factor
    flux_err = flux_err[bounds] * scale_factor


    
    
    pos = [0, .7*np.nanmax(flux),  sigma_nar, 0]  \
    + 1e-10 * np.random.rand(16, len(labels))

    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
            nwalkers, ndim, narrow_ha_log_probability, args=(vel, flux, flux_err
                                                     )
        )
    sampler.run_mcmc(pos, 25000, progress = True, store = True) #(25000)

    labels = ["center", "amp nar", r"$\sigma_{nar}$", "cont"]
    fig, axes = plt.subplots(len(labels), figsize=(10, len(labels)), sharex=True)
    samples = sampler.get_chain()