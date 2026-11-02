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

    center, amp_nar, sigma_nar, cont = theta 
        
    narrow = amp_nar * np.exp(-0.5 * np.power(vel - center,2) / sigma_nar**2)
    
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
    return lp + narrow_ha_log_likelihood(theta, vel, flux, flux_err)

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
    
    ## make our narrow gaussian to plot 
    narrow_model = n_amp_50 * np.exp(-0.5 * ((vel - cen_50) / n_sig_50)**2) + cont_50

    fig, ax = plt.subplots(figsize = [8,5])
    ax.plot(vel, flux, drawstyle = "steps-mid", color = "black")
    ax.fill_between(vel, flux - flux_err, flux + flux_err, color = "lightgray", step='mid')
    ax.plot(vel, narrow_model, color = "blue", label = r" $\mathrm{H\alpha}$ Narrow", linestyle = "--")
    ax.set_xlabel(r"Velocity [km/s]", fontsize = 16)
    ax.set_ylabel(r'$f_{\lambda}$ [10$^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]', fontsize = 16)
    ax.legend(frameon = False, fontsize = 14)
    ax.set_xlim(-5000,5000)

    return fig


def narrow_fit_ha(wv, flux, flux_err, z, spectra_name, save_direc, 
                  *, scale_factor = 1e18, fit_window=5000, plot_chains = False, plot_spec = False):
    
    print("fitting the halpha line...")
    print("---------------------------")

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
        
        full_path = f"{save_direc}/{spectra_name}_narrow_ha_fit.csv"
        data_to_save = pd.DataFrame({   "Center": [np.nan],
                                        "Narrow Amp" : [np.nan],
                                        "Narrow Sigma": [np.nan], 
                                        "Continuum" : [np.nan] })
        data_to_save.to_csv(full_path)

        

    else:
        ## getting rid of units so the fitter doesn't get mad at us 
        flux = flux.value
        flux_err = flux_err.value 

        ## fit parameters
        labels = ["center", "amp nar", r"$\sigma_{nar}$", "cont"]

        ## get initial sigma_nar as 2 * velocity pixel spacing
        ## so initial sigma_nar will change with redshift 

        sigma_nar = (vel[1] - vel[0]) * 2
        
        pos = [0, .7*np.nanmax(flux),  sigma_nar, 0]  \
        + 1e-4 * np.random.rand(16, len(labels))

        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(
                nwalkers, ndim, narrow_ha_log_probability, args=(vel, flux, flux_err
                                                        )
            )
        sampler.run_mcmc(pos, 10000, progress = False, store = True)


        ## get the chains from the emcee fit 
        samples = sampler.get_chain()
        flat_samples_ha = sampler.get_chain(discard= 2500, thin=10, flat=True)

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
                    flat_samples_ha, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles = True, plot_datapoints=False
                    )

        if plot_spec == True:
            print(f"plotting fit...")
            fig = plot_narrow_emission(vel, flux, flux_err, flat_samples_ha)
            fig_path = f"{save_direc}/{spectra_name}_narrow_ha_fit.png"
            fig.savefig(fig_path, bbox_inches = "tight", dpi=300)
            
        ## save halpha fit, yay!

        print(f"saving halpha fit...")
        full_path = f"{save_direc}/{spectra_name}_narrow_ha_fit.csv"
        data_to_save = pd.DataFrame({"Center": flat_samples_ha[:,0], 
                                        "Narrow Amp" : flat_samples_ha[:,1],
                                        "Narrow Sigma": flat_samples_ha[:,2], 
                                        "Continuum" : flat_samples_ha[:,3]})
        data_to_save.to_csv(full_path)



def broad_ha_log_prior(theta, vel,
                 amp_bounds = (0, 20), cont_bounds = (0, 2),
                 *, broad_lower_bound = 250):

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
    
    center, broad_center, amp_nar, amp_broad, sigma_nar, sigma_broad, amp_6585, cont = theta  

    narrow_cen_bounds = (-50, 50)
    cen_bounds = (-250, 250)
    sigma_nar_bounds = (0, (700/2.35))
    sigma_broad_bounds = ((broad_lower_bound/2.35), (7000/2.35))
    
    if ((narrow_cen_bounds[0] < center < narrow_cen_bounds[1]) and
        (cen_bounds[0] < broad_center < cen_bounds[1]) and
        (amp_bounds[0] < amp_nar < amp_bounds[1]) and 
        (amp_bounds[0] < amp_broad < amp_bounds[1]) and 
        (amp_bounds[0] < amp_6585 < amp_bounds[1]) and
        (sigma_nar_bounds[0] < sigma_nar < sigma_nar_bounds[1]) and 
        (sigma_broad_bounds[0] < sigma_broad < sigma_broad_bounds[1]) and 
        (sigma_broad > sigma_nar) and 
        (cont_bounds[0] < cont < cont_bounds[1])):
        return 0.0
    return -np.inf


def broad_ha_log_likelihood(theta, vel, flux, flux_err):
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
    center, broad_center, amp_nar, amp_broad, sigma_nar, sigma_broad, amp_6585, cont = theta

    ## figure out where the NII line is in velocity space


    nII_line = .65834 ## microns 
    line_diff = nII_line - ha_line
    
    ## find dopler shift of nII_line 
    nII6585_vel = (3e5) * (line_diff / ha_line)
    
    nII6550_line = 0.65481  ## microns
    line_diff = (nII6550_line - ha_line)
    nII6550_vel = (3e5) * (line_diff / ha_line)
   
        
    narrow = amp_nar * np.exp(-0.5 * np.power(vel - center,2) / sigma_nar**2)
    broad = amp_broad * np.exp(-0.5 * np.power(vel - broad_center,2) / sigma_broad**2)

    nII_6585 = amp_6585*np.exp(-0.5*np.power(vel - (center + nII6585_vel),2)/sigma_nar**2)
    ## blue nitrogen line is fixed to red line
    nII_6550 = (amp_6585/2.94)*np.exp(-0.5*np.power(vel -(center + nII6550_vel),2)/sigma_nar**2)

    ## total model for narrow and broad halpha, including blended NII lines 
    model = narrow + nII_6585 +  nII_6550 + broad + cont
    
    return -0.5 * np.nansum((flux - model)**2 / (flux_err)**2)

def broad_ha_log_probability(theta, vel, flux, flux_err): 
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

    lp = broad_ha_log_prior(theta, vel)
    if not np.isfinite(lp):
        return -np.inf
    return lp + broad_ha_log_likelihood(theta, vel, flux, flux_err,)

def plot_broad_emission(vel, flux, flux_err, flat_samples):
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
    b_cen_16, b_cen_50, b_cen_84 = np.percentile(flat_samples[:,1], [16, 50, 84]) 
    n_amp_16, n_amp_50, n_amp_84 = np.percentile(flat_samples[:,2], [16, 50, 84]) 
    b_amp_16, b_amp_50, b_amp_84 = np.percentile(flat_samples[:,3], [16, 50, 84]) 
    n_sig_16, n_sig_50, n_sig_84 = np.percentile(flat_samples[:,4], [16, 50, 84])
    b_sig_16, b_sig_50, b_sig_84 = np.percentile(flat_samples[:,5], [16, 50, 84])
    n6585_amp_16, n6585_amp_50, n6585_amp_84 = np.percentile(flat_samples[:,6], [16, 50, 84]) 
    cont_16, cont_50, cont_84 = np.percentile(flat_samples[:,7], [16, 50, 84]) 

    nII_line = .65834 ## microns 
    line_diff = nII_line - ha_line
    
    ## find dopler shift of nII_line 
    nII6585_vel = (3e5) * (line_diff / ha_line)
    
    nII6550_line = 0.65481  ## microns
    line_diff = (nII6550_line - ha_line)
    nII6550_vel = (3e5) * (line_diff / ha_line)
       

    narrow_model = n_amp_50 * np.exp(-0.5 * ((vel- cen_50) / n_sig_50)**2) + cont_50
    nii_6585_model = n6585_amp_50*np.exp(-0.5*np.power(vel -(cen_50 + nII6585_vel),2)/n_sig_50**2)
    nii_6550_model = (n6585_amp_50/ 2.94) *np.exp(-0.5*np.power(vel -(cen_50 + nII6550_vel),2)/n_sig_50**2)

    broad_model = b_amp_50 * np.exp(-0.5 * ((vel - b_cen_50) / b_sig_50)**2) + cont_50
    nii_model = nii_6550_model + nii_6585_model + cont_50

    ## subtracting a cont_50 because both the narrow and broad model have one 
    total_model = narrow_model + broad_model + nii_6585_model + nii_6550_model - cont_50

    fig, ax = plt.subplots(figsize = [8,5])
    ax.plot(vel, flux, drawstyle = "steps-mid", color = "black")
    ax.fill_between(vel, flux - flux_err, flux + flux_err, color = "lightgray", step='mid')
    ax.plot(vel, narrow_model, color = "blue", label = r" $\mathrm{H\alpha}$ Narrow", linestyle = "--")
    ax.plot(vel, broad_model, color = "red", label = r" $\mathrm{H\alpha}$ Broad", linestyle = "--", zorder = 2)

    ax.plot(vel, total_model, color = "darkviolet", label = "Total Fit", linestyle = "--", zorder = 5)

    if n6585_amp_50 > 0.01:
        ax.plot(vel, nii_model, color = "deepskyblue", linestyle = "--", label = r"[NII]")
    ax.axvline(x = nII6550_vel, color = "gray", linestyle = "dotted")
    ax.axvline(x = nII6585_vel, color = "gray", linestyle = "dotted")

    ax.set_xlabel(r"Velocity [km/s]", fontsize = 16)
    ax.set_ylabel(r'$f_{\lambda}$ [10$^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]', fontsize = 16)
    ax.legend(frameon = False, fontsize = 14)
    ax.set_xlim(-5000,5000)

    return fig



def broad_fit_ha(wv, flux, flux_err, z, spectra_name, save_direc, 
                  *, scale_factor = 1e18, fit_window=5000, plot_chains = False, plot_spec = False):
    
    print("fitting the halpha line with a broad component...")
    print("---------------------------")

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
        
        full_path = f"{save_direc}/{spectra_name}_broad_ha_fit.csv"
        data_to_save = pd.DataFrame({   "Center": [np.nan],
                                        "Broad Center" : [np.nan],
                                        "Narrow Amp" : [np.nan],
                                        "Broad Amp" : [np.nan],
                                        "Narrow Sigma" : [np.nan], 
                                        "Broad Sigma" : [np.nan],
                                        "NII Amp" : [np.nan],
                                        "Continuum" : [np.nan] })
        data_to_save.to_csv(full_path)

        

    else:
        ## getting rid of units so the fitter doesn't get mad at us 
        flux = flux.value
        flux_err = flux_err.value 

        ## fit parameters
        labels = ["center", "broad center",
                  "amp nar", "amp broad",
                  r"$\sigma_{nar}$", r"$\sigma_{broad}$",
                  "NII amp", "cont"]

        ## get initial sigma_nar as 2 * velocity pixel spacing
        ## so initial sigma_nar will change with redshift 

        sigma_nar = (vel[1] - vel[0]) * 2
        sigma_broad = 1.2 * sigma_nar
        
        pos = [0, 0, .7*np.nanmax(flux),  .1*np.nanmax(flux),
               sigma_nar, sigma_broad, 0, 0]  \
        + 1e-10 * np.random.rand(16, len(labels))

        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(
                nwalkers, ndim, broad_ha_log_probability, args=(vel, flux, flux_err
                                                        )
            )
        sampler.run_mcmc(pos, 20000, progress = True, store = True)


        ## get the chains from the emcee fit 
        samples = sampler.get_chain()
        flat_samples_ha = sampler.get_chain(discard= 10000, thin=15, flat=True)

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
                    flat_samples_ha, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles = True, plot_datapoints=False
                    )

        if plot_spec == True:
            print(f"plotting fit...")
            fig = plot_broad_emission(vel, flux, flux_err, flat_samples_ha)
            fig_path = f"{save_direc}/{spectra_name}_broad_ha_fit.png"
            fig.savefig(fig_path, bbox_inches = "tight", dpi=300)
            
        ## save halpha fit, yay!

        print(f"saving halpha fit...")
        full_path = f"{save_direc}/{spectra_name}_broad_ha_fit.csv"
        data_to_save = pd.DataFrame({"Center": flat_samples_ha[:,0],
                                        "Broad Center" : flat_samples_ha[:,1],
                                        "Narrow Amp" : flat_samples_ha[:,2],
                                        "Broad Amp" : flat_samples_ha[:,3],
                                        "Narrow Sigma" : flat_samples_ha[:,4], 
                                        "Broad Sigma" : flat_samples_ha[:,5],
                                        "NII Amp" : flat_samples_ha[:,6],
                                        "Continuum" : flat_samples_ha[:,7] })
        data_to_save.to_csv(full_path)