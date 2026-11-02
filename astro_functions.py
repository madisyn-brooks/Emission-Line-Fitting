import numpy as np 
from astropy import units as u
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo


def load_1d_spec(fits_file):
    """
    load in a jwst 1d spec file. this assumes that the data is in order of: wv, flux, err, which 
    is typical for a jwst 1d spec, and how the files are generated from MAST.
    
    Parameters
    -------
    fits_file: a jwst spec file 

    Returns
    -------
    wv: wavelength array
    flux: flux array
    flux_err: flux error array 

    """
    ## open fits file
    a = fits.open(fits_file)
    
    wv = a[1].data[a[1].header["TTYPE1"]]
    wv_unit = u.Unit(a[1].header["TUNIT1"])
    wv = wv * wv_unit

    fnu = a[1].data[a[1].header["TTYPE2"]]
    fnu_unit = u.Unit(a[1].header["TUNIT2"])

    fnu_err = a[1].data[a[1].header["TTYPE3"]]
    fnu_err_unit = u.Unit(a[1].header["TUNIT3"])

    ## convert to cgs units, from Jy or microJy to erg/s/cm^2/Hz
    try: 
        fnu = (fnu * fnu_unit).to(
            u.erg / u.s / u.cm**2 / u.Hz
        )
        fnu_err = (fnu_err * fnu_err_unit).to(
            u.erg / u.s / u.cm**2 / u.Hz
        )

    except u.UnitConversionError:
        raise ValueError(
            f"Cannot convert {fnu_unit} to flux density"
        )

    flam = fnu.to(
    u.erg / u.s / u.cm**2 / u.AA,
    equivalencies=u.spectral_density(wv)
    )
    
    flam_err = fnu_err.to(
        u.erg / u.s / u.cm**2 / u.AA,
        equivalencies=u.spectral_density(wv)
    )
    
    return wv, fnu, fnu_err, flam, flam_err

def wv_to_vel(wv, z, line_use):
    """
    Converts a wavelength array to a velocity array, given an emission/absorption restframe wavelength

    Parameters
    ----------
    wv: np.array 
        wavelength array from jwst spectra, expecting units of microns 
    z: float
        source redshift 
    line_use: float 
        rest wavelength of line of interest in microns
    
     Returns
     ---------
     velocity: np.array
        velocity array centered around line of interest, units of km/s   

    """
    line_use = line_use * u.um

    line_obs = (1+z) * line_use
    vel = (wv-line_obs)/line_obs * 3e5

    return vel 

def calculate_flux_and_luminosity(z, line_use, amp, sigma):

    """
    Calculates an emission line flux given a Gaussian distribution where the area under the curve 
    is equal to A = sqrt(2 * pi) * amp * sigma 

    Parameters:
    ---------------
    z: float
        redshift of source 
    line_use: float
        wavelength of emission line in angstroms 
    amp: np.array
        full amplitude posteior from mcmc fit 
    sigma: np.array
        full sigma posterior from mcmc fit 

    Returns:
    --------------
    flux_percentiles: np.array
        [16,50,84] percentles of integrated emission line flux 
    luminosity_percentiles: np.array
        [16,50,84] percentles of integrated emission line luminoisty 
    
    """

    lambda_use = (1 + z) * (line_use * 1e5)
    
    flux = np.sqrt(2 * np.pi) * amp * ((sigma / 3e5) * lambda_use) * u.erg / u.s / u.cm**2
    flux_percentiles = [np.percentile(flux.value, 16), np.percentile(flux.value, 50), np.percentile(flux.value, 84)]

    dist = (cosmo.luminosity_distance(z).to(u.cm))
    lum = (flux * 4 * np.pi * (dist**2)).to(u.erg / u.s)
    luminosity_percentiles = np.array([np.percentile(lum.value, 16), np.percentile(lum.value, 50), np.percentile(lum.value, 84)])

    return flux_percentiles, luminosity_percentiles
    