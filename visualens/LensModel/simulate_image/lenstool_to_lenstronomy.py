import numpy as np
import os
import sys
from astropy import units as u
from astropy import constants as astro_constants
from lenstronomy.LensModel.lens_model import LensModel

from ....utils.utils_astro.utils_general import world_to_relative
from ....utils.utils_astro.utils_lensing import SigmaCrit
#from ..utils.utils_astro.get_default_cosmology import get_default_cosmology
#cosmo = get_default_cosmology()


def lenstool_ellipticity_to_lenstronomy_ellipticity(epsilon) :
    q = ( (1 - epsilon) / (1 + epsilon) )**0.5
    e = (1 - q) / (1 + q)
    return e

def vdisp2Sigma0(vdisp, a, s) :
    a = a * u.kpc
    s = s * u.kpc
    vdisp = vdisp * u.km/u.s
    Sigma0 = 3*(s**2-a**2) / ( 4*astro_constants.G*a*s**2 ) * vdisp**2
    return Sigma0.to(u.kg/u.m**2)

def lenstool_to_lenstronomy(im, reference=None, z_source=2):
    lens_model_list = []
    z_lens_list = []
    kwargs_lens = []
    
    n_lens = 0
    for lens in im.lt.lt.lens:
        if lens.type == 81 and n_lens==0:  # dPIE/PIEMD
            n_lens+=1
            # Convert Lenstool ellipticity and position angle to lenstronomy e1, e2
            #e = lenstool_ellipticity_to_lenstronomy_ellipticity(lens.emass)
            e = lens.epot            
            e1 = e * np.cos(2 * lens.theta)
            e2 = e * np.sin(2 * lens.theta)

            lens_model_list.append("PJAFFE_ELLIPSE_POTENTIAL")
            
            x = lens.C.x
            y = lens.C.y
            if reference is not None :
                ra, dec = im.lt.relative_to_world(x, y)
                x, y = world_to_relative(ra, dec, reference)
            Sigma0 = vdisp2Sigma0(lens.sigma, lens.rckpc, lens.rcutkpc) #in kg/m2
            Sigma_crit = SigmaCrit(lens.z, z_source)
            sigma0 = ( Sigma0/Sigma_crit ).value
            
            kwargs_lens.append({
                "center_x": x, #in arcsec from reference point
                "center_y": y, #in arcsec from reference point
                "sigma0": sigma0,
                "Ra": lens.rc, #in arcsec
                "Rs": lens.rcut, #in arcsec
                "e1": e1,
                "e2": e2,
            })
            z_lens_list.append(lens.z)

        #elif lens.type == 12:  # NFW
        #    lens_model_list.append("NFW")
        #    kwargs_lens.append({
        #        "center_x": lens.C.x,
        #        "center_y": lens.C.y,
        #        "Rs": lens.rc,
        #        "alpha_Rs": lens.sigma,  # careful: may differ!
        #    })
        
        elif lens.type == 14 and False:
            n_lens+=1
            # external shear
            gamma = lens.emass
            
            gamma1 = gamma * np.cos(2 * lens.theta)
            gamma2 = gamma * np.sin(2 * lens.theta)
            
            lens_model_list.append("SHEAR")
            kwargs_lens.append({
                "gamma1": gamma1,
                "gamma2": gamma2,
                "ra_0": 0,
                "dec_0": 0
            })
            #if kappa != 0.0:
            #    lens_model_list.append("CONVERGENCE")
            #    kwargs_lens.append({"kappa": kappa})
            z_lens_list.append(lens.z)

        #else:
            #print(f"Warning: profile type {lens.type} not yet mapped.")

    # Create a lenstronomy LensModel instance
    print(str(n_lens) + " individual potentials found.")
    lens_model = LensModel(lens_model_list, z_lens=z_lens_list)
    return lens_model, kwargs_lens, lens_model_list, z_lens_list




