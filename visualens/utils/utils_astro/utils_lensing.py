import numpy as np
import os
import sys
from astropy import units as u
from astropy import constants as astro_constants
#sys.path.append( os.path.join(os.path.expanduser("~"), 'Library/Mobile Documents/com~apple~CloudDocs/RESEARCH/PROCESS/') )
from ...utils.utils_astro.get_default_cosmology import get_default_cosmology
cosmo = get_default_cosmology()



def get_dslds(zl, zs) :
    Ds  = cosmo.angular_diameter_distance_z1z2(0 , zs)
    Dls = cosmo.angular_diameter_distance_z1z2(zl, zs)
    return Dls/Ds

def get_DsDlDls(zl, zs) :
    Dl  = cosmo.angular_diameter_distance_z1z2(0 , zl)
    Ds  = cosmo.angular_diameter_distance_z1z2(0 , zs)
    Dls = cosmo.angular_diameter_distance_z1z2(zl, zs)
    return Ds/Dl/Dls


def q2ellipticity(q) :
    epsilon = (1 - q**2) / (1 + q**2)
    return epsilon

def ellipticity2q(epsilon) :
    q = np.sqrt((1 - epsilon) / (1 + epsilon))
    return q


def SigmaCrit(zl, zs) :
    #print("Using cosmology: ", cosmo)
    DsDlDls = get_DsDlDls(zl, zs)
    return ( astro_constants.c**2 / (4*np.pi*astro_constants.G) * DsDlDls ).to(u.kg/u.m**2)

def sigma0LENSTRONOMY_to_vdispLENSTOOL(sigma_0, rcore, rcut, zl, zs) :
    """
    Function to convert from dimensionless sigma0 to vdisp in km/s
    Args:
        sigma_0: dimensionless sigma0
    Returns:
        vdisp: vdisp in km/s
    """
    Sigma_0 = sigma_0 * SigmaCrit(zl, zs)
    Dl = cosmo.angular_diameter_distance_z1z2(0 , zl)
    if type(rcore) != u.Quantity or getattr(rcore, 'unit', None) == u.arcsec :
        rcore = rcore * u.arcsec if type(rcore) != u.Quantity else rcore
        rcore = rcore.to(u.rad).value
        rcore = rcore * Dl.to(u.km)
    if type(rcut) != u.Quantity or getattr(rcut, 'unit', None) == u.arcsec :
        rcut = rcut * u.arcsec if type(rcut) != u.Quantity else rcut
        rcut = rcut.to(u.rad).value
        rcut = rcut * Dl.to(u.km)
    vdisp2 = 4/3 * astro_constants.G * Sigma_0 * (rcore * rcut**2) / (rcut**2 - rcore**2)
    vdisp2 = vdisp2.to(u.km**2/u.s**2)
    return np.sqrt(vdisp2)


def Sigma(R, rcore, rcut, vdisp) :
    return 3/(4*astro_constants.G) * (rcore + rcut) / rcut * (1/np.sqrt(rcore**2 + R**2) - 1/np.sqrt(rcut**2 + R**2)) * vdisp**2

def Mtot(rcore, rcut, vdisp) :
    return 3*np.pi/(2*astro_constants.G) * (rcut**2 - rcore**2) / rcut * vdisp**2

def vdisp(rcore, rcut, Mtot) :
    return np.sqrt(2*astro_constants.G * Mtot * rcut / (3*np.pi * (rcut**2 - rcore**2)))
