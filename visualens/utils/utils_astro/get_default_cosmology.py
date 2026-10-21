from astropy.cosmology import WMAP9, Planck18, FlatLambdaCDM


def get_default_cosmology(cosmo_name='FlatLambdaCDM') :
    if cosmo_name == 'Planck18' :
        cosmo = Planck18
    elif cosmo_name == 'WMAP9' :
        cosmo = WMAP9
    elif cosmo_name == 'FlatLambdaCDM' :
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    else :
        raise ValueError(f"Cosmology {cosmo_name} not found")
    return cosmo
