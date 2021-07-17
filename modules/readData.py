
import configparser
from astropy.io import ascii
import numpy as np
from .HARDCODED import logAge_name, label_name, mag_name, col_name


def readINI():
    """
    Read .ini config file
    """
    in_params = configparser.ConfigParser()
    in_params.read('params.ini')

    clust_pars = in_params['Cluster parameters']
    clust_name = clust_pars['name']
    gaia_ID = clust_pars['gaia_ID']
    met_l = list(map(float, clust_pars['met'].split()))
    age_l = list(map(float, clust_pars['age'].split()))
    ext_l = list(map(float, clust_pars['ext'].split()))
    dist_l = list(map(float, clust_pars['dist'].split()))
    Nvals = clust_pars.getint('Nvals')
    best_pars = list(map(float, clust_pars['best_pars'].split()))

    binar_id = in_params['Binary ID']
    thresh_binar = binar_id.getfloat('thresh_binar')
    splitmethod = binar_id.get('splitmethod')
    if splitmethod not in ("envelope", "isochs"):
        raise ValueError("splitmethod '{}' not recognized".format(splitmethod))
    binar_P_thresh = binar_id.getfloat('binar_P_thresh')

    imf_pars = in_params['IMF parameters']
    IMF_name = imf_pars.get('IMF_name')
    Max_mass = imf_pars.getfloat('Max_mass')
    N_mass_tot = imf_pars.getint('N_mass_tot')
    bins_list = list(map(int, imf_pars['bins_list'].split()))

    return clust_name, gaia_ID, met_l, age_l, ext_l, dist_l, Nvals, best_pars,\
        thresh_binar, binar_P_thresh, splitmethod, IMF_name, Max_mass,\
        N_mass_tot, bins_list


def loadClust(clust_name, max_e=0.3):
    """
    Load observed cluster
    """
    cluster = ascii.read("input/{}.dat".format(clust_name))
    print("{} total stars".format(len(cluster)))
    try:
        msk1 = cluster[mag_name].mask
    except AttributeError:
        msk1 = np.array([])
    try:
        msk2 = cluster[col_name].mask
    except AttributeError:
        msk2 = np.array([])
    # msk2 = (cluster['e_Gmag'] >= max_e) | (cluster['e_BP-RP'] >= max_e)
    if msk1.any() and msk2.any():
        msk = msk1 | msk2
    elif msk1.any() and not msk2.any():
        msk = msk1
    elif not msk1.any() and msk2.any():
        msk = msk2

    print("{} masked stars".format(msk.sum()))
    cluster = cluster[~msk]
    cluster = np.array([cluster[mag_name], cluster[col_name]])
    print("{} final members".format(cluster.shape[-1]))
    return cluster


def loadIsoch(idx_header, met, age, MS_end_ID=2):
    """
    HARDCODED to work with Gaia data

    MS_end_ID: This label marks the end of the MS, i.e. the turn-off point
    """
    # Load CMD isochrone (remove '#' from columns' names) IMPORTANT
    isoch = ascii.read("./isochs/{}.dat".format(
        str(met).replace(".", "_")), header_start=idx_header)

    msk = isoch[logAge_name] == age
    isoch = isoch[msk]

    # Identify the turn-off point
    idx = np.searchsorted(isoch[label_name], MS_end_ID)
    turn_off = isoch[mag_name][idx]

    return isoch, turn_off
