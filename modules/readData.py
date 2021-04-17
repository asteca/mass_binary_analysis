
import configparser
from astropy.io import ascii
import numpy as np


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


def loadClust(clust_name, max_error=0.3):
    """
    Column names are HARDCODED
    """
    cluster = ascii.read("input/{}.dat".format(clust_name))
    print("{} total stars".format(len(cluster)))
    msk = cluster['BP-RP'].mask
    msk2 = (cluster['e_Gmag'] >= max_error) | (cluster['e_BP-RP'] >= max_error)
    msk = msk | msk2.data
    print("{} masked stars".format(msk.sum()))
    cluster = cluster[~msk]
    cluster = np.array([cluster['Gmag'], cluster['BP-RP']])
    print("{} final members".format(cluster.shape[-1]))
    return cluster


def loadIsoch(idx_header, met, age):
    """
    HARDCODED to work with Gaia data <-- IMPORTANT
    """
    # Load CMD isochrone (remove '#' from columns' names) IMPORTANT
    isoch = ascii.read("./isochs/{}.dat".format(
        str(met).replace(".", "_")), header_start=idx_header)

    msk = isoch['logAge'] == age
    isoch = isoch[msk]

    # The label '2' marks the end of the MS, i.e. the turn-off point
    idx = np.searchsorted(isoch['label'], 2)
    turn_off = isoch['Gmag'][idx]

    return isoch, turn_off
