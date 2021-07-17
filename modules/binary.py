
import warnings
from itertools import product
import numpy as np
from modules import isochHandle, clustHandle
from .HARDCODED import cmd_systs, idx_header


def ID(
    cluster, gaia_ID, met_l, age_l, ext_l, dist_l, Nvals, thresh_binar,
        binar_P_thresh, splitmethod):
    """
    Assign probabilities for each star of being a single or binary system.
    """

    envelope = []
    if splitmethod == "envelope":
        # Split single/binary block using 'thresh'
        envelope, single_msk, binar_msk = clustHandle.splitEnv(
            cluster, thresh_binar)

    ext_l = np.linspace(ext_l[0], ext_l[1], Nvals)
    dist_l = np.linspace(dist_l[0], dist_l[1], Nvals)

    params = list(product(*(met_l, age_l, ext_l, dist_l)))
    Ntot = len(params)
    binar_fr_all, single_systs, single_masses_all = [], [], []
    for pi, (met, age, ext, dist) in enumerate(params):
        print("{}/{}. {}, {}, {}, {}".format(
            pi + 1, Ntot, met, age, ext, dist))

        # Read and process (met, age) isochrone
        turn_off, isoch_phot, mass_ini =\
            isochHandle.isochProcess(
                gaia_ID, cmd_systs, idx_header, met, age, ext, dist)[:-1]

        if splitmethod != "envelope":
            # Split single/binary block using 'thresh'
            single_msk, binar_msk = clustHandle.split(
                cluster, isoch_phot, thresh_binar, turn_off)

        # Estimate the *observed* single systems' mass
        single_masses = clustHandle.singleMasses(
            cluster, mass_ini, isoch_phot, single_msk)

        single_systs.append(single_msk.astype(int))
        single_masses_all.append(single_masses)

        b_fr = binar_msk.sum() / cluster.shape[-1]
        binar_fr_all.append(b_fr)

    # Estimate the binary probability as 1 minus the average number of times
    # that a system was identified as a single star
    if splitmethod != "envelope":
        binar_probs = 1. - np.mean(single_systs, 0)
        # Identify as binaries those systems with a binary probability larger
        # than 'binar_P_thresh
        binar_msk = binar_probs > binar_P_thresh
        single_msk = ~binar_msk
    else:
        binar_probs = 1. - single_msk

    # Assign the mean of the estimated single masses to those stars identified
    # as such
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        single_masses_mean = np.nanmean(single_masses_all, 0)
    single_masses = single_masses_mean[single_msk]

    return envelope, single_msk, binar_msk, binar_probs, single_masses,\
        binar_fr_all


def masses(cluster, gaia_ID, best_pars, binar_msk):
    """
    Estimate the *observed* binary systems' mass
    """
    met, age, ext, dist = best_pars
    binar_systs = cluster[:, binar_msk].T

    # Read and process (met, age) isochrone
    isoch_phot_best, mass_ini, isoch_col_mags =\
        isochHandle.isochProcess(
            gaia_ID, cmd_systs, idx_header, met, age, ext, dist)[1:]
    m1_mass, m2_mass, res = clustHandle.binarMasses(
        isoch_phot_best, isoch_col_mags, mass_ini, cluster, binar_systs)

    return isoch_phot_best, m1_mass, m2_mass
