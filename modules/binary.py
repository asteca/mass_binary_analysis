
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
from scipy import stats
from modules import isochHandle, clustHandle
from astropy.stats import knuth_bin_width
from .HARDCODED import cmd_systs, idx_header

import matplotlib.pyplot as plt


def splitEnv(cluster, best_pars, perc=90):
    """

    perc: the magnitude percentile used to estimate the 'y' position of the
      envelope. A smaller value brings the envelope upwards in the CMD (i.e.:
      towards brighter magnitudes). The value of 75 is estimated heuristically
      and gives reasonable results.
    """

    theta = rotIsoch(best_pars)

    origin = (cluster[0].max(), cluster[1].max())
    cluster_rot = rotate(theta, cluster.T, origin).T

    #
    bin_edges = knuth_bin_width(
        cluster_rot[1], return_bins=True, quiet=True)[1]
    msk = bin_edges > np.percentile(cluster_rot[1], .1)
    bin_edges = bin_edges[msk]
    extra_edges = np.linspace(bin_edges[-2], bin_edges[-1], 5)
    bin_edges = list(bin_edges[:-2]) + list(extra_edges)

    lower_env_rot = []
    for i, low in enumerate(bin_edges):
        if i + 1 == len(bin_edges):
            break
        msk = (cluster_rot[1] > low) & (cluster_rot[1] <= bin_edges[i + 1])
        if msk.sum() > 0:
            mid_p = (low + bin_edges[i + 1]) * .5
            lower_env_rot.append([
                np.percentile(cluster_rot[0][msk], perc), mid_p])

    # Rotate the lower envelope to its original position
    lower_env = rotate(-theta, lower_env_rot, origin).T

    plt.subplot(121)
    plt.scatter(cluster_rot[1], cluster_rot[0], marker='.', c='r')
    mag_l, col_l = np.array(lower_env_rot).T
    plt.plot(col_l, mag_l)
    plt.gca().invert_yaxis()
    plt.subplot(122)
    plt.scatter(cluster[1], cluster[0], marker='.', c='r')
    mag_l, col_l = lower_env
    plt.plot(col_l, mag_l)
    plt.gca().invert_yaxis()
    plt.show()

    # col_step: the step used in the in color to generate the envelope. Of
    #   minor importance as the envelope is then interpolated.
    # # Obtain the lower envelope for the cluster's sequence
    # col_min, col_max = cluster_rot[1].min(), cluster_rot[1].max()
    # col_step = (col_max - col_min) / 25
    # lower_env = []
    # for low in np.arange(col_min, col_max, col_step):
    #     msk = (cluster_rot[1] > low) & (cluster_rot[1] <= low + col_step)
    #     if msk.sum() > 0:
    #         mid_p = (low + low + col_step) * .5
    #         lower_env.append([np.percentile(cluster_rot[0][msk], perc), mid_p])

    # Generate binary envelope
    mag_l, col_l = lower_env
    mag_binar = clustHandle.mag_combine(mag_l, mag_l)

    # Generate extra points
    l_envelope = isochHandle.interp(lower_env)
    b_envelope = isochHandle.interp(np.array([mag_binar, col_l]))

    plt.scatter(cluster[1], cluster[0], marker='.', c='g')
    plt.plot(l_envelope[1], l_envelope[0], 'x', ms=2, c='r')
    plt.plot(b_envelope[1], b_envelope[0], 'x', ms=2, c='r')
    plt.gca().invert_yaxis()
    plt.show()

    # Distances to the lower envelope, for all the stars
    dist_l = cdist(cluster.T, l_envelope.T)
    min_dist_l = dist_l.min(1)

    # Distances to the binary envelope, for all the stars
    dist_b = cdist(cluster.T, b_envelope.T)
    min_dist_b = dist_b.min(1)

    # If delta_d>0 then min_dist_l>min_dist_b, and the star is closer to the
    # binary sequence
    delta_d = min_dist_l - min_dist_b

    # Split systems
    binar_msk = delta_d >= 0
    single_msk = ~binar_msk

    # import matplotlib.pyplot as plt
    # plt.scatter(
    #     cluster[1][single_msk], cluster[0][single_msk], marker='.', c='g')
    # plt.scatter(
    #     cluster[1][binar_msk], cluster[0][binar_msk], marker='.', c='r')
    # plt.plot(envelope[1], envelope[0], '.', ms=2, c='k')
    # plt.gca().invert_yaxis()
    # plt.show()

    return (l_envelope, b_envelope), single_msk, binar_msk


def rotIsoch(best_pars):
    """
    Find the angle that rotates the MS part of the isochrone, such that the
    absolute valued slope is minimizes. This aligns the MS with the x axis
    """
    # Read the isochrone
    met, age, ext, dist = best_pars
    turn_off, isoch_phot = isochHandle.isochProcess(
        cmd_systs, idx_header, met, age, ext, dist)[:2]

    # Define the range for the MS. Use a larger value than the turn off, and
    # remove the last points which show an artifact
    min_m, max_m = turn_off + 3, isoch_phot[0][500]
    msk = (isoch_phot[0] > min_m) & (isoch_phot[0] < max_m)
    isoch_phot = isoch_phot[:, msk]
    x, y = isoch_phot[1], -isoch_phot[0]
    xy_origin = (x.max(), y.min())

    def minim(theta):
        xy_rot = rotate(theta[0], np.array([x, y]).T, xy_origin)
        slope = abs(stats.linregress(xy_rot)[0])
        return slope

    # The angle is between these bounds
    bounds = [(50, 89)]
    result = differential_evolution(minim, bounds)
    theta = result.x[0]

    return theta


def rotate(degrees, p, origin):
    """
    https://stackoverflow.com/a/58781388/1391441
    """
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def masses(cluster, best_pars, binar_msk):
    """
    Estimate the *observed* binary systems' mass
    """
    met, age, ext, dist = best_pars
    binar_systs = cluster[:, binar_msk].T

    # Read and process (met, age) isochrone
    isoch_phot_best, mass_ini, isoch_col_mags =\
        isochHandle.isochProcess(
            cmd_systs, idx_header, met, age, ext, dist)[1:]
    m1_mass, m2_mass, res = clustHandle.binarMasses(
        isoch_phot_best, isoch_col_mags, mass_ini, cluster, binar_systs)

    return isoch_phot_best, m1_mass, m2_mass
