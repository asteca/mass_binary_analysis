
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
from modules import isochHandle
# import sys


def split(cluster, isoch_interp, thresh, turn_off):
    """
    Classify single/binary systems, making sure that systems
    classified as 'binaries' are located to the *right* of the isochrone
    """

    # Find the closest points in the isochrone for all the observed systems
    distances = cdist(cluster.T, isoch_interp.T)
    min_dist = distances.min(1)
    # Indexes in 'isoch_interp' for each observed star
    idxs_min_dist = np.argmin(distances, 1)

    # All stars closer to the isochrone than 'thresh' are considered single
    # systems

    print("FINISH THIS")
    # There's a 'dip' in the isochrones in the range G=[15, 17] that makes
    # many stars be counted as binaries when they are visibly not. We handle
    # this by increasing the 'thresh' N_mult times in this range.
    N_mult = 5
    new_thresh = np.ones(cluster.shape[-1]) * thresh
    msk_mag = (cluster[0] > 15.) & (cluster[0] < 17.)
    new_thresh[msk_mag] = N_mult * thresh
    msk1 = min_dist < new_thresh
    # msk1 = min_dist < thresh

    # Closest synthetic stars to observed stars
    synth_stars = isoch_interp[:, idxs_min_dist]
    # Color distance
    left_right = cluster[1, :] - synth_stars[1, :]
    # All stars with negative values are considered single systems
    msk2 = left_right < 0.
    # But only if they are also below the estimated turn-off point
    msk3 = msk2 & (cluster[0, :] > turn_off)

    # Combine with an OR
    single_msk = msk1 | msk3
    binar_msk = ~single_msk

    return single_msk, binar_msk


def splitEnv(
        cluster, thresh, col_step=.05, perc=75):
    """

    col_step: the step used in the in color to generate the envelope. Of
      minor importance as the envelope is then interpolated.

    perc: the magnitude percentile used to estimate the 'y' position of the
      envelope. A smaller value brings the envelope upwards in the CMD (i.e.:
      towards brighter magnitudes). The value of 75 is estimated heuristically
      and gives reasonable results.
    """

    # Obtain the lower envelope for the cluster's sequence
    col_min, col_max = cluster[1].min(), cluster[1].max()
    xx_yy = []
    for low in np.arange(col_min, col_max, col_step):
        msk = (cluster[1] > low) & (cluster[1] <= low + col_step)
        if msk.sum() > 0:
            xx_yy.append([np.percentile(cluster[0][msk], perc), low])
    # Generate extra points
    envelope = isochHandle.interp(np.array(xx_yy).T)

    # Distances to the envelope, for all the stars
    distances = cdist(cluster.T, envelope.T)
    min_dist = distances.min(1)
    idxs_min_dist = np.argmin(distances, 1)
    # Identify those closer to the envelope than the 'thresh' parameter
    msk1 = min_dist < thresh

    # Closest 'synthetic stars' (envelope points) to observed stars
    synth_stars = envelope[:, idxs_min_dist]
    # Color distance
    left_right = cluster[1, :] - synth_stars[1, :]
    # All stars with negative values are considered single systems
    msk2 = left_right < 0.

    # Split systems
    single_msk = msk1 | msk2
    binar_msk = ~single_msk

    # import matplotlib.pyplot as plt
    # plt.scatter(
    #     cluster[1][single_msk], cluster[0][single_msk], marker='.', c='g')
    # plt.scatter(
    #     cluster[1][binar_msk], cluster[0][binar_msk], marker='.', c='r')
    # plt.plot(envelope[1], envelope[0], '.', ms=2, c='k')
    # plt.gca().invert_yaxis()
    # plt.show()

    return envelope, single_msk, binar_msk


def singleMasses(cluster, mass_ini, isoch_interp, single_msk):
    """
    Return the masses for single systems identified as such
    """
    distances = cdist(cluster.T, isoch_interp.T)
    # Indexes in 'isoch_interp' for each observed star
    idxs_min_dist = np.argmin(distances, 1)

    single_masses = np.zeros(cluster.shape[-1])
    single_masses[single_msk] = mass_ini[idxs_min_dist][single_msk]
    # For non-single systems, save 'nan'
    single_masses[~single_msk] = np.nan

    return single_masses


def binarMasses(isoch_phot, isoch_col_mags, mass_ini, cluster, binar_systs):
    """
    Given a system identified as a binary, estimate its masses.
    """
    # xy = np.linspace(isoch_phot[0, :].min(), isoch_phot[0, :].max() + 5, 5000)
    # x, y = np.meshgrid(xy, xy)
    # mag_comb_grid = mag_combine(x, y)

    def minfunc(x, mag_obs, col_obs, plot_flag=False):
        mag1, q = x
        idx = np.argmin(abs(isoch_phot[0] - mag1))
        m1 = mass_ini[idx]
        m2 = q * m1
        i1 = np.searchsorted(mass_ini, m1)
        i2 = np.searchsorted(mass_ini, m2)
        Gmag_s1, Gmag_s2 = isoch_phot[0, i1], isoch_phot[0, i2]
        BPmag_1, RPmag_1 = isoch_col_mags[0, i1], isoch_col_mags[1, i1]
        BPmag_2, RPmag_2 = isoch_col_mags[0, i2], isoch_col_mags[1, i2]

        # Approach 1 (default)
        mag_binar = mag_combine(Gmag_s1, Gmag_s2)
        m1_col_binar = mag_combine(BPmag_1, BPmag_2)
        m2_col_binar = mag_combine(RPmag_1, RPmag_2)

        # # Approach 2 (slowest)
        # mag_binar, m1_col_binar, m2_col_binar = mag_combine(
        #     np.array([Gmag_s1, BPmag_1, RPmag_1]),
        #     np.array([Gmag_s2, BPmag_2, RPmag_2]))

        # # Approach 3 (not faster than 1)
        # i1, i2, i3, i4, i5, i6 = np.searchsorted(
        #     xy, (Gmag_s1, Gmag_s2, BPmag_1, BPmag_2, RPmag_1, RPmag_2))
        # # try:
        # mag_binar = mag_comb_grid[i1, i2]
        # m1_col_binar = mag_comb_grid[i3, i4]
        # m2_col_binar = mag_comb_grid[i5, i6]
        # # except IndexError:

        col_binar = m1_col_binar - m2_col_binar
        if plot_flag:
            return Gmag_s1, Gmag_s2, BPmag_1, RPmag_1, BPmag_2, RPmag_2,\
                mag_binar, col_binar

        # Don't take the square root, it's not necessary
        return (mag_binar - mag_obs)**2 + (col_binar - col_obs)**2

    m1_mass, m2_mass, res = [], [], []
    # Ntotal = cluster.shape[-1]
    for i, bin_phot in enumerate(binar_systs):
        mag, col = bin_phot
        mag_range = (mag, mag + 1.)
        bounds = ((mag_range), (0., 1.))

        result = differential_evolution(minfunc, bounds, args=(mag, col))
        mag1, q = result.x
        idx = np.argmin(abs(isoch_phot[0] - mag1))
        m1 = mass_ini[idx]
        m2 = q * m1
        m1_mass.append(m1)
        m2_mass.append(m2)
        res.append(result.fun)
        # updt(Ntotal, i + 1)

        # import matplotlib.pyplot as plt
        # def plotres(mag1, q):
        #     Gmag_s1, Gmag_s2, BPmag_1, RPmag_1, BPmag_2, RPmag_2, mag_binar,\
        #         col_binar = minfunc((mag1, q), mag, col, True)
        #     plt.plot(isoch_phot[1], isoch_phot[0])
        #     plt.scatter(BPmag_1 - RPmag_1, Gmag_s1, c='k')
        #     plt.scatter(BPmag_2 - RPmag_2, Gmag_s2, c='k')
        #     plt.scatter(col, mag, c='g')
        #     plt.scatter(col_binar, mag_binar, edgecolor='r', facecolor='none',
        #                 marker='s')
        #     plt.gca().invert_yaxis()
        #     plt.show()

        # print(mag, col, m1, q, result.fun)
        # plotres(mag1, q)

    return np.array(m1_mass), np.array(m2_mass), res


def mag_combine(m1, m2):
    """
    Combine two magnitudes. This is a faster re-ordering of the standard
    formula:

    -2.5 * np.log10(10 ** (-0.4 * m1) + 10 ** (-0.4 * m2))

    """

    # 10**-.4 = 0.398107
    mbin = -2.5 * (-.4 * m1 + np.log10(1. + 0.398107 ** (m2 - m1)))

    return mbin


# def updt(total, progress, extra=""):
#     """
#     Displays or updates a console progress bar.

#     Original source: https://stackoverflow.com/a/15860757/1391441
#     """
#     barLength, status = 20, ""
#     progress = float(progress) / float(total)
#     if progress >= 1.:
#         progress, status = 1, "\r\n"
#     block = int(round(barLength * progress))
#     text = "\r[{}] {:.0f}% {}{}".format(
#         "#" * block + "-" * (barLength - block),
#         round(progress * 100, 0), extra, status)
#     sys.stdout.write(text)
#     sys.stdout.flush()
