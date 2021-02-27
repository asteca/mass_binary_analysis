
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import sys


def split(cluster, isoch_interp, mass_ini, thresh, turn_off):
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
    msk1 = min_dist < thresh

    # Closest synthetic stars to observed stars
    synth_stars = isoch_interp[:, idxs_min_dist]
    # Color distance
    left_right = cluster[1, :] - synth_stars[1, :]
    # All stars with negative values are considered single systems
    msk2 = left_right < 0.
    # But only if they are also below the estimated turn-off point
    msk3 = msk2 & (cluster[0, :] > turn_off)

    # Combine with an OR
    msk = msk1 | msk3
    single_msk, binar_msk = msk, ~msk

    b_fr = binar_msk.sum() / (single_msk.sum() + binar_msk.sum())
    single_masses = mass_ini[idxs_min_dist][single_msk]

    return single_msk, binar_msk, single_masses, b_fr


def binarMasses(isoch_phot, isoch_col_mags, mass_ini, cluster, binar_msk):
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
        return np.sqrt((mag_binar - mag_obs)**2 + (col_binar - col_obs)**2)

    m1_mass, m2_mass, res = [], [], []
    # Ntotal = cluster.shape[-1]
    for i, bin_phot in enumerate(cluster[:, binar_msk].T):
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

    return m1_mass, m2_mass, res


def mag_combine(m1, m2):
    """
    Combine two magnitudes. This is a faster re-ordering of the standard
    formula:

    -2.5 * np.log10(10 ** (-0.4 * m1) + 10 ** (-0.4 * m2))

    """

    # 10**-.4 = 0.398107
    mbin = -2.5 * (-.4 * m1 + np.log10(1. + 0.398107 ** (m2 - m1)))

    return mbin


def updt(total, progress, extra=""):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}{}".format(
        "#" * block + "-" * (barLength - block),
        round(progress * 100, 0), extra, status)
    sys.stdout.write(text)
    sys.stdout.flush()
