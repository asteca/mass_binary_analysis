
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
from scipy import stats
from modules import isochHandle, clusterHandle
from astropy.stats import knuth_bin_width


def splitEnv(cluster, turn_off, isoch_phot, low_env_perc=50):
    """
    TODO

    1. implement iterative outliers removal for the estimation of the MSRL
    2. don't use binary envelope. Instead use the following method:
      - estimate the MSRL
      - divide it in (rotated) magnitude bins
      - for each bin, count how many members there are below the MSRL
      - estimate the
    """

    # Estimate the optimal rotation angle using the best fit isochrone
    theta = rotIsoch(turn_off, isoch_phot)

    # Rotate the cluster using 'theta'
    origin = (cluster[0].max(), cluster[1].max())
    cluster_rot = rotate(theta, cluster.T, origin).T

    # Define the edges along the rotated sequence
    bin_edges = knuth_bin_width(
        cluster_rot[1], return_bins=True, quiet=True)[1]
    # Remove edges in the brightest portion
    msk = bin_edges > np.percentile(cluster_rot[1], .1)
    bin_edges = bin_edges[msk]
    # Add resolution to the low mass region
    extra_edges = np.linspace(bin_edges[-2], bin_edges[-1], 5)
    bin_edges = list(bin_edges[:-2]) + list(extra_edges)

    # Obtain main sequence ridge line (MSRL)
    msrl_rot = []
    for i, low in enumerate(bin_edges):
        if i + 1 == len(bin_edges):
            break
        msk = (cluster_rot[1] > low) & (cluster_rot[1] <= bin_edges[i + 1])
        if msk.sum() > 0:
            mid_p = (low + bin_edges[i + 1]) * .5
            msrl_rot.append([
                np.percentile(cluster_rot[0][msk], low_env_perc), mid_p])

    mslr_interp = isochHandle.interp(np.array(msrl_rot).T)
    dist_l = cdist(cluster_rot.T, mslr_interp.T)
    idxs = dist_l.argmin(1)
    mslr_interp[1, idxs]
    y_dist = cluster_rot[1] - mslr_interp[1, idxs]
    msk_l = y_dist < 0
    import matplotlib.pyplot as plt
    plt.scatter(cluster_rot[1][msk_l], cluster_rot[0][msk_l], marker='.', c='g')
    plt.scatter(cluster_rot[1][~msk_l], cluster_rot[0][~msk_l], marker='.', c='r')
    mag_l, col_l = np.array(msrl_rot).T
    plt.plot(col_l, mag_l, c='k')
    # plt.gca().invert_yaxis()
    plt.show()

    # Rotate the lower envelope back to its original position
    msrl = rotate(-theta, msrl_rot, origin).T

    # Extend envelope to lower magnitudes
    poly = np.polyfit(msrl[0][-3:], msrl[1][-3:], deg=1)
    # Extrapolate 1 mag
    x_ext = msrl[0][-1] + 1
    y_ext = np.polyval(poly, x_ext)
    msrl = np.array([list(msrl[0]) + [x_ext], list(msrl[1]) + [y_ext]])

    # import matplotlib.pyplot as plt
    # # plt.subplot(121)
    # plt.scatter(cluster_rot[1], cluster_rot[0], marker='.', c='r')
    # mag_l, col_l = np.array(msrl_rot).T
    # plt.plot(col_l, mag_l)
    # plt.gca().invert_yaxis()
    # # plt.subplot(122)
    # # plt.scatter(cluster[1], cluster[0], marker='.', c='r')
    # # mag_l, col_l = msrl
    # # plt.plot(col_l, mag_l)
    # # plt.gca().invert_yaxis()
    # plt.show()

    # Generate binary envelope
    mag_l, col_l = msrl
    mag_binar = clusterHandle.mag_combine(mag_l, mag_l)

    # Generate extra points
    l_envelope = isochHandle.interp(msrl)
    b_envelope = isochHandle.interp(np.array([mag_binar, col_l]))

    # cluster = remOutliers(cluster, l_envelope, col_max, mag_lim, delta)

    # import matplotlib.pyplot as plt
    # plt.scatter(cluster[1], cluster[0], marker='.', c='g')
    # plt.plot(l_envelope[1], l_envelope[0], 'x', ms=2, c='k')
    # plt.plot(b_envelope[1], b_envelope[0], 'x', ms=2, c='b')
    # plt.gca().invert_yaxis()
    # plt.show()

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

    return cluster, (l_envelope, b_envelope), single_msk, binar_msk


def rotIsoch(turn_off, isoch_phot):
    """
    Find the angle that rotates the MS part of the isochrone, such that the
    absolute valued slope is minimizes. This aligns the MS with the x axis
    """
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


# def remOutliers(cluster, l_envelope, col_max, mag_lim, delta):
#     """
#     Remove outlier stars
#     """
#     Nold = len(cluster[0])
#     # Keep stars with colors below this max value
#     msk1 = cluster[1] < col_max
#     #
#     # Distances to the lower envelope, for all the stars
#     dist_l = cdist(cluster.T, l_envelope.T)
#     idxs_min_dist = np.argmin(dist_l, 1)
#     synth_stars = l_envelope[:, idxs_min_dist]
#     # Color distance
#     left_right = cluster[1, :] - synth_stars[1, :]
#     # Keep stars above the lower envelope
#     msk2 = (left_right > -delta) | (cluster[0] < mag_lim)
#     #
#     msk = msk1 & msk2
#     cluster = cluster[:, msk]
#     print("Removed {} stars".format(Nold - len(cluster[0])))

#     return cluster


def masses(cluster, isoch_phot_best, mass_ini, isoch_col_mags, binar_msk):
    """
    Estimate the *observed* binary systems' mass
    """
    binar_systems = cluster[:, binar_msk].T

    m1_mass, m2_mass, res = clusterHandle.binarMasses(
        isoch_phot_best, isoch_col_mags, mass_ini, cluster, binar_systems)

    return m1_mass, m2_mass
