
import numpy as np
from astropy.stats import bayesian_blocks, knuth_bin_width
from scipy.special import loggamma
from scipy.optimize import differential_evolution


def get(Max_mass, sampled_IMF_cmass, masses, mmin, mmax, bin_method='block'):
    """

    bin_method: hard-coded to 'block'
    """
    obs_mass_prep = prepObsMass(masses, bin_method)
    bounds = [(100, Max_mass)]

    def minfunc(M, sampled_IMF, cumm_mass):
        idx = np.searchsorted(cumm_mass, M[0])
        mass_M = sampled_IMF[:idx]
        # Mask for masses in the observed range
        msk = (mass_M >= mmin) & (mass_M <= mmax)
        return tremmel(mass_M[msk], obs_mass_prep)

    single_mass_lst = []
    for (sampled_IMF, cumm_mass) in sampled_IMF_cmass:

        result = differential_evolution(
            minfunc, bounds, args=(sampled_IMF, cumm_mass))
        # Store the best mass found
        single_mass_lst.append(result.x[0])

        # import matplotlib.pyplot as plt
        # n, bins, _ = plt.hist(masses, bin_e, color='g', alpha=.85)
        # idx = np.searchsorted(cumm_mass, best_mass)
        # mass_M = sampled_IMF[:idx]
        # plt.hist(
        #   mass_M, [0.01] + list(bins), color='r', histtype='step', lw=2)
        # # plt.xscale('log')
        # # plt.yscale('log')
        # plt.title("M={:.0f}".format(best_mass))
        # plt.show()

    return np.array(single_mass_lst)


def prepObsMass(obs_mass, bin_edges):
    """
    """
    # Obtain histogram for observed cluster.
    if bin_edges == 'knuth':
        bin_edges = knuth_bin_width(obs_mass, return_bins=True, quiet=True)[1]
    elif bin_edges == 'block':
        bin_edges = bayesian_blocks(obs_mass)
    cl_histo, bin_edges = np.histogram(obs_mass, bins=bin_edges)

    return [bin_edges, cl_histo]


def tremmel(synth_mass, obs_mass_prep):
    """
    Poisson likelihood ratio as defined in Tremmel et al (2013), E1 10 with
    v_{i,j}=1. This returns the negative log likelihood.
    """
    # Observed cluster's data.
    bin_edges, cl_histo = obs_mass_prep

    # Histogram of the synthetic cluster, using the bin edges calculated
    # with the observed cluster.
    syn_histo = np.histogram(synth_mass, bins=bin_edges)[0]
    SumLogGamma = np.sum(
        loggamma(cl_histo + syn_histo + .5) - loggamma(syn_histo + .5))

    # M = synth_clust.shape[0]
    # ln(2) ~ 0.693
    tremmel_lkl = 0.693 * synth_mass.size - SumLogGamma

    return tremmel_lkl

# # Compare with MASSCLEAN IMF sampling
# import matplotlib.pyplot as plt
# from astropy.io import ascii
# def IMFcomp(M):
#     massclean_cl = ascii.read('{}_M.out'.format(M))
#     sampled_IMF, cumm_mass = getIMF(IMF_name, sum(massclean_cl['col1']))
#     print(len(massclean_cl['col1']), len(sampled_IMF))
#     print(sum(massclean_cl['col1']), sum(sampled_IMF))
#     msk1 = massclean_cl['col1'] < 5
#     n, bins, _ = plt.hist(
#         massclean_cl['col1'][msk1], bins=50, color='r', alpha=.5)
#     msk2 = sampled_IMF < 5
#     plt.hist(sampled_IMF[msk2], bins=bins, color='g', alpha=.5)
#     plt.show()
# IMFcomp(500) # 1000 5000 10000
