
import numpy as np
from astropy.stats import bayesian_blocks, knuth_bin_width
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import loggamma
from scipy.optimize import differential_evolution
# from .clustHandle import updt


# Low mass limits for each IMF. Defined slightly larger to avoid sampling
# issues.
imfs_dict = {
    'chabrier_2001_exp': 0.011, 'chabrier_2001_log': 0.011,
    'kroupa_1993': 0.081, 'kroupa_2002': 0.01, 'salpeter_1955': 0.31}
m_high = 150


def get(
    IMF_name, Max_mass, inv_cdf, masses, mmin, mmax, N_mass_tot,
        bins_list):
    """
    """
    # # Compare with MASSCLEAN IMF sampling
    # import matplotlib.pyplot as plt
    # from astropy.io import ascii
    # def IMFcomp(M):
    #     massclean_cl = ascii.read('{}_M.out'.format(M))
    #     sampled_IMF, cumm_mass = getIMF(IMF_name, sum(massclean_cl['col1']))
    #     print(len(massclean_cl['col1']), len(sampled_IMF))
    #     print(sum(massclean_cl['col1']), sum(sampled_IMF))
    #     msk1 = massclean_cl['col1'] < 5
    #     n, bins, _ = plt.hist(massclean_cl['col1'][msk1], bins=50, color='r', alpha=.5)
    #     msk2 = sampled_IMF < 5
    #     plt.hist(sampled_IMF[msk2], bins=bins, color='g', alpha=.5)
    #     plt.show()
    # IMFcomp(500) # 1000 5000 10000

    # print("Sampling IMF...")
    single_mass_lst = []
    for _ in range(N_mass_tot):
        sampled_IMF, cumm_mass = getIMF(Max_mass, inv_cdf)

        def minfunc(M):
            idx = np.searchsorted(cumm_mass, M[0])
            mass_M = sampled_IMF[:idx]
            # Mask for masses in the observed range
            msk = (mass_M >= mmin) & (mass_M <= mmax)
            return tremmel(mass_M[msk], obs_mass_prep)

        for i, bin_e in enumerate(bins_list):
            obs_mass_prep = prepObsMass(masses, bin_e)
            bounds = [(100, Max_mass)]
            result = differential_evolution(minfunc, bounds)
            best_mass = result.x[0]
            single_mass_lst.append(best_mass)

            # import matplotlib.pyplot as plt
            # n, bins, _ = plt.hist(masses, bin_e, color='g', alpha=.85)
            # idx = np.searchsorted(cumm_mass, best_mass)
            # mass_M = sampled_IMF[:idx]
            # plt.hist(
            #     mass_M, [0.01] + list(bins), color='r', histtype='step', lw=2)
            # # plt.xscale('log')
            # # plt.yscale('log')
            # plt.title("M={:.0f}".format(best_mass))
            # plt.show()

    return np.array(single_mass_lst)


def IMF_CDF(IMF_name, Max_mass):
    """
    Generate the inverse CDF for the selected IMF.
    """

    # IMF low mass limit.
    m_low = imfs_dict[IMF_name]

    # Obtain normalization constant (k = \int_{m_low}^{m_up} \xi(m) dm). This
    # makes the IMF behave like a PDF.
    # norm_const = quad(imfs, m_low, m_high, args=(IMF_name))[0]

    # IMF mass interpolation step and grid values.
    mass_step = 0.05
    mass_values = np.arange(m_low, m_high, mass_step)

    # The CDF is defined as: $F(m)= \int_{m_low}^{m} PDF(m) dm$
    # Sample the CDF
    CDF_samples = []
    for m in mass_values:
        CDF_samples.append(quad(imfs, m_low, m, args=(IMF_name))[0])
    CDF_samples = np.array(CDF_samples)  # / norm_const

    # Normalize values
    CDF_samples /= CDF_samples.max()

    # Inverse CDF
    inv_cdf = interp1d(CDF_samples, mass_values)

    return inv_cdf


def getIMF(Max_mass, inv_cdf):
    """
    """

    def sampled_inv_cdf(N):
        mr = np.random.rand(N)
        return inv_cdf(mr)

    # Sample in chunks of 100 stars until the maximum defined mass is reached.
    mass_samples = []
    while np.sum(mass_samples) < Max_mass:
        mass_samples += sampled_inv_cdf(100).tolist()
    sampled_IMF = np.array(mass_samples)

    cumm_mass = np.cumsum(sampled_IMF)
    idx = np.searchsorted(cumm_mass, Max_mass)
    sampled_IMF, cumm_mass = sampled_IMF[:idx], cumm_mass[:idx]

    return sampled_IMF, cumm_mass


def extrapolate(IMF_name, comb_masses, mmin, N_bootstrp=1000, mass_flag=True):
    """
    """
    m_low = imfs_dict[IMF_name]
    norm_const = quad(imfs, m_low, m_high, args=(IMF_name, mass_flag))[0]
    mass_step = 0.05
    mass_values = np.arange(m_low, m_high, mass_step)

    IMF_int = []
    mlow = m_low + 0.001
    for m in mass_values:
        IMF_int.append(quad(imfs, mlow, m, args=(IMF_name, mass_flag))[0])
        mlow = m
    IMF_int = np.array(IMF_int) / norm_const

    idx = np.argmin(abs(mmin - mass_values))
    perc_mass_min = np.cumsum(IMF_int)[idx]
    mmax = np.percentile(comb_masses, 99.)
    idx = np.argmin(abs(mmax - mass_values))
    perc_mass_max = np.cumsum(IMF_int)[idx]
    perc_tot = perc_mass_min + (1. - perc_mass_max)

    tot_mass = []
    for _ in range(N_bootstrp):
        ran_masses = np.random.choice(comb_masses, len(comb_masses))
        msk = (ran_masses >= mmin) & (ran_masses <= mmax)
        mass_obs = ran_masses[msk].sum()
        tot_mass.append(mass_obs / (1. - perc_tot))

    return np.array(tot_mass)


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


def imfs(m_star, IMF_name, mass_flag=False):
    """
    Define any number of IMFs.

    The package https://github.com/keflavich/imf has some more (I think,
    24-09-2019).
    """

    if IMF_name == 'kroupa_1993':
        # Kroupa, Tout & Gilmore. (1993) piecewise IMF.
        # http://adsabs.harvard.edu/abs/1993MNRAS.262..545K
        # Eq. (13), p. 572 (28)
        alpha = [-1.3, -2.2, -2.7]
        m0, m1, m2 = [0.08, 0.5, 1.]
        factor = [0.035, 0.019, 0.019]
        if m0 < m_star <= m1:
            i = 0
        elif m1 < m_star <= m2:
            i = 1
        elif m2 < m_star:
            i = 2
        imf_val = factor[i] * (m_star ** alpha[i])

    elif IMF_name == 'kroupa_2002':
        # Kroupa (2002) Salpeter (1995) piecewise IMF taken from MASSCLEAN
        # article, Eq. (2) & (3), p. 1725
        alpha = [-0.3, -1.3, -2.3]
        m0, m1, m2 = [0.01, 0.08, 0.5]
        factor = [(1. / m1) ** alpha[0], (1. / m1) ** alpha[1],
                  ((m2 / m1) ** alpha[1]) * ((1. / m2) ** alpha[2])]
        if m0 <= m_star <= m1:
            i = 0
        elif m1 < m_star <= m2:
            i = 1
        elif m2 < m_star:
            i = 2
        imf_val = factor[i] * (m_star ** alpha[i])

    elif IMF_name == 'chabrier_2001_log':
        # Chabrier (2001) lognormal form of the IMF.
        # http://adsabs.harvard.edu/abs/2001ApJ...554.1274C
        # Eq (7)
        imf_val = (1. / (np.log(10) * m_star)) * 0.141 * \
            np.exp(-((np.log10(m_star) - np.log10(0.1)) ** 2)
                   / (2 * 0.627 ** 2))

    elif IMF_name == 'chabrier_2001_exp':
        # Chabrier (2001) exponential form of the IMF.
        # http://adsabs.harvard.edu/abs/2001ApJ...554.1274C
        # Eq (8)
        imf_val = 3. * m_star ** (-3.3) * np.exp(-(716.4 / m_star) ** 0.25)

    elif IMF_name == 'salpeter_1955':
        # Salpeter (1955)  IMF.
        # https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/
        imf_val = m_star ** -2.35

    if mass_flag:
        imf_val *= m_star

    return imf_val
