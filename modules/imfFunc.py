
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d


# Low mass limits for each IMF. Defined slightly larger to avoid sampling
# issues.
imfs_dict = {
    'chabrier_2001_exp': 0.011, 'chabrier_2001_log': 0.011,
    'kroupa_1993': 0.081, 'kroupa_2002': 0.01, 'salpeter_1955': 0.31}
m_high = 150


def IMFSampling(IMF_name, Max_mass, N_mass_tot):
    """
    """
    inv_cdf = IMF_CDF(IMF_name)

    sampled_IMF_cmass = []
    for _ in range(N_mass_tot):
        # Random IMF sample
        sampled_IMF, cumm_mass = getIMF(Max_mass, inv_cdf)
        sampled_IMF_cmass.append([sampled_IMF, cumm_mass])

    return sampled_IMF_cmass, inv_cdf


def IMF_CDF(IMF_name, mass_step=0.05):
    """
    Generate the inverse CDF for the selected IMF.
    """
    # IMF low mass limit.
    m_low = imfs_dict[IMF_name]

    # IMF mass interpolation step and grid values.
    mass_values = np.arange(m_low, m_high, mass_step)

    # The CDF is defined as: $F(m)= \int_{m_low}^{m} PDF(m) dm$
    # Sample the CDF
    CDF_samples = []
    for m in mass_values:
        CDF_samples.append(quad(imfs, m_low, m, args=(IMF_name))[0])
    CDF_samples = np.array(CDF_samples)

    # Normalize CDF
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


def IMFinteg(IMF_name, mass_step=0.05):
    """
    """
    # IMF low mass limit.
    m_low = imfs_dict[IMF_name]

    # IMF mass interpolation step and grid values.
    mass_values = np.arange(m_low, m_high, mass_step)

    # Normalized integral of the IMF (total mass)
    norm_const = quad(imfs, m_low, m_high, args=(IMF_name, True))[0]
    IMF_int = []
    mlow = m_low + 0.001
    for m in mass_values:
        IMF_int.append(quad(imfs, mlow, m, args=(IMF_name, True))[0])
        mlow = m
    IMF_int = np.array(IMF_int) / norm_const

    return mass_values, IMF_int


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
