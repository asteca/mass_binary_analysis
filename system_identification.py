
from pathlib import Path
from itertools import product
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
from modules import readData, isochHandle, clustHandle, totalMass, makePlots

seed = 19198738 # np.random.randint(100000000)
print("Random seed: {}".format(seed))
RandomState(MT19937(SeedSequence(seed)))

clust_name = 'NGC2516'
print("Analyzing cluster: {}".format(clust_name))

# Distance to the isochrone (in magnitudes) to separates single and
# binary systems. The larger this value, the smaller the estimated binary
# fraction will be, and thus the total estimated mass.
thresh = 0.05  # 0.025  0.05  0.075  0.1
print("Distance threshold used: {}".format(thresh))

# ONLY Gaia EDR3 and Gaia DR2 photometric systems are supported
gaia_ID = 'gaiaedr3'  # 'gaiadr2'
cmd_systs = {
    'gaiadr2': (
        ('Gmag', 'G_BPmag', 'G_RPmag'), (6437.7, 5309.57, 7709.85)),
    'gaiaedr3': (
        ('Gmag', 'G_BPmag', 'G_RPmag'), (6422.01, 5335.42, 7739.17))
}

# line where the header starts in the isochrone file
idx_header = 11

# Define (z, a, e, d) ranges for the isochrones using (16, 84) values
met_l = (0.01745, 0.01892,)
age_l = (7.91847, 7.92642)
Nvals = 3
ext_l = np.linspace(0.181808, 0.202378, Nvals)
dist_l = np.linspace(8.09084, 8.13043, Nvals)

# IMF_name, Max_mass = 'salpeter_1955', 8000
IMF_name, Max_mass = 'kroupa_2002', 10000

# Number of times the process to estimate the total single systems' mass
# will be repeated
N_mass_tot = 10
# List of bins to use when comparing the mass distribution of the observed
# single systems versus the sampled IMF.
bins_list = list(np.arange(25, 51, 5)) + [
    'knuth', 'block', 'auto', 'fd', 'doane', 'scott', 'rice', 'sturges',
    'sqrt']
bins_list = np.arange(25, 51, 5)

# IMF used to estimate the total mass, and maximum mass allowed
IMF_method = 'no' #'MASSCLEAN'
#
method = '2' #'1'  #'2'


def main():
    """
    """
    print("\n----------------------------------")
    print("Using Method {}\n".format(method))

    # Load observed cluster
    cluster = readData.loadClust(clust_name)

    binar_fr_all, Gobs_s, Gobs_b = [], [], []
    tot_single_mass_all, tot_binar_mass_all, total_mass_all = [], [], []
    m1_all, m2_all, res_all, G_binars, qvals = [], [], [], [], []

    if IMF_method != 'MASSCLEAN':
        # Use my own IMF sampling
        print("\nGenerating CDF for '{}' IMF (M_max={})...".format(
            IMF_name, Max_mass))
        inv_cdf = totalMass.IMF_CDF(IMF_name, Max_mass)
    else:
        # Usa MASSCLEAN
        print("\nGenerating CDF from MASSCLEAN file (M_max={})...".format(
            Max_mass))
        inv_cdf = CDF_MASSCLEAN(Max_mass)

    params = list(product(*(met_l, age_l, ext_l, dist_l)))
    Ntot = len(params)
    for pi, (met, age, ext, dist) in enumerate(params):
        print("\n{}/{}.\n(z, log(age), E_BV, dm): {}, {}, {}, {}".format(
            pi + 1, Ntot, met, age, ext, dist))

        # Read and process (met, age) isochrone
        turn_off, isoch_phot, isoch_col_mags, mass_ini =\
            isochHandle.isochProcess(
                gaia_ID, cmd_systs, idx_header, met, age, ext, dist)

        # Split single/binary block using 'thresh'
        # Estimate the *observed* single systems' mass
        single_msk, binar_msk, single_masses, b_fr = clustHandle.split(
            cluster, isoch_phot, mass_ini, thresh, turn_off)

        # Estimate the *observed* binary systems' mass
        m1_mass, m2_mass, res = clustHandle.binarMasses(
            isoch_phot, isoch_col_mags, mass_ini, cluster, binar_msk)

        # This call plots the CMD for a given (z,a,e,d) set. Generate only
        # once.
        if pi == 0:
            makePlots.CMD(
                thresh, met, age, ext, dist, clust_name, cluster,
                isoch_phot, single_msk, binar_msk, b_fr, single_masses,
                m1_mass, m2_mass)

        import pdb; pdb.set_trace()  # breakpoint 233a47a3 //

        print("Binary fraction: {:.2f}".format(b_fr))
        single_masses_sum = single_masses.sum()
        print("Mass of single systems in obs range: {:.0f}".format(
            single_masses_sum))
        binar_fr_all.append(b_fr)
        Gobs_s += list(cluster[0, :][single_msk])
        Gobs_b += list(cluster[0, :][binar_msk])

        if method == '1':
            tot_single_mass, tot_binar_mass = method1(
                IMF_name, Max_mass, inv_cdf, m1_mass, m2_mass, single_masses)
        elif method == '2':
            tot_single_mass, tot_binar_mass = method2(
                IMF_name, Max_mass, inv_cdf, single_masses, m1_mass, m2_mass)

        tot_single_mass_all += tot_single_mass
        m1_all += m1_mass
        m2_all += m2_mass
        res_all += res
        G_binars += list(cluster[0, binar_msk])
        qvals += list(np.array(m2_mass) / np.array(m1_mass))
        tot_binar_mass_all += list(tot_binar_mass)
        total_mass = tot_single_mass + tot_binar_mass
        total_mass_all += list(total_mass)
        print("Total single+binary cluster mass: {:.0f}+/-{:.0f}".format(
            total_mass.mean(), total_mass.std()))

    makePlots.final(
        thresh, method, clust_name, cluster, np.array(tot_single_mass_all),
        np.array(tot_binar_mass_all), np.array(total_mass_all), binar_fr_all,
        m1_all, m2_all, np.array(G_binars), np.array(qvals), Gobs_s, Gobs_b)
    print("Finished")


def method1(IMF_name, Max_mass, inv_cdf, m1_mass, m2_mass, single_masses):
    """
    """
    # Estimate the *total* single systems' mass
    mmin, mmax = single_masses.min(), single_masses.max()
    tot_single_mass = totalMass.get(
        IMF_name, Max_mass, inv_cdf,
        single_masses, mmin, mmax, N_mass_tot, bins_list)
    print("Total mass for single systems: {:.0f}+/-{:.0f}".format(
        np.mean(tot_single_mass), np.std(tot_single_mass)))

    binar_mass_sum = sum(m1_mass) + sum(m2_mass)
    print("Mass of binary systems in obs range: {:.0f}".format(
        binar_mass_sum))

    # Estimate the *total* binary systems' mass
    # M_fr: Fraction of single systems' mass in the observed range
    M_fr = single_masses.sum() / tot_single_mass
    # Assume that the same fraction of binary mass is observed. Hence the
    # observed binary mass can be written as:
    # binar_mass_sum = tot_binar_mass * M_fr
    # From this we can estimate the total binary mass as:
    tot_binar_mass = binar_mass_sum / M_fr
    print("Total mass for binary systems: {:.0f}+/-{:.0f}".format(
        tot_binar_mass.mean(), tot_binar_mass.std()))

    return tot_single_mass, tot_binar_mass


def method2(IMF_name, Max_mass, inv_cdf, single_masses, m1_mass, m2_mass):
    """
    """
    # Combine all masses
    comb_masses = np.array(list(single_masses) + m1_mass + m2_mass)

    # Estimate the *total* combined single systems' mass
    mmin, mmax = single_masses.min(), single_masses.max()
    tot_single_mass = totalMass.get(
        IMF_name, Max_mass, inv_cdf,
        comb_masses, mmin, mmax, N_mass_tot, bins_list)
    print("Total mass for single systems: {:.0f}+/-{:.0f}".format(
        np.mean(tot_single_mass), np.std(tot_single_mass)))

    tot_binar_mass = np.array([0.])

    return tot_single_mass, tot_binar_mass


def CDF_MASSCLEAN(Max_mass):
    """
    Use MASSCLEAN's 'goimf2' to generate a file with mass sampled in small
    steps until a total mass is generated
    """
    import subprocess
    from astropy.io import ascii
    from astropy.stats import bayesian_blocks
    from scipy import stats

    # Define IMF parameters in imf.ini file.
    # IMPORTANT: (5) is the minimum mass sampled from the IMF. By default this
    # is set to 0.15 because: "this is the lower limit of Padova models"
    # (Bogdan,private email). This must be lowered to 0.01 (the lower limit
    # of the IMF) otherwise mass is lost.
    # "about half of the mass lies in the 0.01-0.15 M_Sun range", Bogdan
    ini_file = """0.01       (1)
    0.08       (2)
    0.50       (3)
    500.0      (4)
    0.01       (5)
    150        (6)
    -99        (7)
    0.3        (8)
    1.3        (9)
    2.35      (10)
    1         (11)
    """
    with open("ini.files/imf.ini", "w") as f:
        f.write(ini_file)

    # Define total mass in cluster.ini file
    ini_file = """0       (1)
    {}    (2)
    10    (3)
    10    (4)
    2048    (5)
    2048    (6)
    2048   (7)
    2048   (8)
    1       (9)
    3.1    (10)
    0.0    (11)
    0.0    (12)
    0.    (13)
    0.0    (14)
    0.0    (15)
    0.0    (16)
    0.0    (17)
    0.0    (18)
    0      (19)
    0      (20)
    0      (21)
    0      (22)
    0      (23)
    """.format(Max_mass)
    with open("ini.files/cluster.ini", "w") as f:
        f.write(ini_file)
    # Call 'goimf2'
    bashCommand = "./goimf2"
    subprocess.call(bashCommand, stdout=subprocess.PIPE)

    # Read IMF
    sampled_imf = np.array(ascii.read(
        "n_distribution.out", format="no_header")['col1'])

    pk, xk = bayesian_blocks(sampled_imf)
    import matplotlib.pyplot as plt
    import pdb; pdb.set_trace()  # breakpoint d2113edb //

    pk, xk = np.histogram(sampled_imf, 10000)
    pk = 1. * pk
    pk /= pk.sum()
    fq = stats.rv_discrete(values=(xk[:-1], pk))
    # Inverse CDF
    inv_cdf = fq.ppf

    return inv_cdf


if __name__ == '__main__':
    out_folder = "out"
    # Create 'output' folder if it does not exist
    Path('./{}'.format(out_folder)).mkdir(parents=True, exist_ok=True)
    main()


# print("\nMethod 1 (use the total number of binary systems)")
# # Given the binary fraction, the number of single systems, and their
# # mass estimate the number of binary systems as:
# # Ns + Nb = Nt
# # b_fr = Nb/Nt
# # Nb = b_fr*Nt = b_fr*(Ns+Nb) <--> Nb = b_fr*Ns/(1-b_fr)
# N_single = np.mean(N_single)
# N_binar = b_fr * N_single / (1. - b_fr)
# # print("Estimated number of binary systems: {:.0f}".format(N_binar))
# # Estimate the binary mass as:
# # s*Nb = Ns_b
# # Ns   --- Mt
# # Ns_b --- x --> x = 2*Nb*Mt/Ns
# binar_mass = 2. * N_binar * np.mean(best_mass) / N_single
# print("Total mass for binary systems: {:.0f}".format(binar_mass))
# print("Total single+binary mass: {:.0f}".format(
#     np.mean(best_mass) + binar_mass))
