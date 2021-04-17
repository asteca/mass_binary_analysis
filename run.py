
from pathlib import Path
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
from modules import readData, binary, totalMass, makePlots

seed = np.random.randint(100000000)
print("Random seed: {}".format(seed))
RandomState(MT19937(SeedSequence(seed)))


def main():
    """
    """
    clust_name, gaia_ID, met_l, age_l, ext_l, dist_l, Nvals, best_pars,\
        thresh_binar, binar_P_thresh, splitmethod, IMF_name, Max_mass,\
        N_mass_tot, bins_list = readData.readINI()

    print("\n----------------------------------")
    print("Analyzing cluster: {}".format(clust_name))

    # Load observed cluster
    cluster = readData.loadClust(clust_name)

    print("\nSplit single/binary systems method: {}".format(splitmethod))

    # Use my own IMF sampling
    print("\nGenerating CDF for '{}' IMF (M_max={})...".format(
        IMF_name, Max_mass))
    inv_cdf = totalMass.IMF_CDF(IMF_name, Max_mass)
    # # Use MASSCLEAN
    # print("Generating CDF from MASSCLEAN file (M_max={})...".format(
    #     Max_mass))
    # inv_cdf = CDF_MASSCLEAN(Max_mass)

    print("\nIdentifying single/binary systems")
    envelope, single_msk, binar_msk, binar_probs, single_masses,\
        binar_fr_all = binary.ID(
            cluster, gaia_ID, met_l, age_l, ext_l, dist_l, Nvals, thresh_binar,
            binar_P_thresh, splitmethod)
    print("\nBinary fraction: {:.2f}+/-{:.2f}".format(
        np.mean(binar_fr_all), np.std(binar_fr_all)))
    print("Binary fraction (P>{}): {:.2f}".format(
        binar_P_thresh, binar_msk.sum() / cluster.shape[-1]))
    print("Mass of single systems in obs range: {:.0f}".format(
        single_masses.sum()))

    print("Estimating binary masses...")
    isoch_phot_best, m1_mass, m2_mass = binary.masses(
        cluster, gaia_ID, best_pars, binar_msk)
    print("Mass of binary systems in obs range: {:.0f}".format(
        m1_mass.sum() + m2_mass.sum()))

    tot_mass_1, tot_mass_2, tot_mass_3 =\
        totMass(IMF_name, Max_mass, N_mass_tot, bins_list, inv_cdf,
                single_masses, m1_mass, m2_mass)

    makePlots.final(
        clust_name, cluster, best_pars, isoch_phot_best, envelope, single_msk, binar_msk,
        binar_probs, binar_P_thresh, splitmethod, single_masses, binar_fr_all,
        m1_mass, m2_mass, tot_mass_1, tot_mass_2, tot_mass_3)

    print("Finished")


def totMass(
    IMF_name, Max_mass, N_mass_tot, bins_list, inv_cdf, single_masses,
        m1_mass, m2_mass):
    """
    Estimate the *total* combined single systems' mass
    """
    mmin, mmax = single_masses.min(), single_masses.max()

    # Method 1
    print("\nMethod 1...")
    tot_single_mass_1 = totalMass.get(
        IMF_name, Max_mass, inv_cdf, single_masses, mmin, mmax, N_mass_tot,
        bins_list)
    # print("Total mass for single systems (1): {:.0f}+/-{:.0f}".format(
    #     tot_single_mass_1.mean(), tot_single_mass_1.std()))
    binar_mass_sum = sum(m1_mass) + sum(m2_mass)
    # print("Mass of binary systems in obs range: {:.0f}".format(
    #     binar_mass_sum))
    # Estimate the *total* binary systems' mass
    # M_fr: Fraction of single systems' mass in the observed range
    M_fr = single_masses.sum() / tot_single_mass_1
    # Assume that the same fraction of binary mass is observed. Hence the
    # observed binary mass can be written as:
    # binar_mass_sum = tot_binar_mass * M_fr
    # From this we can estimate the total binary mass as:
    tot_binar_mass = binar_mass_sum / M_fr
    # print("Total mass for binary systems (1): {:.0f}+/-{:.0f}".format(
    #     tot_binar_mass.mean(), tot_binar_mass.std()))
    tot_mass_1 = tot_single_mass_1 + tot_binar_mass
    print("Total single+binary cluster mass: {:.0f}+/-{:.0f}".format(
        tot_mass_1.mean(), tot_mass_1.std()))

    # Method 2
    # Combine all masses
    print("\nMethod 2...")
    comb_masses = np.array(
        list(single_masses) + list(m1_mass) + list(m2_mass))
    tot_mass_2 = totalMass.get(
        IMF_name, Max_mass, inv_cdf, comb_masses, mmin, mmax, N_mass_tot,
        bins_list)
    print("Total single+binary cluster mass: {:.0f}+/-{:.0f}".format(
        tot_mass_2.mean(), tot_mass_2.std()))

    # Method 3
    print("\nMethod 3...")
    tot_mass_3 = totalMass.extrapolate(
        IMF_name, comb_masses, mmin)
    print("Total single+binary cluster mass: {:.0f}+/-{:.0f}".format(
        tot_mass_3.mean(), tot_mass_3.std()))

    return tot_mass_1, tot_mass_2, tot_mass_3


def CDF_MASSCLEAN(Max_mass):
    """
    Use MASSCLEAN's 'goimf2' to generate a file with mass sampled in small
    steps until a total mass is generated
    """
    import subprocess
    from astropy.io import ascii
    from astropy.stats import bayesian_blocks
    from scipy import stats

    Max_mass = 10000

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
    print("MASSCLEAN mass:", sampled_imf.sum())
    # sampled_imf = sampled_imf[sampled_imf < 1.]

    # pk, xk = bayesian_blocks(sampled_imf)
    import matplotlib.pyplot as plt
    plt.title("Kroupa (alpha: 0.3, 1.3, 2.3), M={:.0f}".format(Max_mass))

    inv_cdf = totalMass.IMF_CDF('kroupa_2002', Max_mass)
    my_IMF, cumm_mass = totalMass.getIMF(Max_mass, inv_cdf)
    print("My mass:", my_IMF.sum())
    # my_IMF = my_IMF[my_IMF < 1.]

    sampled_imf.sort()
    plt.plot(sampled_imf, np.cumsum(sampled_imf), c='r', label="goimf2")
    my_IMF.sort()
    plt.plot(my_IMF, np.cumsum(my_IMF), c='g', label="my sampling")

    # n, bins, _ = plt.hist(my_IMF, 50, color='g', alpha=.85,
    #     label='my sampling')
    # # [0.01] + 
    # plt.hist(
    #     sampled_imf, list(bins), color='r', histtype='step', lw=2,
    #     label="goimf2")

    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.show()

    import pdb; pdb.set_trace()  # breakpoint 611fb2ff //


    pk, xk = np.histogram(sampled_imf, Max_mass)
    # xk = np.arange(pk.size)
    pk = 1. * pk
    pk /= pk.sum()
    fq = stats.rv_discrete(values=(xk[:-1], pk))
    # Inverse CDF
    # inv_cdf = fq.ppf
    dist = fq.ppf(np.random.uniform(0., 1., 100000))
    dist = dist[dist < .5]

    plt.hist(dist, 100, histtype='step')
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(xk, fq.pmf(xk), 'ro', ms=12, mec='r')
    # ax.vlines(xk, 0, fq.pmf(xk), colors='r', lw=4)
    plt.show()
    import pdb; pdb.set_trace()  # breakpoint 6b7b57eb //


    return inv_cdf


if __name__ == '__main__':
    out_folder = "out"
    # Create 'output' folder if it does not exist
    Path('./{}'.format(out_folder)).mkdir(parents=True, exist_ok=True)
    main()
