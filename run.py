
from pathlib import Path
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
from modules import readData, totalMass, binary, clustHandle, makePlots


def main():
    """
    """

    clust_name, best_pars, IMF_name, Max_mass, N_mass_tot, bins_list =\
        readData.readINI()

    print("\n----------------------------------")
    print("Analyzing cluster: {}".format(clust_name))

    # Load observed cluster
    cluster = readData.loadClust(clust_name)

    print("\nGenerating CDF for '{}' IMF (M_max={})...".format(
        IMF_name, Max_mass))
    inv_cdf = totalMass.IMF_CDF(IMF_name, Max_mass)

    print("\nIdentifying single/binary systems")
    envelopes, single_msk, binar_msk = binary.splitEnv(cluster, best_pars)

    single_masses = clustHandle.singleMasses(cluster, best_pars, single_msk)

    print("\nBinary fraction: {:.2f}".format(
        binar_msk.sum() / cluster.shape[1]))
    print("Mass of single systems in obs range: {:.0f}".format(
        single_masses.sum()))

    print("Estimating binary masses...")
    isoch_phot_best, m1_mass, m2_mass = binary.masses(
        cluster, best_pars, binar_msk)
    print("Mass of binary systems in obs range: {:.0f}".format(
        m1_mass.sum() + m2_mass.sum()))

    tot_mass_1, tot_mass_2, tot_mass_3 =\
        totMass(IMF_name, Max_mass, N_mass_tot, bins_list, inv_cdf,
                single_masses, m1_mass, m2_mass)

    makePlots.final(
        clust_name, cluster, best_pars, isoch_phot_best, envelopes, single_msk,
        binar_msk, single_masses, m1_mass, m2_mass, tot_mass_1, tot_mass_2,
        tot_mass_3)

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


if __name__ == '__main__':
    # Set random seed
    seed = np.random.randint(100000000)
    print("Random seed: {}".format(seed))
    RandomState(MT19937(SeedSequence(seed)))

    out_folder = "out"
    # Create 'output' folder if it does not exist
    Path('./{}'.format(out_folder)).mkdir(parents=True, exist_ok=True)
    main()
