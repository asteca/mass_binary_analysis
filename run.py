
from pathlib import Path
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
from modules import readData, imfFunc, totalMass, binary, clusterHandle,\
    isochHandle, makePlots
from modules.HARDCODED import cmd_systs, idx_header


def main():
    """
    """
    clust_name, best_pars, low_env_perc, IMF_name, Max_mass, N_IMF_samp =\
        readData.readINI()

    print("\n----------------------------------")
    print("Analyzing cluster: {}".format(clust_name))

    # Load observed cluster
    cluster = readData.loadClust(clust_name)

    print("\nSampling the '{}' IMF (N={})...".format(IMF_name, N_IMF_samp))
    sampled_IMF_cmass, inv_cdf = imfFunc.IMFSampling(
        IMF_name, Max_mass, N_IMF_samp)

    # Obtain the integrated IMF and mass values. Used by 'method3'
    mass_values, IMF_int = imfFunc.IMFinteg(IMF_name)

    print("Load isochrone")
    met, age, ext, dist = best_pars
    turn_off, isoch_phot, mass_ini, isoch_col_mags = isochHandle.isochProcess(
        cmd_systs, idx_header, met, age, ext, dist)

    print("\nIdentifying single/binary systems")
    cluster, envelopes, single_msk, binar_msk = binary.splitEnv(
        cluster, turn_off, isoch_phot, low_env_perc)
    print("Binary fraction: {:.2f}".format(binar_msk.sum() / cluster.shape[1]))

    single_masses = clusterHandle.singleMasses(
        cluster, isoch_phot, mass_ini, single_msk)
    print("\nMass of single systems in obs range: {:.0f}".format(
        single_masses.sum()))

    print("Estimating binary masses...")
    m1_mass, m2_mass = binary.masses(
        cluster, isoch_phot, mass_ini, isoch_col_mags, binar_msk)
    print("Mass of binary systems in obs range: {:.0f}".format(
        m1_mass.sum() + m2_mass.sum()))

    mmin, mmax = single_masses.min(), single_masses.max()
    # Combine single and binary (primary and secondary) masses
    comb_masses = np.array(list(single_masses) + list(m1_mass) + list(m2_mass))

    print("\nEstimating the cluster's total mass...")
    print("Method 1...")
    tot_mass_1 = method1(
        Max_mass, sampled_IMF_cmass, single_masses, m1_mass, m2_mass, mmin,
        mmax)
    print("Total single+binary cluster mass: {:.0f}+/-{:.0f}".format(
        tot_mass_1.mean(), tot_mass_1.std()))

    print("Method 2...")
    tot_mass_2 = method2(Max_mass, sampled_IMF_cmass, comb_masses, mmin, mmax)
    print("Total single+binary cluster mass: {:.0f}+/-{:.0f}".format(
        tot_mass_2.mean(), tot_mass_2.std()))

    print("Method 3...")
    tot_mass_3 = method3(
        N_IMF_samp, mass_values, IMF_int, comb_masses, mmin, mmax)
    print("Total single+binary cluster mass: {:.0f}+/-{:.0f}".format(
        tot_mass_3.mean(), tot_mass_3.std()))

    avrg_mass, avrg_Nmemb = avrgMassNmembs(
        inv_cdf, tot_mass_1, tot_mass_2, tot_mass_3)
    print("\nEstimating the average mass and total number of stars...")
    print("<m>={:.3f} , <N>={:.0f}".format(avrg_mass, avrg_Nmemb))

    makePlots.final(
        clust_name, cluster, low_env_perc, envelopes, single_msk, binar_msk,
        single_masses, m1_mass, m2_mass, tot_mass_1, tot_mass_2, tot_mass_3)


def method1(
    Max_mass, sampled_IMF_cmass, single_masses, m1_mass, m2_mass, mmin,
        mmax):
    """
    """
    # Total mass for single systems
    tot_single_mass_1 = totalMass.get(
        Max_mass, sampled_IMF_cmass, single_masses, mmin, mmax)

    # Mass of binary systems in obs range
    binar_mass_sum = sum(m1_mass) + sum(m2_mass)

    # Estimate the *total* binary systems' mass
    # M_fr: Fraction of single systems' mass in the observed range
    M_fr = single_masses.sum() / tot_single_mass_1
    # Assume that the same fraction of binary mass is observed. Hence the
    # observed binary mass can be written as:
    # binar_mass_sum = tot_binar_mass * M_fr
    # From this we can estimate the total binary mass as:
    tot_binar_mass = binar_mass_sum / M_fr

    # Total system's mass
    tot_mass_1 = tot_single_mass_1 + tot_binar_mass

    return tot_mass_1


def method2(Max_mass, sampled_IMF_cmass, comb_masses, mmin, mmax):
    """
    Same as 'method1' but ....
    """
    return totalMass.get(Max_mass, sampled_IMF_cmass, comb_masses, mmin, mmax)


def method3(N_IMF_samp, mass_values, IMF_int, comb_masses, mmin, mmax):
    """
    """
    # Percentage of mass up to the minimum observed mas
    idx = np.argmin(abs(mmin - mass_values))
    perc_mass_min = np.cumsum(IMF_int)[idx]
    mmax = np.percentile(comb_masses, 99.)
    idx = np.argmin(abs(mmax - mass_values))
    # Percentage of mass up to the maximum observed mass
    perc_mass_max = np.cumsum(IMF_int)[idx]
    # Percentage of mass in the observed range, given by the IMF
    perc_obs = perc_mass_max - perc_mass_min
    print("(M_below~{:.0f}% , M_obs~{:.0f}% , M_above~{:.0f}%)".format(
        perc_mass_min * 100, perc_obs * 100, (1 - perc_mass_max) * 100))

    # Bootstrap the total mass
    tot_mass = []
    for _ in range(N_IMF_samp):
        # Sample the masses of *all* single systems
        ran_masses = np.random.choice(comb_masses, len(comb_masses))
        # Keep stars within the observed range
        msk = (ran_masses >= mmin) & (ran_masses <= mmax)
        # Mass in the observed range
        mass_obs = ran_masses[msk].sum()
        # Total mass given by a simple rule:
        # if   perc_obs  ----> mass_obs
        # then    1  --------> M_t = mass_ob/perc_obs
        tot_mass.append(mass_obs / perc_obs)

    return np.array(tot_mass)


def avrgMassNmembs(inv_cdf, tot_mass_1, tot_mass_2, tot_mass_3):
    """
    Estimate the average mass and total number of members
    """

    # Combine a third of each list of masses into a single list
    idx1 = np.random.choice(
        range(tot_mass_1.size), int(.3 * (tot_mass_1.size)), replace=False)
    idx2 = np.random.choice(
        range(tot_mass_2.size), int(.3 * (tot_mass_2.size)), replace=False)
    idx3 = np.random.choice(
        range(tot_mass_3.size), int(.3 * (tot_mass_3.size)), replace=False)
    mass_123 = list(tot_mass_1[idx1]) + list(tot_mass_2[idx2])\
        + list(tot_mass_3[idx3])

    # For each mass, sample the IMF and estimate the average mass and number
    # of stars
    avrg_m_N = []
    for m123 in mass_123:
        sampled_IMF = imfFunc.getIMF(m123, inv_cdf)[0]
        avrg_m_N.append([sampled_IMF.mean(), sampled_IMF.size])
    avrg_m_N = np.array(avrg_m_N).T
    avrg_mass, avrg_Nmemb = avrg_m_N[0].mean(), avrg_m_N[1].mean()

    return avrg_mass, avrg_Nmemb


if __name__ == '__main__':
    # Set random seed
    seed = np.random.randint(100000000)
    print("Random seed: {}".format(seed))
    RandomState(MT19937(SeedSequence(seed)))

    out_folder = "out"
    # Create 'output' folder if it does not exist
    Path('./{}'.format(out_folder)).mkdir(parents=True, exist_ok=True)
    main()
