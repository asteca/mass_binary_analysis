
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.visualization import hist


def final(
    cluster_name, cluster, low_env_perc, envelopes, single_msk, binar_msk,
        single_masses, m1_mass, m2_mass, tot_mass_1, tot_mass_2, tot_mass_3):
    """
    """
    fig = plt.figure(figsize=(12, 12))
    GS = gridspec.GridSpec(2, 2) 

    ax = plt.subplot(GS[0])
    ax.minorticks_on()
    plt.plot(envelopes[0][1], envelopes[0][0], ls='-.', c='k',
             label="Lower envelope ({:.0f})".format(low_env_perc))
    plt.plot(envelopes[1][1], envelopes[1][0], ls='--', c='b',
             label="Binary envelope")

    plt.scatter(
        cluster[1][single_msk], cluster[0][single_msk], s=20,
        edgecolor='g', facecolor='none', alpha=.5,
        label="Single (N={})".format(single_msk.sum()))
    b_fr = binar_msk.sum() / (binar_msk.sum() + single_msk.sum())
    plt.scatter(
        cluster[1][binar_msk], cluster[0][binar_msk], marker='x', s=25, c='r',
        lw=.8, alpha=.5, label="Binary (N={}) | b_fr={:.2f}".format(
            binar_msk.sum(), b_fr))
    plt.gca().invert_yaxis()
    plt.grid()
    plt.ylabel(r"$G$")
    plt.xlabel(r"$BP-RP$")
    plt.legend(loc=3)

    ax = plt.subplot(GS[1])
    q = m2_mass / m1_mass
    plt.title("N={}".format(len(q)))
    hist(q, bins=10, ax=ax, histtype='stepfilled', alpha=0.4,
         density=True)
    plt.xlabel(r"$q\;(m2/m1)$")
    plt.xlim(-0.02, 1.02)

    ax = plt.subplot(GS[2])
    # TODO this result depends on the binning
    Nb, m1_edges = np.histogram(m1_mass, 15)
    mass_intervals, binar_frac, m0 = [], [], m1_edges[0]
    for i, m1 in enumerate(m1_edges[1:]):
        msk = (single_masses > m0) & (single_masses <= m1)
        bf = Nb[i] / (msk.sum() + Nb[i])
        if bf > 0.:
            mass_intervals.append(.5 * (m0 + m1))
            binar_frac.append(min(bf, 1))
        m0 = 1. * m1
    plt.scatter(mass_intervals, binar_frac)
    plt.ylim(0., 1.04)
    plt.xlabel("M1")
    plt.ylabel("Binary fraction")

    ax = plt.subplot(GS[3])
    tot_mass = np.array(list(tot_mass_1) + list(tot_mass_2) + list(tot_mass_3))
    _mean, _std = tot_mass.mean(), tot_mass.std()
    plt.title(r"Single+binary systems: ${:.0f}\pm{:.0f}$".format(_mean, _std))
    _16, _84 = np.percentile(tot_mass, (16, 84))
    plt.axvline(_16, c='k', ls=':', lw=3)
    plt.axvline(_84, c='k', ls=':', lw=3)
    plt.axvline(_mean, c='k', ls='-', lw=3)
    # M1
    _mean, _std = tot_mass_1.mean(), tot_mass_1.std()
    hist(tot_mass_1, bins='knuth', ax=ax, histtype='stepfilled', alpha=0.2,
         color='g', density=True,
         label=r"M1: ${:.0f}\pm{:.0f}$".format(_mean, _std))
    # M2
    _mean, _std = tot_mass_2.mean(), tot_mass_2.std()
    hist(tot_mass_2, bins='knuth', ax=ax, histtype='stepfilled', alpha=0.2,
         color='r', density=True,
         label=r"M2: ${:.0f}\pm{:.0f}$".format(_mean, _std))
    # M3
    _mean, _std = tot_mass_3.mean(), tot_mass_3.std()
    hist(tot_mass_3, bins='knuth', ax=ax, histtype='stepfilled', alpha=0.2,
         color='b', density=True,
         label=r"M3: ${:.0f}\pm{:.0f}$".format(_mean, _std))
    plt.legend()
    plt.xlabel(r"$Mass\;(M_{\odot})$")

    fig.tight_layout()
    plt.savefig("out/{}.png".format(cluster_name), dpi=150, bbox_inches='tight')

    # ax = plt.subplot(GS[0])
    # plt.title("N={}".format(len(binar_fr_all)))
    # hist(binar_fr_all, bins=20, ax=ax, histtype='stepfilled', alpha=0.4,
    #      density=True)
    # _mean, _std = np.mean(binar_fr_all), np.std(binar_fr_all)
    # _16, _84 = np.percentile(binar_fr_all, (16, 84))
    # plt.axvline(_mean, c='g', ls=':', label="Mean ({:.2f} +/- {:.2f})".format(
    #     _mean, _std))
    # plt.axvline(_16, c='r', ls=':', label="16p ({:.2f})".format(_16))
    # plt.axvline(_84, c='r', ls=':', label="84p ({:.2f})".format(_84))
    # plt.legend()
    # plt.xlabel(r"$b_fr$")

    # ax = plt.subplot(GS[1])
    # # hist(binar_probs, bins='knuth', ax=ax, histtype='stepfilled', alpha=0.4,
    # #      density=True)
    # plt.hist(binar_probs, histtype='stepfilled', alpha=0.4, density=True)
    # plt.axvline(binar_P_thresh, c='r', ls=':', label="P_thresh= {:.2f}".format(
    #     binar_P_thresh))
    # plt.xlabel(r"$P_{binar}$")
    # plt.legend()

    # ax = plt.subplot(GS[1])
    # t1 = 'Single systems {:.0f} '.format(single_masses.sum()) + r"$M_{\odot}$"
    # t2 = "\n[{:.2f}, {:.2f}] ".format(
    #     single_masses.min(), single_masses.max()) + r"$M_{\odot}$"
    # hist(single_masses, bins=25, ax=ax, histtype='step',
    #      alpha=0.7, color='green', label=t1 + t2, lw=2, density=True)
    # hist(m1_mass, bins=25, ax=ax, histtype='step',
    #      alpha=0.7, color='blue', label='Binary systems (M1) {:.0f} '.format(
    #          sum(m1_mass)) + r"$M_{\odot}$", lw=2, density=True)
    # hist(m2_mass, bins=25, ax=ax, histtype='step',
    #      alpha=0.7, color='red', label='Binary systems (M2) {:.0f} '.format(
    #          sum(m2_mass)) + r"$M_{\odot}$", ls='-', lw=2, density=True)
    # plt.legend(loc=9)
    # ax.set_yscale('log')
    # plt.xlabel(r"$Mass\;(M_{\odot})$")

    # def histplot(ax, data, _mean, bins='knuth'):
    #     _16, _84 = np.percentile(data, (16, 84))
    #     # plt.hist(data, 100, color='grey')
    #     hist(data, bins=bins, ax=ax, histtype='stepfilled', alpha=0.4)
    #     plt.axvline(_16, c='r', ls=':', label="16p ({:.0f})".format(_16))
    #     plt.axvline(_84, c='r', ls=':', label="84p ({:.0f})".format(_84))
    #     plt.axvline(_mean, c='g', ls=':', label="Mean ({:.0f})".format(_mean))

    # ax = plt.subplot(GS[6])
    # _mean, _std = np.mean(tot_single_mass_1), np.mean(tot_single_mass_1)
    # plt.title(r"Single systems: ${:.0f}\pm{:.0f}$ (N={})".format(
    #     _mean, _std, len(tot_single_mass_1)))
    # histplot(ax, tot_single_mass_1, _mean)

    # ax = plt.subplot(GS[7])
    # _mean, _std = np.mean(tot_binar_mass), np.mean(tot_binar_mass)
    # plt.title(r"Binary systems: ${:.0f}\pm{:.0f}$ (N={})".format(
    #     _mean, _std, len(tot_binar_mass)))
    # histplot(ax, tot_binar_mass, _mean)

    # ax = plt.subplot(GS[8])
    # total_mass = tot_single_mass_1 + tot_binar_mass
    # _mean, _std = np.mean(total_mass), np.mean(total_mass)
    # plt.title(r"Single+binary systems (1): ${:.0f}\pm{:.0f}$ (N={})".format(
    #     _mean, _std, len(total_mass)))
    # histplot(ax, total_mass, _mean)

    # ax = plt.subplot(GS[11])
    # total_mass = tot_single_mass_3
    # _mean, _std = np.mean(total_mass), np.mean(total_mass)
    # plt.title(r"Single+binary systems (3): ${:.0f}\pm{:.0f}$ (N={})".format(
    #     _mean, _std, len(total_mass)))
    # histplot(ax, total_mass, _mean)

    # ax = plt.subplot(GS[5])
    # x_steps, yboxes = boxPlotArray(G_binars, qvals)
    # tcks = np.round(x_steps, 2)
    # boxPlot(ax, x_steps, yboxes, "ecc", "ecc", tcks)
    # plt.xlabel(r"$G$")
    # plt.ylabel(r"$q\;(m2/m1)$")

    # ax = plt.subplot(GS[7])
    # Gobs = cluster[0, :]
    # rmin, rmax = min(Gobs), max(Gobs)
    # bfr_vals, step = [], 3
    # _range = np.arange(rmin, rmax, step)
    # for i, ra in enumerate(_range):
    #     msk_s = (Gobs_s >= ra) & (Gobs_s < _range[i] + step)
    #     msk_b = (Gobs_b >= ra) & (Gobs_b < _range[i] + step)
    #     bfr_vals.append(msk_b.sum() / msk_s.sum())
    # Ggrid = list(.5 * (_range[1:] + _range[:-1])) + [_range[-1] + .5 * step]

    # plt.bar(Ggrid, bfr_vals)
    # plt.xlabel(r"$G$")
    # plt.ylabel(r"$b_fr$")


# def boxPlotArray(xvals, yvals, step=2):
#     rmin, rmax = min(xvals), max(xvals)
#     yarr = []
#     _range = np.arange(rmin, rmax, step)
#     for i, ra in enumerate(_range):
#         msk = (xvals >= ra) & (xvals < _range[i] + step)
#         yarr.append(yvals[msk])
#     grid = list(.5 * (_range[1:] + _range[:-1])) + [_range[-1] + .5 * step]
#     return grid, yarr


# def boxPlot(ax, xgrid, par_delta, p_n, pID, tcks):
#     ax.minorticks_on()
#     ax.grid(b=True, which='major', color='gray', linestyle=':', lw=.5)
#     plt.boxplot(par_delta, positions=xgrid, widths=1)
#     ax.set_xticks(tcks)
#     ax.set_xticklabels(tcks)
