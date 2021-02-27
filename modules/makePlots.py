
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.visualization import hist


def CMD(
    thresh, met, age, ext, dist, clust_name, cluster, isoch_phot,
        single_msk, binar_msk, b_fr, single_masses, m1_mass, m2_mass):
    """
    """
    fig = plt.figure(figsize=(16, 16))
    GS = gridspec.GridSpec(2, 2)

    plt.subplot(GS[0])
    plt.title("Binary distance threshold: {:.3f}".format(thresh))
    plt.plot(
        isoch_phot[1], isoch_phot[0],
        label="({:.5f}, {:.2f}, {:.2f}, {:.2f})".format(met, age, ext, dist))
    plt.scatter(
        cluster[1][single_msk], cluster[0][single_msk], s=20,
        edgecolor='g', facecolor='none', alpha=.5,
        label="Single (N={})".format(single_msk.sum()))
    plt.scatter(
        cluster[1][binar_msk], cluster[0][binar_msk], marker='x', s=25, c='r',
        lw=.8, alpha=.5, label="Binary (N={}) | b_fr={:.2f}".format(
            binar_msk.sum(), b_fr))
    plt.gca().invert_yaxis()
    plt.ylabel(r"$G$")
    plt.xlabel(r"$BP-RP$")
    plt.legend(loc=3)

    ax = plt.subplot(GS[1])
    t1 = 'Single systems {:.0f} '.format(single_masses.sum()) + r"$M_{\odot}$"
    t2 = "\n[{:.2f}, {:.2f}] ".format(
        single_masses.min(), single_masses.max()) + r"$M_{\odot}$"
    hist(single_masses, bins=25, ax=ax, histtype='step',
         alpha=0.7, color='green', label=t1 + t2, lw=2, density=True)
    hist(m1_mass, bins=25, ax=ax, histtype='step',
         alpha=0.7, color='blue', label='Binary systems (M1) {:.0f} '.format(
             sum(m1_mass)) + r"$M_{\odot}$", lw=2, density=True)
    hist(m2_mass, bins=25, ax=ax, histtype='step',
         alpha=0.7, color='red', label='Binary systems (M2) {:.0f} '.format(
             sum(m2_mass)) + r"$M_{\odot}$", ls='-', lw=2, density=True)
    plt.legend(loc=9)

    plt.legend(loc=9)
    ax.set_yscale('log')
    plt.xlabel(r"$Mass\;(M_{\odot})$")

    # ax = plt.subplot(GS[2])


    fig.tight_layout()

    plt.savefig(
        "out/zaed_{}_{}.png".format(clust_name, str(thresh).split('.')[1]),
        dpi=150, bbox_inches='tight')


def final(
    thresh, method, clust_name, cluster, tot_single_mass_all,
    tot_binar_mass_all, total_mass_all, binar_fr_all, m1_all, m2_all, G_binars,
        qvals, Gobs_s, Gobs_b):
    """
    """
    fig = plt.figure(figsize=(20, 20))
    GS = gridspec.GridSpec(3, 3)

    def histplot(ax, data, _mean, bins='knuth'):
        _16, _84 = np.percentile(data, (16, 84))
        # plt.hist(data, 100, color='grey')
        hist(data, bins=bins, ax=ax, histtype='stepfilled', alpha=0.4)
        plt.axvline(_16, c='r', ls=':', label="16p ({:.0f})".format(_16))
        plt.axvline(_84, c='r', ls=':', label="84p ({:.0f})".format(_84))
        plt.axvline(_mean, c='g', ls=':', label="Mean ({:.0f})".format(_mean))
        plt.legend()
        plt.xlabel(r"$Mass\;(M_{\odot})$")

    ax = plt.subplot(GS[0])
    _mean, _std = tot_single_mass_all.mean(), tot_single_mass_all.std()
    plt.title(r"Single systems: ${:.0f}\pm{:.0f}$ (N={})".format(
        _mean, _std, len(tot_single_mass_all)))
    histplot(ax, tot_single_mass_all, _mean)

    if method == '1':
        ax = plt.subplot(GS[1])
        _mean, _std = tot_binar_mass_all.mean(), tot_binar_mass_all.std()
        plt.title(r"Binary systems: ${:.0f}\pm{:.0f}$ (N={})".format(
            _mean, _std, len(tot_binar_mass_all)))
        histplot(ax, tot_binar_mass_all, _mean)

    ax = plt.subplot(GS[2])
    _mean, _std = total_mass_all.mean(), total_mass_all.std()
    plt.title(r"Single+binary systems: ${:.0f}\pm{:.0f}$ (N={})".format(
        _mean, _std, len(total_mass_all)))
    histplot(ax, total_mass_all, _mean)

    ax = plt.subplot(GS[3])
    plt.title("N={}".format(len(binar_fr_all)))
    hist(binar_fr_all, bins='knuth', ax=ax, histtype='stepfilled', alpha=0.4,
         density=True)
    _mean = np.mean(binar_fr_all)
    plt.axvline(_mean, c='g', ls=':', label="Mean ({:.2f})".format(_mean))
    plt.legend()
    plt.xlabel(r"$b_fr$")

    ax = plt.subplot(GS[4])
    q = np.array(m2_all) / np.array(m1_all)
    plt.title("N={}".format(len(q)))
    hist(q, bins=10, ax=ax, histtype='stepfilled', alpha=0.4,
         density=True)
    plt.xlabel(r"$q\;(m2/m1)$")

    ax = plt.subplot(GS[5])
    x_steps, yboxes = boxPlotArray(G_binars, qvals)
    tcks = np.round(x_steps, 2)
    boxPlot(ax, x_steps, yboxes, "ecc", "ecc", tcks)
    plt.xlabel(r"$G$")
    plt.ylabel(r"$q\;(m2/m1)$")

    ax = plt.subplot(GS[7])
    Gobs = cluster[0, :]
    rmin, rmax = min(Gobs), max(Gobs)
    bfr_vals, step = [], 3
    _range = np.arange(rmin, rmax, step)
    for i, ra in enumerate(_range):
        msk_s = (Gobs_s >= ra) & (Gobs_s < _range[i] + step)
        msk_b = (Gobs_b >= ra) & (Gobs_b < _range[i] + step)
        bfr_vals.append(msk_b.sum() / msk_s.sum())
    Ggrid = list(.5 * (_range[1:] + _range[:-1])) + [_range[-1] + .5 * step]

    plt.bar(Ggrid, bfr_vals)
    plt.xlabel(r"$G$")
    plt.ylabel(r"$b_fr$")

    fig.tight_layout()
    plt.savefig(
        "out/{}_{}_{}.png".format(
            clust_name, method, str(thresh).split('.')[1]), dpi=150,
        bbox_inches='tight')


def boxPlotArray(xvals, yvals, step=2):
    rmin, rmax = min(xvals), max(xvals)
    yarr = []
    _range = np.arange(rmin, rmax, step)
    for i, ra in enumerate(_range):
        msk = (xvals >= ra) & (xvals < _range[i] + step)
        yarr.append(yvals[msk])
    grid = list(.5 * (_range[1:] + _range[:-1])) + [_range[-1] + .5 * step]
    return grid, yarr


def boxPlot(ax, xgrid, par_delta, p_n, pID, tcks):
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='gray', linestyle=':', lw=.5)
    plt.boxplot(par_delta, positions=xgrid, widths=1)
    ax.set_xticks(tcks)
    ax.set_xticklabels(tcks)
