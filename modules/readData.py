
from astropy.io import ascii
import numpy as np


def loadClust(clust_name, max_error=0.3):
    """
    Column names are HARDCODED
    """
    cluster = ascii.read("input/{}.dat".format(clust_name))
    print("{} total stars".format(len(cluster)))
    msk = cluster['BP-RP'].mask
    msk2 = (cluster['e_Gmag'] >= max_error) | (cluster['e_BP-RP'] >= max_error)
    msk = msk | msk2.data
    print("{} masked stars".format(msk.sum()))
    cluster = cluster[~msk]
    cluster = np.array([cluster['Gmag'], cluster['BP-RP']])
    print("{} final members".format(cluster.shape[-1]))
    return cluster


def loadIsoch(idx_header, met, age):
    """
    HARDCODED to work with Gaia data <-- IMPORTANT
    """
    # Load CMD isochrone (remove '#' from columns' names) IMPORTANT
    isoch = ascii.read("./isochs/{}.dat".format(
        str(met).replace(".", "_")), header_start=idx_header)

    msk = isoch['logAge'] == age
    isoch = isoch[msk]

    # The label '2' marks the end of the MS, i.e. the turn-off point
    idx = np.searchsorted(isoch['label'], 2)
    turn_off = isoch['Gmag'][idx]

    return isoch, turn_off


# def lowerEnvelope(cluster, step=.05, perc=70):
#     xx_yy = []
#     for low in np.arange(-0.1, 3, step):
#         msk = (cluster[1] > low) & (cluster[1] <= low + step)
#         if msk.sum() > 0:
#             xx_yy.append([low, np.percentile(cluster[0][msk], perc)])

#     # plt.plot(memb_d['BP-RP'], memb_d['Gmag'], '.')
#     # plt.plot(np.array(xx_yy).T[0], np.array(xx_yy).T[1])
#     # plt.gca().invert_yaxis()
#     # plt.show()
#     return np.array(xx_yy)
