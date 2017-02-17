import numpy as np


def fractional_aeff(photons, filterfunc=None):
    '''Calculate the fraction of photons that are detected in a specific CCD

    Parameters
    ----------
    photons : `astropy.table.Table`
        Photon event list
    filterfunc : callable or ``None``
        If not ``None``, a function that takes the photon table and returns an
        index array. This can be used, e.g. filter out photons that hit
        particular CCDs or hot columns.

    Returns
    -------
    prop : np.array
        Probability for a photon to be detected on CCD 0,1,2, or 3
    '''
    prob = np.zeros(4, dtype=float)
    for i in [0, 1, 2, 3]:
        ind = (photons['CCD_ID'] == i)
        if filterfunc is not None:
            ind = ind & filterfunc(photons)
        prob[i] = np.sum(photons['probability'][ind]) / len(photons)
    return prob


def calculate_modulation(photons):
    '''Calculate modulation factor on each CCD

    Parameters
    ----------
    photons : `astropy.table.Table`
        Photon event list. The event list should be split exactly,
        where the top half contains photons of one polarization direction
        and the bottom half photons of the perpendicular direction.

    Returns
    -------
    modulation : np.array
        Modulation factor for photons on CCD 0, 1, 2, 3 calculated by
        comparing photons in the upper half of the photon table to
        photons in the lower half.
    '''
    n = len(photons) // 2
    indhalf = np.ones((len(photons)), dtype=bool)
    indhalf[n:] = False
    modulation = np.zeros(4, dtype=float)

    for i in [0, 1, 2, 3]:
        indccd = (photons['CCD_ID'] == i)
        pol1 = photons['probability'][indhalf & indccd].sum()
        pol2 = photons['probability'][~indhalf & indccd].sum()
        modulation[i] = (pol1 - pol2) / (pol1 + pol2)
    return modulation
