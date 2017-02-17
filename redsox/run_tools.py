import os
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from marxs.source import PointSource, FixedPointing
import astropy.table

import redsox
from analysis import fractional_aeff, calculate_modulation

wave = np.arange(25., 75., 1.) * u.Angstrom
energies = wave.to(u.keV, equivalencies=u.spectral()).value
mypointing = FixedPointing(coords=SkyCoord(30, 30., unit='deg'),
                           reference_transform=redsox.xyz2zxy)


def run_aeff(n_photons, outpath, mission):
    '''

    Parameters
    ----------
    n_photons : int
         Number of photons fir each simulation
    outpath : string or ``None``.
        Path to an existing directory where ray files will be saved.
        Set to ``None`` if not files shall be written.
    mission : marxs optical elements
        Total mission description. Typically ``redsox.redsox``
    '''
    frac_aeff = np.zeros((len(energies), 4))
    for i, e in enumerate(energies):
        print '{0}/{1}'.format(i + 1, len(energies))
        mysource = PointSource(coords=SkyCoord(30., 30., unit='deg'),
                               energy=e, flux=1.)

        photons = mysource.generate_photons(n_photons)
        photons = mypointing(photons)
        photons = mission(photons)
        if outpath is not None:
            photons.write(os.path.join(outpath,
                                       'aeff{0:05.2f}.fits'.format(wave.value[i])),
                          overwrite=True)
        frac_aeff[i, :] = fractional_aeff(photons)
    return frac_aeff


def run_modulation(n_photons, outpath, mission):

    modulation = np.zeros((len(energies), 4))
    for i, e in enumerate(energies):
        print '{0}/{1}'.format(i + 1, len(energies))
        mysource = PointSource(coords=SkyCoord(30., 30., unit='deg'),
                               energy=e, flux=1., polarization=0.)
        mysource2 = PointSource(coords=SkyCoord(30., 30., unit='deg'),
                               energy=e, flux=1., polarization=90.)

        p1 = mysource.generate_photons(n_photons)
        p2 = mysource2.generate_photons(n_photons)
        photons = astropy.table.vstack([p1, p2])
        photons = mypointing(photons)
        photons = mission(photons)
        if outpath is not None:
            photons.write(os.path.join(outpath,
                                       'merrit{0:05.2f}.fits'.format(wave.value[i])),
                          overwrite=True)

        modulation[i, :] = calculate_modulation(photons)
        return modulation
