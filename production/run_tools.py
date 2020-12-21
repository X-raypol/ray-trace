import os
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import marxs
from marxs.source import PointSource, FixedPointing
import astropy.table

import sys
sys.path.append('..')

from redsox import redsox
from redsox.analysis import fractional_aeff, calculate_modulation

wave = np.arange(25., 75., 1.) * u.Angstrom
mypointing = FixedPointing(coords=SkyCoord(30, 30., unit='deg'),
                           reference_transform=redsox.xyz2zxy)


def run_aeff(mission, n_photons=10000, outpath=None,
             pointing=mypointing, wave=wave, orders=['all']):
    '''

    Parameters
    ----------
    n_photons : int
         Number of photons for each simulation
    outpath : string or ``None``.
        Path to an existing directory where ray files will be saved.
        Set to ``None`` if not files shall be written.
    mission : marxs optical elements
        Total mission description.``
    '''
    energies = wave.to(u.keV, equivalencies=u.spectral())
    frac_aeff = np.zeros((len(energies), 4, len(orders)))

    for i, e in enumerate(energies):
        print('{0}/{1}'.format(i + 1, len(energies)))
        mysource = PointSource(coords=SkyCoord(30., 30., unit='deg'),
                               energy=e)

        photons = mysource.generate_photons(n_photons * u.s)
        photons = pointing(photons)
        photons = mission(photons)
        if outpath is not None:
            photons.write(os.path.join(outpath,
                                       'aeff{0:05.2f}.fits'.format(wave.value[i])),
                          overwrite=True)
        for j, order in enumerate(orders):
            if order == 'all':
                filterfunc = None
            else:
                filterfunc = lambda photons: photons['order'] == order
            frac_aeff[i, :, j] = fractional_aeff(photons, filterfunc=filterfunc)
    return frac_aeff


def run_modulation(mission, n_photons=10000, outpath=None,
                   pointing=mypointing, wave=wave, orders=['all']):
    energies = wave.to(u.keV, equivalencies=u.spectral())
    modulation = np.zeros((len(energies), 4, len(orders)))
    for i, e in enumerate(energies):
        print('{0}/{1}'.format(i + 1, len(energies)))
        mysource = PointSource(coords=SkyCoord(30., 30., unit='deg'),
                               energy=e, polarization=0. * u.rad)
        mysource2 = PointSource(coords=SkyCoord(30., 30., unit='deg'),
                                energy=e, polarization=np.pi/2 * u.rad)

        p1 = mysource.generate_photons(n_photons * u.s)
        p2 = mysource2.generate_photons(n_photons * u.s)
        photons = astropy.table.vstack([p1, p2])
        photons = pointing(photons)
        photons = mission(photons)
        if outpath is not None:
            photons.write(os.path.join(outpath,
                                       'merrit{0:05.2f}.fits'.format(wave.value[i])),
                          overwrite=True)
        for j, order in enumerate(orders):
            if order == 'all':
                phot = photons
            else:
                phot = photons[photons['order'] == order]

            modulation[i, :, j] = calculate_modulation(phot)
    return modulation
