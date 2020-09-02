import run_tools
from astropy.table import Table
import astropy.units as u

import sys
sys.path.append('..')
from redsox.gosox import PerfectGosox

import numpy as np
import astropy.units as u

wave = np.arange(17., 61., 1.) * u.Angstrom


n_photons = 1e5
outpath = '../data/aeff'

aeff = []
modulation = []

mission = PerfectGosox()

modulation.append(run_tools.run_modulation(n_photons=n_photons, wave=wave,
                                           mission=mission))

aeff.append(run_tools.run_aeff(n_photons=n_photons, wave=wave,
                               mission=mission))

outfixed = Table([aeff, modulation],
                 names=['Aeff', 'modulation'])
outfixed.write(outpath + 'go_aeff_mod.fits', overwrite=True)
