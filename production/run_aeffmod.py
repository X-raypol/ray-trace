import numpy as np

import run_tools
from astropy.table import Table
import astropy.units as u

import sys
sys.path.append('..')
from redsox.pisox import PerfectPisox



wave = np.arange(17., 61., 1.) * u.Angstrom


n_photons = 1e5
outpath = '../run_results/pisox/'

aeff = []
modulation = []

mission = PerfectPisox()

modulation.append(run_tools.run_modulation(n_photons=n_photons, wave=wave,
                                           mission=mission))

aeff.append(run_tools.run_aeff(n_photons=n_photons, wave=wave,
                               mission=mission))

outfixed = Table([aeff, modulation],
                 names=['Aeff', 'modulation'])
outfixed.write(outpath + 'pi_aeff_mod.fits', overwrite=True)
