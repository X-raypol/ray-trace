import argparse
import numpy as np

import run_tools
from astropy.table import Table
import astropy.units as u

import sys
sys.path.append('..')


wave = np.arange(17., 61., 1.) * u.Angstrom


parser = argparse.ArgumentParser(description='Run ray-tracing to determine effective area and modulation factor')
parser.add_argument('mission', choices=['redsox', 'pisox'], help='Select mission')
parser.add_argument('--n_photons', default=100000, type=int, help='Number of photons per simulation (default 100,000')
args = parser.parse_args()


aeff = []
modulation = []

if args.mission == 'pisox':
    from redsox.pisox import PerfectPisox
    mission = PerfectPisox()
elif args.mission == 'redsox':
    from redsox.redsox import PerfectRedsox
    mission = PerfectRedsox()

modulation.append(run_tools.run_modulation(n_photons=args.n_photons, wave=wave,
                                           mission=mission))

aeff.append(run_tools.run_aeff(n_photons=args.n_photons, wave=wave,
                               mission=mission))

outfixed = Table([[wave.T], aeff, modulation],
                 names=['wavelength', 'Aeff', 'modulation'])
outfixed.write(f'../run_results/{args.mission}_aeff_mod.fits', overwrite=True)
