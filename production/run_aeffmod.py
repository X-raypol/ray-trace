import argparse
import numpy as np

import run_tools
from astropy.table import Table
import astropy.units as u

import sys
sys.path.append('..')

# I used 17 - 61 before, not sure why.
wave = np.arange(30., 83., 1.) * u.Angstrom
orders = ['all', 0, -1, -2]

parser = argparse.ArgumentParser(description='Run ray-tracing to determine effective area and modulation factor')
parser.add_argument('mission', choices=['redsox', 'pisox', 'gosox'],
                    help='Select mission')
parser.add_argument('--n_photons', default=100000, type=int,
                    help='Number of photons per simulation (default 100,000')
args = parser.parse_args()


if args.mission == 'pisox':
    from redsox.pisox import PerfectPisox
    mission = PerfectPisox()
elif args.mission == 'redsox':
    from redsox.redsox import PerfectRedsox
    mission = PerfectRedsox()
elif args.mission == 'gosox':
    from redsox.gosox import PerfectGosox
    mission = PerfectGosox()

modulation = run_tools.run_modulation(n_photons=args.n_photons, wave=wave,
                                      mission=mission,
                                      orders=orders)

aeff = run_tools.run_aeff(n_photons=args.n_photons, wave=wave,
                          mission=mission,
                          orders=orders)

outfixed = Table([[wave.T] * aeff.shape[-1],
                  [aeff[:, :, j] for j in range(aeff.shape[-1])],
                  [modulation[:, :, j] for j in range(aeff.shape[-1])],
                  orders],
                 names=['wavelength', 'Aeff', 'modulation', 'order'])
outfixed.write(f'../run_results/{args.mission}_aeff_mod.fits', overwrite=True)
