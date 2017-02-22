from __future__ import print_function
import numpy as np
import run_tools
import redsox
from astropy.coordinates import SkyCoord
from astropy.table import Table
from marxs.source import FixedPointing, JitterPointing
import astropy.units as u

n_photons = 1e4
outpath = '/melkor/d1/guenther/projects/REDSoX/sims/aeff/'

aeff = []
modulation = []

ra = 30. + np.array([0., 0,0, 0, 0, 0, 5, 10, 20, 40, 80, 10, 20, 40]) / 3600
dec = 30. + np.array([0., 5, 10, 20, 40, 80, 0, 0, 0, 0, 0, 10, 20, 40]) / 3600

for r, d in zip(ra, dec):
    print('### ra: {0}, dec: {1}'.format(r, d))
    p = FixedPointing(coords=SkyCoord(r, d, unit='deg'),
                      reference_transform=redsox.xyz2zxy)
    aeff.append(run_tools.run_aeff(pointing=p))
    modulation.append(run_tools.run_modulation(pointing=p))

outfixed = Table([ra, dec, aeff, modulation],
                 names=['ra', 'dec', 'Aeff', 'modulation'])
outfixed.write(outpath + 'pointing_fixed.fits', overwrite=True)

aeff = []
modulation = []

jitter = np.array([0.1, 5, 10., 15., 30., 50., 70, 120.]) * u.arcsec

for j in jitter:
    print('### jitter: {0}'.format(j))
    p = JitterPointing(coords=SkyCoord(30., 30., unit='deg'),
                       reference_transform=redsox.xyz2zxy,
                       jitter=j)
    aeff.append(run_tools.run_aeff(pointing=p))
    modulation.append(run_tools.run_modulation(pointing=p))


outjitter = Table([jitter, aeff, modulation],
                  names=['jitter', 'Aeff', 'modulation'])
outjitter.write(outpath + 'pointing_jitter.fits', overwrite=True)
