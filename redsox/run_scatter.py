from __future__ import print_function
import numpy as np
import run_tools
import redsox
from astropy.coordinates import SkyCoord
from astropy.table import Table
from marxs.source import FixedPointing
import astropy.units as u

n_photons = 1e4
outpath = '/melkor/d1/guenther/projects/REDSoX/sims/aeff/'

aeff = []
modulation = []

inplane = np.deg2rad([30., 60., 120., 150., 180., 240., 300.]) / 3600 / 2.3545
perpplane = inplane / 3

mission = redsox.redsox

for inp, perp in zip(inplane, perpplane):
    print('### inplane: {0}, perpplane: {1}'.format(inp, perp))
    mission.elements[1].elements[1].inplanescatter = inp
    mission.elements[1].elements[1].perpplanescatter = perp
    aeff.append(run_tools.run_aeff(mission=mission))
    modulation.append(run_tools.run_modulation(mission=mission))

out = Table([inplane, perpplane, aeff, modulation],
                 names=['inplane', 'perpplane', 'Aeff', 'modulation'])
out.write(outpath + 'mirrorscatter.fits', overwrite=True)
