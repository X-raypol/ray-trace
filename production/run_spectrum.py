import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.utils.metadata import enable_merge_strategies
from marxs import utils
from marxs.source import PointSource

import redsox
from mirror import Ageom
from run_tools import wave, energies, mypointing

# spectrum = {'energy': energies[::-1],
#            'flux': 0.02 * energies[::-1]**(-2)}

spectrum = Table.read('/melkor/d1/guenther/Dropbox/REDSoX File Transfers/raytrace/inputdata/mk421_spec.txt', format='ascii.no_header',
                      names=['wave','fluxperwave'])
spectrum['energy'] = 1.2398419292004202e-06 / (spectrum['wave'] * 1e-7)
spectrum['flux'] = spectrum['fluxperwave'] / 12.398419292004202 * spectrum['wave']**2
spectrum.sort('energy')
# Now limit to the range where I have coefficients for gratings etc.
spectrum = spectrum[(spectrum['wave'] > 25.) & (spectrum['wave'] < 75.)]
flux = np.sum(spectrum['flux'][1:] * np.diff(spectrum['energy']))
my_sourcepol = PointSource(coords=SkyCoord(30., 30., unit='deg'),
                           energy=spectrum,
                           flux=0.2 * flux,
                           polarization=120.,
                           geomarea=Ageom)
my_sourceunpol = PointSource(coords=SkyCoord(30., 30., unit='deg'),
                             energy=spectrum,
                             flux=0.8 * flux,
                             geomarea=Ageom)
ppol = my_sourcepol.generate_photons(300)
punpol = my_sourceunpol.generate_photons(300)
with enable_merge_strategies(utils.MergeIdentical):
    photons = vstack([ppol, punpol])

photons = mypointing(photons)
len(photons)

photons = redsox.redsox(photons)
photons.write('/melkor/d1/guenther/projects/REDSoX/sims/photons_spectrum.fits', overwrite=True)
