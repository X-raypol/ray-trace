import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u
from mayavi import mlab
import marxs.visualization.mayavi
from marxs.source import PointSource, FixedPointing
from marxs.visualization.utils import format_saved_positions

import redsox

%matplotlib

fig = mlab.figure()
mlab.clf()
redsox.redsox.plot(format='mayavi', viewer=fig)

angles = np.array([0., 1.5, 1.7, 2 * np.pi])
prob = np.array([1, 1., 8., 1.])
my_source = PointSource(coords=SkyCoord(30., 30., unit='deg'), energy=0.25,
                        polarization={'angle': angles, 'probability': prob})
my_source = PointSource(coords=SkyCoord(30., 30., unit='deg'), energy=0.25,
                        polarization=120.)
my_pointing = FixedPointing(coords=SkyCoord(30., 30., unit='deg'),
                            reference_transform=redsox.xyz2zxy)

photons = my_source.generate_photons(100000)
photons = my_pointing(photons)

photons = redsox.redsox(photons)

pos = format_saved_positions(redsox.keeppos)
ind = np.isfinite(photons['order']) & (photons['grating_id'] < 200)
marxs.visualization.mayavi.plot_rays(pos[ind, :, :], scalar=photons['order'][ind], viewer=fig)
