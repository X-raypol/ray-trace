import os
from mayavi import mlab
import marxs.visualization.mayavi as marxsavi
from settings import figureout, kwargsfig

import numpy as np

from astropy.coordinates import SkyCoord
import marxs.visualization.mayavi
from marxs.source import PointSource, FixedPointing
from marxs.visualization.utils import format_saved_positions


import sys
sys.path.append('../redsox')
import redsox
from mirror import Ageom

%matplotlib


my_source = PointSource(coords=SkyCoord(30., 30., unit='deg'), energy=0.25,
                        polarization=120.,
                        geomarea=Ageom)
my_pointing = FixedPointing(coords=SkyCoord(30., 30., unit='deg'),
                            reference_transform=redsox.xyz2zxy)

photons = my_source.generate_photons(1)
photons = my_pointing(photons)

photons = redsox.redsox(photons)


fig = mlab.figure(**kwargsfig)
out = marxsavi.plot_object(redsox.redsox, viewer=fig)

pos = format_saved_positions(redsox.keeppos)
ind = np.isfinite(photons['order']) & (photons['grating_id'] < 5000)
out = marxs.visualization.mayavi.plot_rays(pos[ind, :, :],
                                           scalar=photons['order'][ind],
                                           viewer=fig)

mlab.view(azimuth=44, elevation=60, distance=2500, focalpoint=[75, -72, 1700], roll=180)
fig.scene.save(os.path.join(figureout, 'overview_mayavi.pdf'))
