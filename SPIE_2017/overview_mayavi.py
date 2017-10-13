import os
import copy
from mayavi import mlab
import marxs.visualization.mayavi as marxsavi
from settings import figureout, kwargsfig

import numpy as np

from astropy.coordinates import SkyCoord
import marxs.visualization.mayavi
from marxs.source import PointSource, FixedPointing


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
redsox.mirror.display = copy.deepcopy(redsox.mirror.display)
redsox.mirror.display['color'] = (1., 0.6, 0.)
out = marxsavi.plot_object(redsox.redsox, viewer=fig)

pos_full = redsox.keeppos.format_positions()
ind = np.isfinite(photons['order']) & (photons['grating_id'] < 5000)
pos = np.empty((pos_full.shape[0], pos_full.shape[1] + 1, pos_full.shape[2]))
pos[:, 0:3, :] = pos_full[:, 0:3, :]
pos[:, 3:, :] = pos_full[:, 2:, :]
col = np.zeros_like(pos[:, :, 0])
col[:, 3:] = photons['order'][:, None]
# some tricks to color photons red that are 1st order and then pass through a second overlapping
# grating with no direction change (order 0)
col[ind, 3:] = (photons['CCD_ID'][ind, None] > 0)
col = 0.5 * col  # 0.5 is red in this color map
kwargssurface = {'opacity': .6, 'line_width': 1, 'colormap': 'summer'}

out = marxs.visualization.mayavi.plot_rays(pos[ind, :, :],
                                           scalar=col[ind, :],
                                           kwargssurface=kwargssurface,
                                           viewer=fig)


mlab.view(azimuth=44, elevation=60, distance=2500, focalpoint=[75, -72, 1700], roll=180)
fig.scene.save(os.path.join(figureout, 'overview_mayavi.pdf'))


mlab.view(azimuth=6, elevation=66, distance=84, focalpoint=[40, 5, 6])
fig.scene.save(os.path.join(figureout, 'lgml_mayavi.pdf'))

fig.scene.save(os.path.join(figureout, 'talk', 'REDSoX.x3d'))
