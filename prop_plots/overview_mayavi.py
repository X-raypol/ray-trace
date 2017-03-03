import os
from mayavi import mlab
import marxs.visualization.mayavi as marxsavi
from settings import figureout, kwargsfig
import copy

import numpy as np

from astropy.coordinates import SkyCoord
import marxs.visualization.mayavi
from marxs.source import PointSource, FixedPointing
from marxs.visualization.utils import format_saved_positions
from marxs.optics import FlatOpticalElement, FlatDetector

import sys
sys.path.append('../redsox')
import redsox
from mirror import Ageom
from mlmirrors import LGMLMirror

%matplotlib
redsox.grat_args_full['elem_args']['zoom'] = [.5, 15., 5.]
grat1 = redsox.GratingGrid(id_num_offset=1000, zoom_grating=[15., 5.],
                    **redsox.grat_args_full)
grat_pos2 = [np.dot(redsox.rotchan2, e.pos4d) for e in grat1.elements]
grat_pos3 = [np.dot(redsox.rotchan3, e.pos4d) for e in grat1.elements]
grat2 = redsox.simulator.Parallel(elem_pos=grat_pos2, id_num_offset=2000,
                                  **redsox.grat_args)
grat3 = redsox.simulator.Parallel(elem_pos=grat_pos3, id_num_offset=3000,
                                  **redsox.grat_args)

grat = redsox.simulator.Sequence(elements=[grat1, grat2, grat3, redsox.catsupport,
                                    redsox.gratquality, redsox.CATSupportbars()])
# Monkey patch redsox.redsox with 30 mm * 10 mm gratings
redsox.redsox.elements[2] = grat

my_source = PointSource(coords=SkyCoord(30., 30., unit='deg'), energy=0.25,
                        polarization=120.,
                        geomarea=Ageom)
my_pointing = FixedPointing(coords=SkyCoord(30., 30., unit='deg'),
                            reference_transform=redsox.xyz2zxy)

photons = my_source.generate_photons(2.)
photons = my_pointing(photons)

photons = redsox.redsox(photons)


fig = mlab.figure(**kwargsfig)

FlatOpticalElement.display['color'] = (1., .84, .0)
# Don't remember why that does overwrite display ...
# In any case, need to set here separately
LGMLMirror.display['color'] = (1., 0., 1.)
redsox.det0.display = copy.deepcopy(redsox.det0.display)
redsox.det0.display['color'] = (0., 0., 1.)
for e in redsox.det123.elements:
    e.display = copy.deepcopy(e.display)
    e.display['color'] = (0., 0., 1.)

# Do not plot aperture and mirror
for e in redsox.redsox.elements[2:]:
    out = marxsavi.plot_object(e, viewer=fig)

pos_full = format_saved_positions(redsox.keeppos)[:, :, :]
ind = np.isfinite(photons['order']) & (photons['grating_id'] < 2000)
pos = np.empty_like(pos_full)
pos[:, 0:2, :] = pos_full[:, 1:3, :]
pos[:, 2:, :] = pos_full[:, 2:, :]
col = np.zeros_like(pos[:, :, 0])
col[:, 2:] = photons['order'][:, None]
# some tricks to color photons red that are 1st order and then pass through a second overlapping
# grating with no direction change (order 0)
col[ind, 2:] = (photons['CCD_ID'][ind, None] > 0)
col = 0.5 * col  # 0.5 is red in this color map
kwargssurface = {'opacity': .6, 'line_width': 1, 'colormap': 'gist_heat'}

out = marxs.visualization.mayavi.plot_rays(pos[ind, :, :],
                                           scalar=col[ind, :],
                                           kwargssurface=kwargssurface,
                                           viewer=fig)

mlab.view(azimuth=-140, elevation=20, distance=1000, focalpoint=[12, 14, 1500], roll=120)
fig.scene.save(os.path.join(figureout, 'overview_mayavi.pdf'))

mlab.view(azimuth=30, elevation=65, distance=1000, focalpoint=[-24, 6, 1580], roll=-35)
fig.scene.save(os.path.join(figureout, 'overview_mayavi_side.pdf'))


fig = mlab.figure(**kwargsfig)
# Do not plot aperture and mirror
out = marxsavi.plot_object(redsox.ml, viewer=fig)
out = marxsavi.plot_object(redsox.det, viewer=fig)

ind = np.isfinite(photons['order']) & (photons['grating_id'] < 5000)
out = marxs.visualization.mayavi.plot_rays(pos[ind, 2:, :],
                                           scalar=col[ind, 2:],
                                           kwargssurface=kwargssurface,
                                           viewer=fig)

mlab.view(azimuth=-12, elevation=55, distance=140, focalpoint=[10,10,-10], roll=-88)
fig.scene.save(os.path.join(figureout, 'lgml_mayavi.pdf'))
