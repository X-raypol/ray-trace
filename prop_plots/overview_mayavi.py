import os
from mayavi import mlab
import marxs.visualization.mayavi as marxsavi
from settings import figureout, kwargsfig
import copy

import numpy as np

from astropy.coordinates import SkyCoord
import marxs.visualization.mayavi
from marxs.source import PointSource, FixedPointing
from marxs.optics import FlatOpticalElement, FlatDetector

import sys
sys.path.append('../redsox')
import redsox


%matplotlib
conf = copy.deepcopy(redsox.conf)
conf['gratingzoom'] = [.5, 15., 5.]
instrum = redsox.PerfectRedsox(conf=conf)

instrum3 = redsox.PerfectRedsox(conf=conf, channels='3')
my_source = PointSource(coords=SkyCoord(30., 30., unit='deg'), energy=0.25,
                        polarization=120.,
                        geomarea=instrum.elements[0].area)
my_pointing = FixedPointing(coords=SkyCoord(30., 30., unit='deg'),
                            reference_transform=redsox.xyz2zxy)

photons = my_source.generate_photons(.2)
photons = my_pointing(photons)
photons = instrum3(photons)


ind = (photons['facet'] >= 0) & (photons['CCD_ID'] >= 0)
positions = [instrum3.KeepPos.format_positions()[ind, :, :]]
colorid = [photons['colorindex'][ind]]
# saving to x3d resets color scale form in to max, ignoring vmin and vmax
# Thus, add a non-visible line here with color 5 to prevent that
positions.append(np.zeros((2, 5, 3)))
colorid.append([0, 5])


fig = mlab.figure(**kwargsfig)
out = marxsavi.plot_object(instrum, viewer=fig)
out = marxs.visualization.mayavi.plot_rays(np.concatenate(positions, axis=0),
                                           scalar=np.hstack(colorid),
                                           kwargssurface={'opacity': .5,
                                                          'line_width': 1,
                                                          'colormap': 'gist_rainbow',
                                                          'vmin': 0,
                                                          'vmax': 5})
# overview
mlab.view(100, 60, 2800, [20, -200, 1800], roll=225)
fig.scene.save('../JATIS/mayavi_overview.png')
fig.scene.save('../JATIS/mayavi_overview.pdf')
fig.scene.save('../JATIS/web/REDSoX.x3d')

# gratings
mlab.view(60, 60, 900, [-30, -60, 1600], roll=270)
fig.scene.save('../JATIS/mayavi_cat.png')
fig.scene.save('../JATIS/mayavi_cat.pdf')


# zoom on multilayers and detectors
mlab.view(-110, 60, 170, [-10, 5, -6], roll=45)
fig.scene.save('../JATIS/mayavi_det.png')
fig.scene.save('../JATIS/mayavi_det.pdf')

print('''Manually edit x3d file with these viewpoints:
<Viewpoint id='default' position="-495.98150 2726.27726 3515.55164" orientation="-0.32931 0.43365 0.83875 2.32722" description="Default View"></Viewpoint>
<Viewpoint id="detectors" position="-174.63039 -146.73217 217.09764" orientation="0.67666 -0.71889 -0.15913 0.82107" description="Detectors and ML mirrors"></Viewpoint>
<Viewpoint id="gratings" position="566.05992 641.19197 1920.07059" orientation="0.10952 0.64622 0.75526 2.19224" description="Gratings"></Viewpoint>
''')
