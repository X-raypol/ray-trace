import os
from mayavi import mlab
import marxs.visualization.mayavi as marxsavi
from settings import figureout, kwargsfig
import sys
sys.path.append('../redsox')
import redsox

%matplotlib
fig = mlab.figure(**kwargsfig)
for g in redsox.grat2.elements:
    g.display['color'] = (1.0, 1., 1.)
out = marxsavi.plot_object(redsox.grat1, viewer=fig)
for g in redsox.grat2.elements:
    g.display['color'] = (1.0, 0.5, 0.5)
out = marxsavi.plot_object(redsox.grat2, viewer=fig)

for g in redsox.grat3.elements:
    g.display['color'] = (0.5, 0.5, 1.0)
out = marxsavi.plot_object(redsox.grat3, viewer=fig)


mlab.plot3d([0, 0], [0, 0], [2e3, 1.1e3], color=(1., 1., 1.),
            opacity=0.3, tube_radius=20, tube_sides=20)
mlab.view(azimuth=40, elevation=70, distance=700, focalpoint=[0, 0, 1620])

fig.scene.save(os.path.join(figureout, 'gratings_mayavi.pdf'))
