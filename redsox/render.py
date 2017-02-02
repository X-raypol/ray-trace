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
out = marxs.visualization.mayavi.plot_object(redsox.redsox, viewer=fig)

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


# plot of detector images
# Note: Order 0 has far too many photons because my entrance aperture is not a ring
fig = plt.figure()
for i in range(4):
    ax = fig.add_subplot(2, 2, i+1)
    ax.set_title('CCD ID: {0}'.format(i))
    ind = photons['CCD_ID'] == i
    im = ax.hist2d(photons['detpix_x'][ind], photons['detpix_y'][ind],
                   weights=photons['probability'][ind], range=[[0,1631], [0, 1607]],
                   bins=100)
    plt.colorbar(im[3], ax=ax)



# Make table with grating positions
posdat = np.zeros((len(gg.elements), 6))
for i, e in enumerate(gg.elements):
    trans, rot, zoom, shear = transforms3d.affines.decompose(e.pos4d)
    posdat[i, :3] = trans
    posdat[i, 3:] = np.rad2deg(transforms3d.euler.mat2euler(np.dot(redsox.xyz2zxy[:3,:3].T, rot),'sxyz'))

from astropy.table import Table
from astropy import units as u
outtab = Table(posdat, names=['x','y','z','rot_x', 'rot_y', 'rot_z'])
for n in 'xyz':
    outtab[n].unit = u.mm
    outtab['rot_' + n].unit = u.degree
for c in outtab.colnames:
    outtab[c].format = '5.1f'
