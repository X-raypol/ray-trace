'''Simulate a source with a real spectrum with some interesting polarization behavior.
This is not necessarily a realistic source, but it meant for an exercise in data
reduction where I don't tell what the input spectrum is until after the reduction.

Therefore, in fact, I have to use a spectrum with non-physical properties, since
Herman knows better what to expect from this source than I do.
'''
import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import table
from marxs.source import PointSource, FixedPointing
from marxs.visualization.utils import format_saved_positions
from marxs.source.source import poisson_process
import marxs.visualization.mayavi
import redsox
from mirror import Ageom

from mayavi import mlab
%matplotlib

# Make two components with different polarization directions
energy = np.arange(.1, .9, .01)
polangle = np.arange(0, 360., 1.)

# Component 1: Powerlaw gamma = 1.5, polangle = 50 deg, polarization fraction 50 %
fluxdens1 = energy ** (-1.5)
pol1 = np.ones_like(polangle) / len(polangle)
pol1[45:55] += 1.  / 10
src1 = PointSource(coords=SkyCoord(30., 30., unit='deg'),
                   energy={'energy': energy, 'flux': fluxdens1},
                   polarization={'angle': polangle, 'probability': pol1},
                   geomarea=Ageom,
                   flux=poisson_process(2))

# Component 2: polarization fraction 33 %
spec = table.Table.read('../inputdata/bb36.tbl', format='ascii', names=['energy','flux'])
pol2 = np.ones_like(polangle) / len(polangle)
pol2 [100:110] += .5 / 10
src2 = PointSource(coords=SkyCoord(30., 30., unit='deg'),
                   energy={'energy': energy, 'flux': fluxdens1},
                   polarization={'angle': polangle, 'probability': pol1},
                   geomarea=Ageom,
                   flux=poisson_process(1))

my_pointing = FixedPointing(coords=SkyCoord(30., 30., unit='deg'),
                            reference_transform=redsox.xyz2zxy)

p1 = src1.generate_photons(5e3)
p2 = src2.generate_photons(5e3)
p = table.vstack([p1, p2])
p.sort(['time'])

p = my_pointing(p)

p = redsox.redsox(p)

fig = mlab.figure()
mlab.clf()
out = marxs.visualization.mayavi.plot_object(redsox.redsox, viewer=fig)
pos = format_saved_positions(redsox.keeppos)
ind = np.isfinite(p['order']) & (p['probability'] > 1e-4)
marxs.visualization.mayavi.plot_rays(pos[ind, :, :], scalar=p['energy'][ind], viewer=fig)


# plot of detector images

import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(4):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.set_title('CCD ID: {0}'.format(i))
    ind = p['CCD_ID'] == i
    im = ax.hist2d(p['detpix_x'][ind], p['detpix_y'][ind],
                   weights=p['probability'][ind], range=[[0,1631], [0, 1607]],
                   bins=100)
    plt.colorbar(im[3], ax=ax)

# save
import os
p['PI'] = p['energy'] + 0.1 * np.random.randn(len(p))
p.write(os.path.join(redsox.redsoxbase,'raytrace', 'evtfiles','fullstack.fits'))
# draw from the distribution and save for Herman
pout = p[['time', 'detpix_x', 'detpix_y', 'CCD_ID', 'PI']]
pout = pout[(pout['CCD_ID'] >= 0) & (np.random.rand(len(p)) < p['probability'])]
pout.write(os.path.join(redsox.redsoxbase,'raytrace', 'evtfiles','obs1.fits'), overwrite=True)
