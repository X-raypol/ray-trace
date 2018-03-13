# Add color for CATs (gold? Different color for each channel?)
# Run and plot rays one channel at a time, only plot rays that go through facet
# that should weed out the double diffracted rays
from __future__ import print_function

import os
from mayavi import mlab
import marxs.visualization.mayavi as marxsavi
from settings import figureout, kwargsfig
import copy

import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
import marxs.visualization.mayavi
from marxs.source import PointSource, FixedPointing
from marxs.optics import FlatOpticalElement, FlatDetector
from marxs.simulator import Sequence
import marxs

import sys
sys.path.append('../redsox')
import redsox

%matplotlib


def check_second_interaction(photons, pos, dir, channel):
    '''In this compressed layout it happens that a ray that is diffracted in a high
    sector hits a grating in one of the "low" sectors again. In the real design
    that is not a problem because the diffraction angle is much smaller. So, to
    avoid confusion, we will just not chose rays that do that for plotting
    here.
    So, to do this test, we need a REDSox with only low sector gratings and
    those low sector gratings have to be labelled in a way that I can recognize
    them. At the same time, we have to make sure that we retain photons that go
    through the low sector of the channel that we are looking at at the moment.
    '''
    photons['pos'] = pos
    photons['dir'] = dir
    photons.rename_column('facet', 'facet_real')
    cats = redsox.PerfectRedsox.cat_class(conf=conf,
                                          channels='123'.replace(channel, ''))
    photons = cats(photons)
    return photons['facet'] >= 0

conf = {'aper_z': 290.,
        'aper_rin': 35.,
        'aper_rout': 45.,
        'mirr_rout': 46,
        'mirr_length': 20,
        'f': 250,
        'inscat': redsox.FWHMarcs2sigrad(30.),
        'perpscat': redsox.FWHMarcs2sigrad(10.),
        'blazeang': 0.8 * u.degree,
        'gratingzoom': [0.25, 8, 3],
        'd': 2e-5,
        'rotchan': {'1': np.eye(4),
                    '2': redsox.euler2aff(np.pi * 2 / 3, 0, 0, 'szyz'),
                    '3': redsox.euler2aff(-np.pi * 2 / 3, 0, 0, 'szyz')},
        'grat_id_offset': {'1': 1000, '2': 2000, '3': 3000},
        'beta_lim': np.deg2rad([7., 12.]),
        'mirrorfile': 'ml_refl_2017_withCrSc_width.txt',
        'MLzoom': [0.25, 15, 5],
        'MLpos': [30, 0, 0],
        'pixsize': 0.016,
        'detsize0': [408, 1608],
        'detsize123': [1632, 1608],
        }


instrum = redsox.PerfectRedsox(conf=conf)
instrums = [redsox.PerfectRedsox(conf=conf, channels=c) for c in '123']
for i in instrums:
    i.KeepDir = marxs.simulator.KeepCol('dir')
    i.postprocess_steps.append(i.KeepDir)
colors = 'ryg'
# Now change some formatting and display to get better output
instrum.elements[0].display['outer_factor'] = 1.5
instrum.elements[1].display = copy.copy(instrum.elements[1].display)
instrum.elements[1].display['shape'] = 'cylinder'
instrum.elements[1].display['color'] = (.7, .7, .7)
for i in range(3):
    disp = {'shape': 'box', 'color': colors[i]}
    gg = instrum.elements[2].elements[i]
    for e in gg.elements:
        e.display = disp
    for e in instrums[i].elements[2].elements[0].elements:
        e.display = disp


my_source = PointSource(coords=SkyCoord(30., 30., unit='deg'), energy=0.3,
                        polarization=120.,
                        geomarea=instrum.elements[0].area)
my_pointing = FixedPointing(coords=SkyCoord(30., 30., unit='deg'),
                            reference_transform=redsox.xyz2zxy)

photons = my_source.generate_photons(20.)
photons = my_pointing(photons)
p123 = [i(photons.copy()) for i in instrums]
posfull123 = [i.KeepPos.format_positions() for i in instrums]

colorid = []
positions = []
for i in range(3):
    secondinterac = check_second_interaction(p123[i].copy(),
                                             instrums[i].KeepPos.data[2],
                                             instrums[i].KeepDir.data[2],
                                             '123'[i])
    # Plot all dispersed rays
    ind = (p123[i]['facet'] >= 0) & (p123[i]['CCD_ID'] > 0) & ~secondinterac
    positions.append(posfull123[i][ind, :, :])
    colorid.append(i * np.ones(ind.sum()))
    # But only a sample of 0 order rays
    ind = (p123[i]['facet'] >= 0) & (p123[i]['CCD_ID'] == 0) & ~secondinterac
    ind[100:] = False
    positions.append(posfull123[i][ind, :, :])
    colorid.append(i * np.ones(ind.sum(), dtype=int))


def plot_halfplanes(x=50, y=50, z=300):
    '''x,y,z upper boundaries of the half planes'''
    # yz - plane
    mlab.triangular_mesh(np.array([0., 0, 0, 0]),
                         y * np.array([0., 1, 1, 0]),
                         z * np.array([0., 0, 1, 1]),
                         [[0, 1, 2], [0, 2, 3]],
                         opacity=0.3,
                         color=(.4, .4, .4))
    # xz - plane
    mlab.triangular_mesh(x * np.array([0., 1, 1, 0]),
                         np.array([0., 0, 0, 0]),
                         z * np.array([0., 0, 1, 1]),
                         [[0, 1, 2], [0, 2, 3]],
                         opacity=0.3,
                         color=(.8, .4, .0))
    # xy - plane
    mlab.triangular_mesh(x * np.array([0., 1, 1, 0]),
                         y * np.array([1., 1, 0, 0]),
                         np.array([0., 0, 0, 0]),
                         [[0, 1, 2], [0, 2, 3]],
                         opacity=0.3,
                         color=(.0, .4, .8))


def plot_axes(mode='mayavi', x=[-60, 60], y=[-60, 60], z=[-20, 310]):
    '''Draw axes. I don't get the mayavi axis to hit the right points, so I do it by hand here.'''
    xyz = np.zeros((3, 2, 3))
    # x-axis
    xyz[0, 0, 0] = x[0]
    xyz[0, 1, 0] = x[1]
    # y-axis
    xyz[1, 0, 1] = y[0]
    xyz[1, 1, 1] = y[1]
    # z-axis
    xyz[2, 0, 2] = z[0]
    xyz[2, 1, 2] = z[1]
    out = marxs.visualization.mayavi.plot_rays(xyz,
                                               scalar=np.zeros(3),
                                               kwargssurface={'opacity': 1.,
                                                              'line_width': 3,
                                                              'colormap': 'gist_gray'})
    if mode == 'mayavi':
        # For mayavi to camera:
        mlab.text3d(x[1] + 2, 0, 0, 'x', scale=12, color=(0, 0, 0))
        mlab.text3d(0, y[1] + 2, 0, 'y', scale=12, color=(0, 0, 0))
        mlab.text3d(0, 0, z[1] + 5, 'z', scale=12, color=(0, 0, 0))
    elif mode == 'mayavi_rot':
        mlab.text3d(x[1] + 2, -6, -4, 'x', scale=12, color=(0, 0, 0),
                    orient_to_camera=False, orientation=(90, 0, 90))
        mlab.text3d(-6, y[1] + 2, -5, 'y', scale=12, color=(0, 0, 0),
                    orient_to_camera=False, orientation=(90, 0, 0))
        mlab.text3d(-6, 0, z[1] + 5, 'z', scale=12, color=(0, 0, 0),
                    orient_to_camera=False, orientation=(90, 0, 0))
    elif mode == 'x3d':
        mlab.text3d(x[1] + 2, -6, -14, 'x', scale=12, color=(0, 0, 0),
                    orient_to_camera=False, orientation=(0, -90, -90))
        mlab.text3d(-6, y[1] + 2, -5, 'y', scale=12, color=(0, 0, 0),
                    orient_to_camera=False, orientation=(-90, 0, 0))
        mlab.text3d(-6, 0, z[1] + 5, 'z', scale=12, color=(0, 0, 0),
                    orient_to_camera=False, orientation=(90, 0, 0))


def plot_rg():
    '''Now mark r_g for one grating for explain coordinate system'''
    gratingcoords = np.array(instrums[0].elements[2].elements[0].elem_pos)[:, :3, 3]
    # lower sector
    ind = (gratingcoords[:, 2] < np.mean(gratingcoords[:, 2])).nonzero()[0]
    # furthest form yz plane for visibility
    indmax = np.argmax(gratingcoords[ind, 0])
    rgx, rgy, rgz = gratingcoords[ind[indmax], :]
    # plane for grating coords
    r = np.sqrt(rgx**2 + rgy**2 + rgz**2)
    out = mlab.triangular_mesh(np.zeros_like(alpha),
                               np.array([0., 0., rgy]),
                               np.zeros(3),
                               [[0, 1, 2]],
                               opacity=0.7,
                               color=(0., 0., 1.))
    out = mlab.triangular_mesh(np.array([0., rgx, rgx]),
                               np.array([0., 0., rgy]),
                               np.array([rgz, rgz, rgz]),
                               [[0, 1, 2]],
                               opacity=0.7,
                               color=(0., 0., 1.))

    out = mlab.triangular_mesh(np.array([0., 0., rgx]),
                               np.array([0., 0., rgy]),
                               np.array([0., rgz, rgz]),

                               [[0, 1, 2]],
                               opacity=0.7,
                               color=(.5, .5, .5))
    # Draw a line projecting grating coord to the xy plane
    lines = np.array([[[rgx, rgy, rgz], [0, rgy, rgz]],
                      [[rgx, rgy, rgz], [0, rgy, rgz]],
                      [[rgx, rgy, rgz], [0, 0, 0]]])
    out = marxs.visualization.mayavi.plot_rays(lines,
                                               scalar=np.zeros(3),
                                               kwargssurface={'opacity': 1.,
                                                              'line_width': 3,
                                                              'colormap': 'blue-red'})
    return rgx, rgy, rgz


def plot_correct_rg(n=50):
    gratingcoords = np.array(instrums[0].elements[2].elements[0].elem_pos)[:, :3, 3]
    # lower sector
    ind = (gratingcoords[:, 2] < np.mean(gratingcoords[:, 2])).nonzero()[0]
    # furthest form yz plane for visibility
    indmax = np.argmax(gratingcoords[ind, 0])
    rgx, rgy, rgz = gratingcoords[ind[indmax], :]
    # plane for grating coords
    r = np.sqrt(rgx**2 + rgy**2 + rgz**2)

    gammag =  np.arctan(np.sqrt(rgy**2 + rgz**2) / rgx)
    betag = np.arctan(rgy/rgz)
    ang = np.linspace(0, gammag, n)
    points = np.zeros((n + 1, 3))
    points[1:, 0] = r * np.cos(ang)
    points[1:, 1] = r * np.sin(ang) * np.sin(betag)
    points[1:, 2] = r * np.sin(ang) * np.cos(betag)
    triangles = np.zeros((n - 1 , 3), dtype=int)
    triangles[:, 1] = np.arange(n - 1) + 1
    triangles[:, 2] = np.arange(n - 1) + 2
    out = mlab.triangular_mesh(points[:, 0],
                               points[:, 1],
                               points[:, 2],
                               triangles,
                               opacity=0.7,
                               color=(0., 0., 1.))

    ang = np.linspace(0, betag, n)
    points[1:, 0] = 0
    points[1:, 1] = r * np.sin(gammag) * np.sin(ang)
    points[1:, 2] = r * np.sin(gammag) * np.cos(ang)
    out = mlab.triangular_mesh(points[:, 0],
                               points[:, 1],
                               points[:, 2],
                               triangles,
                               opacity=0.7,
                               color=(0., 0., 0.))

    lines = np.array([[[rgx, rgy, rgz], [0, rgy, rgz]],
                      [[0, 0, 0], [0, rgy, rgz]],
                      [[rgx, rgy, rgz], [0, 0, 0]]])
    out = marxs.visualization.mayavi.plot_rays(lines,
                                               scalar=np.zeros(3),
                                               kwargssurface={'opacity': 1.,
                                                              'line_width': 3,
                                                              'colormap': 'blue-red'})
    return rgx, rgy, rgz, r


figcoordsys = mlab.figure(bgcolor=(1,1,1), size=(1000, 1000))
out = marxsavi.plot_object(Sequence(elements=instrum.elements[3:]),
                           viewer=figcoordsys)
out = marxsavi.plot_object(instrums[0].elements[2], viewer=figcoordsys)
rgx, rgy, rgz, r = plot_correct_rg()
#plot_halfplanes(r * 1, rgy, rgz)
plot_axes(x=[-0.1 * r, 1.05 * r], z=[0, rgz * 1.05])
# Mark center of gratings
mlab.points3d(rgx, rgy, rgz, scale_factor=3)
# yz - plane
mlab.triangular_mesh(np.array([0., 0, 0, 0]),
                     rgy*2 * np.array([-1, 1, 1., -1]),
                     r * np.array([0, 0., 1, 1]),
                     [[0, 1, 2], [0, 2, 3]],
                     opacity=0.3,
                     color=(.4, .4, .4))

# mlab.view(azimuth=33, elevation=50, distance=300, focalpoint=[20, 14, 62], roll=300)
mlab.view(-70, 130, 355, [50, 3, 46], roll=180)
figcoordsys.scene.save('../JATIS/coordsys.png')
figcoordsys.scene.save('../JATIS/coordsys.pdf')


# saving to x3d resets color scale form in to max, ignoring vmin and vmax
# Thus, add a non-visible line here with color 5 to prevent that
positions.append(np.zeros((1, 5, 3)))
colorid.append([5])

fig = mlab.figure(**kwargsfig)
out = marxsavi.plot_object(instrum, viewer=fig)

out = marxs.visualization.mayavi.plot_rays(np.concatenate(positions, axis=0),
                                           scalar=np.hstack(colorid),
                                           kwargssurface={'opacity': .5,
                                                          'line_width': 1,
                                                          'colormap': 'gist_rainbow',
                                                          'vmin': 0,
                                                          'vmax': 5})
mlab.view(-100, 60, 355, [9, -2, 170], roll=90)
fig.scene.save('../JATIS/not2scale.png')
fig.scene.save('../JATIS/not2scale.pdf')
fig.scene.save('../JATIS/web/not2scale.x3d')

print('''Manually edit x3d file with these viewpoints:
<Viewpoint id='default' position="-277.57241 -74.48035 311.14562" orientation="-0.53494 0.07354 0.84168 3.74926" description="Default View"></Viewpoint>
<Viewpoint id="detectors" position="-92.27657 110.76398 51.15283" orientation="-0.29571 0.52103 0.80067 3.35585" description="Detectors and ML mirrors"></Viewpoint>
<Viewpoint id="gratings" position="-120.68255 171.85418 230.28632" orientation="-0.30311 0.50321 0.80926 3.34962" description="Gratings"></Viewpoint>
''')
