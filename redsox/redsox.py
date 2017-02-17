import os
import copy
import numpy as np
from scipy.interpolate import interp1d
from transforms3d.euler import euler2mat
import transforms3d
from astropy.table import Table

import marxs
from marxs import optics, simulator

from read_grating_data import InterpolateRalfTable, RalfQualityFactor
from gratings import GratingGrid
from mlmirrors import LGMLMirror


def euler2aff(*args, **kwargs):
    mat = euler2mat(*args, **kwargs)
    return transforms3d.affines.compose(np.zeros(3), mat, np.ones(3))

redsoxbase = '/melkor/d1/guenther/Dropbox/REDSoX File Transfers'
inputpath = os.path.join(redsoxbase, 'raytrace', 'inputdata')

xyz2zxy = np.array([[0., 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).T
rotchan2 = euler2aff(np.pi * 2 / 3, 0, 0, 'szyz')
rotchan3 = euler2aff(-np.pi * 2 / 3, 0, 0, 'szyz')

# Optics
# Scatter as FWHM ~30 arcsec. Divide by 2.3545 to get Gaussian sigma.
# Scatter numbers are still all wrong.
aper = optics.CircleAperture(orientation=xyz2zxy[:3, :3], position=[0, 0, 2700],
                             zoom=[1, 223, 223], r_inner=165)

# Nickel reflectivity
ni_refl = Table.read(os.path.join(inputpath, 'ni_refl_1deg.txt'),
                     format='ascii.no_header', data_start=2,
                     names=['energy', 'refl'])


mirror = optics.FlatStack(orientation=xyz2zxy[:3, :3], position=[0, 0, 2500],
                          zoom=[1, 225, 225],
                          elements=[optics.PerfectLens,
                                    optics.RadialMirrorScatter,
                                    optics.EnergyFilter,
                                    optics.EnergyFilter
                                ],
                          keywords=[{'focallength': 2500},
                                    {'inplanescatter': 30. / 2.3545 / 3600 / 180. * np.pi,
                                     'perpplanescatter': 10 / 2.345 / 3600. / 180. * np.pi},
                                    {'filterfunc': interp1d(ni_refl['energy'] * 1e-3, ni_refl['refl']**2),
                                     'name': 'double Ni reflectivity'},
                                    {'filterfunc': lambda x: np.ones_like(x) * 0.88,
                                     'name': 'support spider'}
                                    ])
mirror.elements[1].display = {'color': (0.0, 0.5, 0.0), 'opacity': 0.1}

# Gratings
grating_coords = Table.read(os.path.join(redsoxbase, 'GratingCoordinates.txt'),
                            format='ascii.commented_header', header_start=2)
for n in ['pitch', 'yaw', 'roll']:
    grating_coords[n] = np.deg2rad(grating_coords[n])
grating_pos = [[p['X'], p['Y'], p['Z']] for p in grating_coords]
grating_orient = [np.dot(euler2mat(p['roll'], p['pitch'], 0, 'rxyz'), xyz2zxy[:3, :3]) for p in grating_coords]
grating_zoom = [0.25, 4, 5]

ralfdata = os.path.join(inputpath, 'CATefficiencies_v3.xlsx')
gratquality = RalfQualityFactor(d=200.e-3, sigma=1.75e-3)

order_selector = InterpolateRalfTable(ralfdata)

# Define L1, L2 blockage as simple filters due to geometric area
# L1 support: blocks 18 %
# L2 support: blocks 19 %
catsupport = optics.GlobalEnergyFilter(filterfunc=lambda e: 0.81 * 0.82)

blazeang = 0.8
blazemat = transforms3d.axangles.axangle2mat(np.array([0, 0, 1]), np.deg2rad(-blazeang))
grat_args = {'elem_class': optics.CATGrating,
             'elem_args': {'d': 2e-4, 'order_selector': order_selector},
             'id_col': 'grating_id'}
grat_args_full = copy.deepcopy(grat_args)
grat_args_full['elem_args']['zoom'] = grating_zoom
#grat_args_full['elem_args']['orientation'] = np.dot(xyz2zxy[:3, :3], blazemat)
#grat_args_full['elem_args']['orientation'] = blazemat
grat_args_full['elem_args']['orientation'] = np.dot(transforms3d.axangles.axangle2mat([1,0,0], np.pi/2), blazemat)

grat1 = GratingGrid(id_num_offset=1000,
                    **grat_args_full)
grat_pos2 = [np.dot(rotchan2, e.pos4d) for e in grat1.elements]
grat_pos3 = [np.dot(rotchan3, e.pos4d) for e in grat1.elements]
grat2 = simulator.Parallel(elem_pos=grat_pos2, id_num_offset=2000, **grat_args)
grat3 = simulator.Parallel(elem_pos=grat_pos3, id_num_offset=3000, **grat_args)


class CATSupportbars(marxs.base.SimulationSequenceElement):
    '''carbon fiber structure that holds grating facets will absorb all photons
    that do not pass through a grating facet.

    We might want to call this L3 support ;-)
    '''
    def process_photons(self, photons):
        photons['probability'][photons['grating_id'] < 0] = 0.
        return photons


grat = simulator.Sequence(elements=[grat1, grat2, grat3, catsupport,
                                    gratquality, CATSupportbars()])


# ML mirrors
# mirrorfile = 'ml_refl_2015_minimal.txt'
mirrorfile = 'ml_refl_2017_withCrSc_width.txt'

ml1 = LGMLMirror(datafile=os.path.join(inputpath, mirrorfile),
                 zoom=[0.25, 15., 5.], position=[44.55, 0, 0],
                 orientation=np.dot(euler2mat(-np.pi / 4, 0, 0, 'sxyz'),
                                    xyz2zxy[:3, :3])
)
ml2 = LGMLMirror(datafile=os.path.join(inputpath, mirrorfile),
                 pos4d=np.dot(rotchan2, ml1.pos4d))
ml3 = LGMLMirror(datafile=os.path.join(inputpath, mirrorfile),
                 pos4d=np.dot(rotchan3, ml1.pos4d))
ml = simulator.Sequence(elements=[ml1, ml2, ml3])

# Detectors
# read optical blocking filter and CCD QE
ccdqe = Table.read(os.path.join(inputpath, 'xgs_bi_ccdqe.dat'),
                   format='ascii.no_header', comment='!',
                   names=['energy', 'qe' , 'filtertrans', 'temp'])

pixsize = 0.016
detkwargs = {'pixsize': pixsize}
detzoom0 = np.array([1, pixsize * 408 / 2, pixsize * 1608 / 2])
detzoom = np.array([1, pixsize * 1632 / 2, pixsize * 1608 / 2])
# rotate by 45 deg
rot = transforms3d.euler.euler2mat(np.pi/4, 0, 0, 'sxyz')
# flip to x-z plane
flip = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
detposlist = [transforms3d.affines.compose(np.zeros(3), xyz2zxy[:3, :3],
                                           detzoom0),
              transforms3d.affines.compose([44.55, 25, 0],
                                           np.dot(flip, rot),
                                           detzoom)]
detposlist.append(np.dot(rotchan2, detposlist[1]))
detposlist.append(np.dot(rotchan3, detposlist[1]))
det = simulator.Parallel(elem_class=optics.FlatDetector,
                         elem_pos=detposlist,
                         elem_args=detkwargs)

ccd123_args = {'elements': [optics.FlatDetector, optics.EnergyFilter],
               'keywords': [detkwargs,
                            {'filterfunc': interp1d(ccdqe['energy'] * 1e-3,
                                                    ccdqe['qe']),
                             'name': 'CCD QE'},
                           ],
               }
ccd0_args = copy.deepcopy(ccd123_args)
ccd0_args['keywords'][0]['id_num'] = 0
ccd0_args['keywords'][0]['id_col'] = 'CCD_ID'
ccd0_args['elements'].append(optics.EnergyFilter)
ccd0_args['keywords'].append({'filterfunc': interp1d(ccdqe['energy'] * 1e-3,
                                                     ccdqe['filtertrans']),
                              'name': 'optical blocking filter'})
det0 = optics.FlatStack(pos4d=detposlist[0], **ccd0_args)
det123 = simulator.Parallel(elem_class=optics.FlatStack,
                            elem_pos=detposlist[1:],
                            elem_args=ccd123_args,
                            id_num_offset=1,
                            id_col='CCD_ID')
det = simulator.Sequence(elements=[det0, det123])

# Place an additional detector in the focal plane for comparison
# Detectors are transparent to allow this stuff
detfp = optics.FlatDetector(position=[0, 0, -100], orientation=xyz2zxy[:3, :3],
                            zoom=[.2, 500, 500])
detfp.loc_coos_name = ['detfp_x', 'detfp_y']
detfp.detpix_name = ['detfppix_x', 'detfppix_y']
detfp.display['opacity'] = 0.1


keeppos = simulator.KeepCol('pos')
redsox = simulator.Sequence(elements=[aper, mirror, grat, ml, det],
                            postprocess_steps=[keeppos])

redsoxfp = simulator.Sequence(elements=[aper, mirror, grat, ml, det, detfp],
                              postprocess_steps=[keeppos])

chan1 = simulator.Sequence(elements=[aper, mirror, grat1, ml1, det.elements[1]])
