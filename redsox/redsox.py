import os
import copy
import numpy as np
from transforms3d.euler import euler2mat
import transforms3d
from astropy.table import Table
from astropy.modeling.models import Gaussian1D
from scipy.interpolate import interp1d

from marxs import optics, simulator
from marxs import energy2wave
from marxs.optics.multiLayerMirror import FlatBrewsterMirror

from read_grating_data import InterpolateRalfTable, RalfQualityFactor
from gratings import GratingGrid


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
                             zoom=[1, 200, 200])

mirror = optics.FlatStack(orientation=xyz2zxy[:3, :3], position=[0, 0, 2500],
                          zoom=[1, 200, 200],
                          elements=[optics.PerfectLens, optics.RadialMirrorScatter],
                          keywords=[{'focallength': 2500},
                                    {'inplanescatter': 30. / 2.3545 / 3600 / 180. * np.pi,
                                     'perpplanescatter': 10 / 2.345 / 3600. / 180. * np.pi}])
mirror.elements[1].display = {'color': (0.0, 0.5, 0.0), 'opacity': 0.1}

# Gratings
grating_coords = Table.read(os.path.join(redsoxbase, 'GratingCoordinates.txt'),
                            format='ascii.commented_header', header_start=2)
for n in ['pitch', 'yaw', 'roll']:
    grating_coords[n] = np.deg2rad(grating_coords[n])
grating_pos = [[p['X'], p['Y'], p['Z']] for p in grating_coords]
grating_orient = [np.dot(euler2mat(p['roll'], p['pitch'], 0, 'rxyz'), xyz2zxy[:3, :3]) for p in grating_coords]
grating_zoom = [0.25, 15, 5]

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

grat1 = GratingGrid(id_num_offset=100,
                    **grat_args_full)
grat_pos2 = [np.dot(rotchan2, e.pos4d) for e in grat1.elements]
grat_pos3 = [np.dot(rotchan3, e.pos4d) for e in grat1.elements]
grat2 = simulator.Parallel(elem_pos=grat_pos2, id_num_offset=200, **grat_args)
grat3 = simulator.Parallel(elem_pos=grat_pos3, id_num_offset=300, **grat_args)
grat = simulator.Sequence(elements=[grat1, grat2, grat3, catsupport,
                                    gratquality])


# ML mirrors
class MirrorEfficiency(optics.FlatOpticalElement):
    sigma_scale = 0.01
    '''Width of the Gaussian.'''

    def __init__(self, datafile, **kwargs):
        data = Table.read(datafile, format='ascii.no_header', data_start=1,
                          names=['wave', 'R', 'M', 'Fig. Merit'])
        self.amp = interp1d(data['wave'], data['R'])
        super(MirrorEfficiency, self).__init__(**kwargs)

    def D(self, x):
        '''Herman D(x) = 0.88 Ang/mm * x (in mm) + 26 Ang,
        where x is measured from the short wavelength end of the mirror.
        In marxs x is measured from the center, so we add 15 mm (the half-length.)
        '''
        return 0.88 * (x + 15) + 26

    def specific_process_photons(self, photons,
                                 intersect, interpoos, intercoos):
        cosang = np.dot(photons['dir'].data[intersect, :],
                        self.geometry('e_x'))
        wave_braggpeak = 2 * self.D(intercoos[intersect, 0]) * cosang
        wave_nominal = 2 * self.D(intercoos[intersect, 0]) * 2**(-0.5)
        amp = self.amp(wave_nominal)
        gaussians = Gaussian1D(amplitude=amp, mean=1.,
                               stddev=self.sigma_scale / 2.355)
        wave = energy2wave / photons['energy'][intersect] * 1e7
        return {'probability': gaussians(wave / wave_braggpeak),
                'mlwave_nominal': wave_nominal,
                'mlwave_braggpeak': wave_braggpeak,
                'mlcosang': cosang,
                'ml_x': intercoos[intersect, 0] }

mlkwargs = {'elements': [FlatBrewsterMirror, MirrorEfficiency],
            'keywords': [{}, {'datafile': os.path.join(inputpath, 'ml_refl_2015_minimal.txt')},
                     ]}

ml1 = optics.FlatStack(zoom=[0.25, 15., 5.], position=[44.55, 0, 0],
                       orientation=np.dot(euler2mat(-np.pi / 4, 0, 0, 'sxyz'),
                                          xyz2zxy[:3, :3]),
                       **mlkwargs)
ml1.loc_coos_name = ['ml1_x', 'ml1_y']
ml2 = optics.FlatStack(pos4d=np.dot(rotchan2, ml1.pos4d), **mlkwargs)
ml3 = optics.FlatStack(pos4d=np.dot(rotchan3, ml1.pos4d), **mlkwargs)
ml = simulator.Sequence(elements=[ml1, ml2, ml3])

# Detectors
pixsize = 0.016
detkwargs = {'pixsize': pixsize, 'id_col': 'CCD_ID'}
detzoom = np.array([1, pixsize * 1632 / 2, pixsize * 1608 / 2])
detposlist = [transforms3d.affines.compose(np.zeros(3), xyz2zxy[:3, :3],
                                           detzoom),
              transforms3d.affines.compose([44.55, 15, 0],
                                           np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                                           detzoom)]
detposlist.append(np.dot(rotchan2, detposlist[1]))
detposlist.append(np.dot(rotchan3, detposlist[1]))
det = simulator.Parallel(elem_class=optics.FlatDetector,
                         elem_pos=detposlist,
                         elem_args=detkwargs)

# Place an additional detector in the focal plane for comparison
# Detectors are transparent to allow this stuff
detfp = optics.FlatDetector(position=[0, 0, -100], orientation=xyz2zxy[:3, :3],
                            zoom=[.2, 500, 500])
detfp.loc_coos_name = ['detfp_x', 'detfp_y']
detfp.detpix_name = ['detfppix_x', 'detfppix_y']
detfp.display['opacity'] = 0.1


keeppos = simulator.KeepCol('pos')
redsox = simulator.Sequence(elements=[aper, mirror, grat, ml, det, detfp],
                            postprocess_steps=[keeppos])
chan1 = simulator.Sequence(elements=[aper, mirror, grat1, ml1, det.elements[1]])
