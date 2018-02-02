import os
import copy
import numpy as np
from scipy.interpolate import interp1d
from transforms3d.euler import euler2mat
import transforms3d
from astropy.table import Table
import astropy.units as u

import marxs
from marxs import optics, simulator

from arcus.ralfgrating import (RalfQualityFactor,
                               catsupport, catsupportbars)
from gratings import GratingGrid
from mlmirrors import LGMLMirror


def euler2aff(*args, **kwargs):
    mat = euler2mat(*args, **kwargs)
    return transforms3d.affines.compose(np.zeros(3), mat, np.ones(3))

redsoxbase = '/melkor/d1/guenther/Dropbox/REDSoX File Transfers'
inputpath = os.path.join(redsoxbase, 'raytrace', 'inputdata')

xyz2zxy = np.array([[0., 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).T


def FWHMarcs2sigrad(x):
    '''Convert FWHM in arcsec to Gaussian sigma in rad'''
    return x / 2.3545 / 3600 / 180. * np.pi


conf = {'aper_z': 2700.,
        'aper_rin': 165.,
        'aper_rout': 223.,
        'mirr_rout': 225,
        'f': 2500,
        'inscat': FWHMarcs2sigrad(30.),
        'perpscat': FWHMarcs2sigrad(10.),
        'blazeang': 0.8 * u.degree,
        'gratingzoom': [0.25, 4, 5],
        'd': 2e-4,
        'rotchan': {'1': np.eye(4),
                    '2': euler2aff(np.pi * 2 / 3, 0, 0, 'szyz'),
                    '3': euler2aff(-np.pi * 2 / 3, 0, 0, 'szyz')},
        'grat_id_offset': {'1': 1000, '2': 2000, '3': 3000},
        'beta_lim': np.deg2rad([3.5, 5.05]),
        'mirrorfile': 'ml_refl_2017_withCrSc_width.txt',
        'MLzoom': [0.25, 15., 5.],
        'MLpos': [44.55, 0, 0],
        'pixsize': 0.016,
        'detsize0': [408, 1608],
        'detsize123': [1632, 1608],
        }


# Aperture
class CircleAperture(optics.CircleAperture):
    def __init__(self, channels, conf):
        super(CircleAperture, self).__init__(orientation=xyz2zxy[:3, :3],
                                             position=[0, 0, conf['aper_z']],
                                             zoom=[1, conf['aper_rout'], conf['aper_rout']],
                                             r_inner=conf['aper_rin'])


class SimpleMirror(optics.FlatStack):
    # Optics
    # Scatter as FWHM ~30 arcsec. Divide by 2.3545 to get Gaussian sigma.
    # Scatter numbers are still all wrong.
    spider_fraction = 0.88

    def refl(self):
        '''Read Ni reflectivity'''
        self.refl = Table.read(os.path.join(inputpath, 'ni_refl_1deg.txt'),
                               format='ascii.no_header', data_start=2,
                               names=['energy', 'refl'])
        self.refl['energy'] *= 1e-3  # convert eV to keV
        return self.refl

    def __init__(self, channels, conf):
        refl = self.refl()
        kwords = [{'focallength': conf['f']},
                  {'inplanescatter': conf['inscat'],
                   'perpplanescatter': conf['perpscat']},
                  {'filterfunc': interp1d(refl['energy'], refl['refl']**2),
                   'name': 'double Ni reflectivity'},
                  {'filterfunc': lambda x: np.ones_like(x) * self.spider_fraction,
                   'name': 'support spider'}]
        super(SimpleMirror, self).__init__(orientation=xyz2zxy[:3, :3],
                                           position=[0, 0, conf['f']],
                                           zoom=[1, conf['mirr_rout'],
                                                 conf['mirr_rout']],
                                           elements=[optics.PerfectLens,
                                                     optics.RadialMirrorScatter,
                                                     optics.EnergyFilter,
                                                     optics.EnergyFilter],
                                           keywords=kwords)
        self.set_display()

    def set_display(self):
        self.elements[1].display = {'color': (0.0, 0.5, 0.0), 'opacity': 0.1}

# Gratings
def read_grating_coords():
    '''Read grating coordinates from a file. This is currently not used.'''
    grating_coords = Table.read(os.path.join(redsoxbase, 'GratingCoordinates.txt'),
                                format='ascii.commented_header', header_start=2)
    for n in ['pitch', 'yaw', 'roll']:
        grating_coords[n] = np.deg2rad(grating_coords[n])
    trans = [[p['X'], p['Y'], p['Z']] for p in grating_coords]
    rot = [np.dot(euler2mat(p['roll'], p['pitch'], 0, 'rxyz'),
                  xyz2zxy[:3, :3]) for p in grating_coords]
    return trans, rot


class CATGratings(simulator.Sequence):
    gratquality_class = RalfQualityFactor

    def __init__(self, channels, conf, **kwargs):

        elements = []

        self.gratquality = self.gratquality_class()
        for chan in channels:
            elements.append(GratingGrid(channel=chan, conf=conf))
        elements.extend([catsupport, catsupportbars, self.gratquality])
        super(CATGratings, self).__init__(elements=elements, **kwargs)


class MLMirrors(simulator.Parallel):
    elem_class = LGMLMirror

    def __init__(self, channels, conf):
        lgmlpos = [transforms3d.affines.compose(conf['MLpos'],
                                                np.dot(euler2mat(-np.pi / 4, 0, 0, 'sxyz'),
                                                       xyz2zxy[:3, :3]),
                                                conf['MLzoom'])]
        for chan in '23':
            lgmlpos.append(np.dot(conf['rotchan'][chan], lgmlpos[0]))

        datafile = os.path.join(inputpath, conf['mirrorfile'])
        super(MLMirrors, self).__init__(elem_class=self.elem_class,
                                        elem_pos=lgmlpos,
                                        elem_args={'datafile': datafile},
                                        id_num_offset=1,
                                        id_col='LGML')


class Detectors(simulator.Sequence):
    elem_class = optics.FlatDetector

    def filterqe(self):
        ccdqe = Table.read(os.path.join(inputpath, 'xgs_bi_ccdqe.dat'),
                           format='ascii.no_header', comment='!',
                           names=['energy', 'qe' , 'filtertrans', 'temp'])
        ccdqe['energy'] *= 1e-3  # ev to keV
        return ccdqe

    def __init__(self, channels, conf):
        ccdqe = self.ccedq()

        detkwargs = {'pixsize': conf['pixsize']}
        detzoom0 = np.array([1, pixsize * conf['detsize0'][0] / 2,
                             pixsize * conf['detsize0'][1] / 2])
        detzoom = np.array([1, pixsize * conf['detsize123'][0] / 2,
                            pixsize * conf['detsize123'][1] / 2])
        # rotate by 45 deg
        rot = transforms3d.euler.euler2mat(np.pi/4, 0, 0, 'sxyz')
        # flip to x-z plane
        flip = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        detposlist = [transforms3d.affines.compose(np.zeros(3), xyz2zxy[:3, :3],
                                                   detzoom0),
                      transforms3d.affines.compose([44.55, 25, 0],
                                                   np.dot(flip, rot),
                                                   detzoom)]
        for chan in '23':
            detposlist.append(np.dot(conf['rotchan'][chan], detposlist[0]))


            ccd123_args = {'elements': [self.elem_class, optics.EnergyFilter],
                           'keywords': [detkwargs,
                                        {'filterfunc': interp1d(ccdqe['energy'],
                                                                ccdqe['qe']),
                                         'name': 'CCD QE'},
                                    ],
                       }
            ccd0_args = copy.deepcopy(ccd123_args)
            ccd0_args['keywords'][0]['id_num'] = 0
            ccd0_args['keywords'][0]['id_col'] = 'CCD_ID'
            # add optical blocking filter for CCD 0
            ccd0_args['elements'].append(optics.EnergyFilter)
            ccd0_args['keywords'].append({'filterfunc': interp1d(ccdqe['energy'],
                                                                 ccdqe['filtertrans']),
                                          'name': 'optical blocking filter'})
            det0 = optics.FlatStack(pos4d=detposlist[0], **ccd0_args)
            det123 = simulator.Parallel(elem_class=optics.FlatStack,
                                        elem_pos=detposlist[1:],
                                        elem_args=ccd123_args,
                                        id_num_offset=1,
                                        id_col='CCD_ID')

        super(Detectors, self).__init__(elements=[det0, det123])
        self.set_display()

    def set_display(self):
        self.elements[0].display = copy.deepcopy(self.elements[0].display)
        self.elements[0].display['color'] = (0., 0., 1.)
        for e in self.elements[1].elements:
            e.display = copy.deepcopy(e.display)
            e.display['color'] = (0., 0., 1.)


class FocalPlaneDet(marxs.optics.FlatDetector):
    loc_coos_name = ['detfp_x', 'detfp_y']
    detpix_name = ['detfppix_x', 'detfppix_y']

    def __init__(self, **kwargs):
        if ('zoom' not in kwargs) and ('pos4d' not in kwargs):
            kwargs['zoom'] = [.2, 500, 500]
        if ('orientation' not in kwargs) and ('pos4d' not in kwargs):
            kwargs['orientation'] = xyz2zxy[:3, :3]
        if ('position' not in kwargs) and ('pos4d' not in kwargs):
            kwargs['position'] = [0, 0, -100]
        super(FocalPlaneDet, self).__init__(**kwargs)


LGMLMirror.display['color'] = (1., 0., 1.)


class PerfectRedsox(simulator.Sequence):

    aper_class = CircleAperture
    mirr_class = SimpleMirror
    cat_class = CATGratings
    ml_class = MLMirrors

    def add_detectors(self, conf):
        return [Detectors]

    def post_process(self):
        self.KeepPos = simulator.KeepCol('pos')
        return [self.KeepPos]

    def __init__(self, channels=['1', '2', '3'], conf=conf,
                 **kwargs):
        list_of_classes = [self.aper_class, self.mirr_class,
                           self.cat_class, self.ml_class]
        elem = []
        for c in list_of_classes:
            if c is not None:
                elem.append(c(channels, conf))
        elem.extend(self.add_detectors(conf))

        super(PerfectRedsox, self).__init__(elements=elem,
                                            postprocess_steps=self.post_process(),
                                            **kwargs)
