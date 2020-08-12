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
import marxs.design.tolerancing as tol

from marxs.missions.mitsnl.catgrating import catsupportbars
from .gratings import GratingGrid
from .mlmirrors import LGMLMirror
from .tolerances import MirrorMover

from . import inputpath, xyz2zxy

def euler2aff(*args, **kwargs):
    mat = euler2mat(*args, **kwargs)
    return transforms3d.affines.compose(np.zeros(3), mat, np.ones(3))


conf = {'aper_z': 2900.,
        'aper_rin': 165.,
        'aper_rout': 223.,
        'mirr_rout': 228,
        'mirr_length': 300,
        'f': 2500,
        'inscat': 30. / 2.3545 * u.arcsec,
        'perpscat': 10. / 2.3435 * u.arcsec,
        'blazeang': 0.8 * u.degree,
        'gratingzoom': [0.25, 15, 5],
        'gratingframe': [0, 1.5, .5],
        'grating_z_bracket': [1e3, 2.5e3],
        'd': 2e-4,
        'rotchan': {'1': np.eye(4),
                    '2': euler2aff(np.pi * 2 / 3, 0, 0, 'szyz'),
                    '3': euler2aff(-np.pi * 2 / 3, 0, 0, 'szyz')},
        'grat_id_offset': {'1': 1000, '2': 2000, '3': 3000},
        'beta_lim': np.deg2rad([3.5, 5.05]),
        'grating_ypos': [50, 220],
        'ML': {'mirrorfile': 'ml_refl_2017_withCrSc_width.txt',
               'zoom': [0.25, 15., 5.],
               'pos': [44.55, 0, 0],
    # Herman D(x) = 0.88 Ang/mm * x (in mm) + 26 Ang,
    # where x is measured from the short wavelength end of the mirror.
    # In marxs x is measured from the center, so we add 15 mm (the half-length.)
               'lateral_gradient': 8.8e-8,  # Ang/mm converted to unitless
               'spacing_at_center': (0.88 * (0 + 15) + 26) * 1E-7,
        },
        'pixsize': 0.016,
        'detsize0': [408, 1608],
        'detsize123': [1632, 1608],
        'det1pos': [44.55, 25, 0],
        'bend': 1664,
        }


# Aperture
class CircleAperture(optics.CircleAperture):
    display = {'color': (0.0, 0.75, 0.75),
               'opacity': 0.3,
               'outer_factor': 1.5,
               'shape': 'triangulation'}

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
    display = {'shape': 'cylinder',
               'color': (.7, .7, .7)}

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
                                           zoom=[conf['mirr_length'],
                                                 conf['mirr_rout'],
                                                 conf['mirr_rout']],
                                           elements=[optics.PerfectLens,
                                                     optics.RadialMirrorScatter,
                                                     optics.EnergyFilter,
                                                     optics.EnergyFilter],
                                           keywords=kwords)


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
    colors = 'ryg'
    color_chan = {'1': 'r', '2': 'y', '3': 'g'}
    color_index = {'1': 0, '2': 1, '3': 2}
    GratingGrid = GratingGrid

    def __init__(self, channels, conf, **kwargs):

        elements = []

        # All all top sectors first so that a ray passing though overlapping top and
        # bottom sectors correctly passes through the top grating first.
        for chan in channels:
            gg = self.GratingGrid(channel=chan, conf=conf, y_in=[-conf['grating_ypos'][1], -conf['grating_ypos'][0]],
                             id_num_offset=conf['grat_id_offset'][chan]
                             # color_index=self.color_index[chan]
            )
            for e in gg.elements:
                e.display['color'] = self.color_chan[chan]
            elements.append(gg)
        # then add bottom sectors as requested
        for chan in channels:
            gg = self.GratingGrid(channel=chan, conf=conf, y_in=conf['grating_ypos'],
                             id_num_offset=conf['grat_id_offset'][chan] + 500
                             # color_index=self.color_index[chan]
            )
            for e in gg.elements:
                e.display['color'] = self.color_chan[chan]
            elements.append(gg)

        elements.append(catsupportbars)
        super().__init__(elements=elements, **kwargs)


class MLMirrors(simulator.Parallel):
    elem_class = LGMLMirror

    def __init__(self, channels, conf):
        c = conf['ML']
        lgmlpos = []
        lgmlpos1 = transforms3d.affines.compose(c['pos'],
                                                np.dot(euler2mat(-np.pi / 4, 0, 0, 'sxyz'),
                                                       xyz2zxy[:3, :3]),
                                                c['zoom'])
        if '1' in channels:
            lgmlpos.append(lgmlpos1)
        for chan in '23':
            if chan in channels:
                lgmlpos.append(np.dot(conf['rotchan'][chan], lgmlpos1))

        datafile = os.path.join(inputpath, c['mirrorfile'])
        super(MLMirrors, self).__init__(elem_class=self.elem_class,
                                        elem_pos=lgmlpos,
                                        elem_args={'datafile': datafile,
                                                   'lateral_gradient': c['lateral_gradient'],
                                                   'spacing_at_center': c['spacing_at_center']},
                                        id_num_offset=1,
                                        id_col='LGML')


class Detectors(simulator.Sequence):
    elem_class = optics.FlatDetector

    def filterqe(self):
        ccdqe = Table.read(os.path.join(inputpath, 'xgs_bi_ccdqe.dat'),
                           format='ascii.no_header', comment='!',
                           names=['energy', 'qe', 'filtertrans', 'temp'])
        ccdqe['energy'] = 1e-3 * ccdqe['energy']  # ev to keV
        return ccdqe

    def __init__(self, channels, conf):
        ccdqe = self.filterqe()

        detkwargs = {'pixsize': conf['pixsize']}
        detzoom0 = np.array([1, conf['pixsize'] * conf['detsize0'][0] / 2,
                             conf['pixsize'] * conf['detsize0'][1] / 2])
        detzoom = np.array([1, conf['pixsize'] * conf['detsize123'][0] / 2,
                            conf['pixsize'] * conf['detsize123'][1] / 2])
        # rotate by 45 deg
        rot = transforms3d.euler.euler2mat(np.pi / 4, 0, 0, 'sxyz')
        # flip to x-z plane
        flip = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        detposlist = [transforms3d.affines.compose(np.zeros(3),
                                                   xyz2zxy[:3, :3],
                                                   detzoom0)]
        #Position of the detector for channel 1
        det_channel = transforms3d.affines.compose(conf['det1pos'],
                                                   np.dot(flip, rot),
                                                   detzoom)
        for chan in channels:
            detposlist.append(np.dot(conf['rotchan'][chan], det_channel))
        ccd123_args = {'elements': (self.elem_class, optics.EnergyFilter),
                       'keywords': (detkwargs,
                                    {'filterfunc': interp1d(ccdqe['energy'],
                                                            ccdqe['qe']),
                                     'name': 'CCD QE'},
                                ),
                   }
        ccd0_args = copy.deepcopy(ccd123_args)
        ccd0_args['keywords'][0]['id_num'] = 0
        ccd0_args['keywords'][0]['id_col'] = 'CCD_ID'
        # add optical blocking filter for CCD 0
        ccd0_args['elements'] += (optics.EnergyFilter, )
        ccd0_args['keywords'] += ({'filterfunc': interp1d(ccdqe['energy'],
                                                             ccdqe['filtertrans']),
                                      'name': 'optical blocking filter'}, )
        det0 = optics.FlatStack(pos4d=detposlist[0], **ccd0_args)
        det123 = simulator.Parallel(elem_class=optics.FlatStack,
                                    elem_pos=detposlist[1:],
                                    elem_args=ccd123_args,
                                    id_num_offset=1,
                                    id_col='CCD_ID')

        super(Detectors, self).__init__(elements=[det0, det123])
        self.set_display()

    def set_display(self):
        self.disp = copy.deepcopy(self.elements[0].display)
        self.disp['color'] = (0., 0., 1.)
        self.disp['box-half'] = '+x'

        self.elements[0].display = self.disp
        for e in self.elements[1].elements:
            e.display = self.disp


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

    def add_detectors(self, channels, conf):
        return [Detectors(channels, conf)]

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
        elem.extend(self.add_detectors(channels, conf))

        super(PerfectRedsox, self).__init__(elements=elem,
                                            postprocess_steps=self.post_process(),
                                            **kwargs)


class MisalignedRedsox(PerfectRedsox):
    def __init__(self, channels=['1', '2', '3'], conf=conf,
                 **kwargs):
        super().__init__(conf=conf, channels=channels, **kwargs)
        move_mirror = MirrorMover(conf)
        for row in conf['alignmentbudget']:
            elem = self.elements_of_class(row[0])
            if row[1] == 'global':
                tol.moveglobal(elem, *row[3])
            elif row[1] == 'individual':
                tol.wiggle(elem, *row[3])
            elif row[1] == 'mirror':
                move_mirror(elem[0], *row[3])
            else:
                raise NotImplementedError('Alignment error {} not implemented'.format(row[1]))


def reformat_errorbudget(budget, globalfac=0.8):
    '''
    Also, units need to be converted: mu -> mm, arcsec -> rad
    Last, global misalignment (that's not random) must be
    scaled in some way. Here, I use 0.8 sigma, which is the
    mean absolute deviation for a Gaussian.

    Parameters
    ----------
    budget : list
        See reference implementation for list format
    globalfac : ``None`` or float
        Factor to apply for global tolerances. A "global" tolerance is drawn
        only once per simulation. In contrast, for "individual" tolerances
        many draws are done and thus the resulting layout actually
        represents a distribution. For a "global" tolerance, the result hinges
        essentially on a single random draw. If this is set to ``None``,
        misalignments are drawn statistically. Instead, the toleracnes can be
        scaled determinisitically, e.g. by "0.8 sigma" (the mean absolute
        deviation for a Gaussian distribution).
    '''
    for row in budget:
        tol = np.zeros(6)
        for i in [0, 1, 2]:
            tol[i] = row[2][i].to(u.mm).value
        for i in [3, 4, 5]:
            tol[i] = row[2][i].to(u.rad).value
        if (row[1] == 'global') or (row[1] == 'mirror'):
            if globalfac is not None:
                tol *= globalfac
            else:
                tol *= np.random.randn(len(tol))

        row[3] = tol


align_requirement_moritz = [
    #[SimpleMirror, 'mirror', [.1*u.mm, .1*u.mm, 5*u.mm, .5*u.degree, .5*u.degree, 10*u.degree],
    # None, 'Mirror with respect to its center of mass'],
    [GratingGrid, 'global', [1*u.mm, 1*u.mm, 1*u.mm, .1*u.degree, .1*u.degree, 1*u.degree],
     None, 'Grating petal to structure'],
    [GratingGrid, 'individual', [.5*u.mm, .5*u.mm, .5*u.mm, .1*u.degree, .1*u.degree, 1*u.degree],
     None, 'CAT grating to petal'],
    [MLMirrors, 'global', [.1*u.mm, .2*u.mm, 1*u.mm, .25*u.degree, .25*u.degree, 1*u.degree],
     None, 'ML mirror'],
    #[Detectors, 'global', [2*u.mm, 2*u.mm, 2*u.mm, 5*u.degree, 5*u.degree, 5*u.degree],
    # None, 'individual CAT to window'],
]

align_requirement = copy.deepcopy(align_requirement_moritz)
reformat_errorbudget(align_requirement, 1.)

conf['alignmentbudget'] = align_requirement
