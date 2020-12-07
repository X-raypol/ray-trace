import os
import copy
import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.table import Table
from numpy.core.umath_tests import inner1d


from marxs import optics, simulator
from marxs.math.utils import h2e
from marxs.math.utils import h2e, e2h, norm_vector
from marxs.math.polarization import parallel_transport

from transforms3d import affines
from transforms3d.euler import euler2mat

from .redsox import CATGratings, euler2aff
from .gratings import GratingGrid, bend_gratings
from .mlmirrors import LGMLMirror
from . import inputpath, xyz2zxy


conf = {'aper_z': 1400.,
        'aper_zoom': [1, 65, 75],
        'aper_rin': 65,  # mirrors use this number too.
        'mirr_rout': 200,
        'mirr_length': 100,  # mirrors should run from f to f - 100
                             # right now, plotting is symmetric to f
                             # so make longer
        'mirr_module_halfwidth': 75,
        'f': 1250,
        'shell_thick': 0.3,
        'n_shells': 17,
        'shell_half_opening_angle': np.deg2rad(18.),
        'inscat': 13 * u.arcsec,
        'perpscat': 0. * u.arcsec,
        'blazeang': 0.8 * u.degree,
        'gratingzoom': [0.25, 15, 5],
        'gratingframe': [0, 1.5, 1.5],
        'grating_ypos': [50, 200],
        'grating_z_bracket': [500, 1200],
        'd': 2e-4,
        'grat_id_offset': {'1': 0},
        'beta_lim': np.deg2rad([3.5, 5.05]),
        'ML': {'mirrorfile': 'ml_refl_smallsat.txt',
               'zoom': [0.25, 15., 5.],
               'pos': [0, 28, 0],
               'lateral_gradient': 1.6e-7,  # Ang/mm, converted to dimensionless
               'spacing_at_center': 1.6e-7 * 28,
        },
        'pixsize': 0.024,
        'detsize': [2048, 1024],
        'det0pos': [0, -13 , 0], # Shift from zero to avoid overlap of detectors
        'det1pos': [15, 28 + 7, 0], # Shift from CCD center opposite ML center
                                     # to avoid overlap of CCDs
        'bend': 900,
        #'bend': False,
        'rotchan': {'1': euler2aff(np.pi / 2, 0, 0, 'szyz')},
        }


refl_theory = {'period': np.array([21., 35., 53]),
               'lambda45': np.array([29.45, 48.80, 73.20]),
               'angle': np.deg2rad([35., 38., 41., 43., 45, 47, 49, 52., 55.]),
               'rp': np.array([[1.96e-02, 1.48e-02, 9.91e-03],
                               [1.25e-02, 7.60e-03, 4.73e-03],
                               [7.40e-04, 5.30e-03, 9.70e-03],
                               [1.70e-04, 1.04e-03, 1.60e-03],
                               [7.00e-06, 1.60e-04, 1.30e-03],
                               [3.34e-04, 3.90e-03, 1.70e-02],
                               [1.26e-03, 1.45e-02, 6.20e-02],
                               [1.58e-03, 1.52e-02, 1.78e-02],
                               [1.94e-03, 3.40e-02, 2.15e-02]]),
               'rs': np.array([[0.147,  0.118, 0.0927],
                               [0.180,  0.124, 0.0957],
                               [0.0447, 0.295, 0.418],
                               [0.0472, 0.309, 0.464],
                               [0.0503, 0.322, 0.515],
                               [0.0537, 0.344, 0.571],
                               [0.0577, 0.378, 0.632],
                               [0.0255, 0.176, 0.1246],
                               [0.0163, 0.201, 0.1340]])
               }
# The predicted Bragg peak lambda at 45 deg differs from the simple
# lambda = 2 D cos(45).
# So we don't use the period from the theoretical data, but get the D that
# we would expect for the given lambda, since that is the quantity we use in
# the lab when we calibrate this stuff.
refl_theory['period_lab'] = refl_theory['lambda45'] / np.cos(np.pi / 4.) / 2


class RectangleAperture(optics.RectangleAperture):
    display = {'color': (0.0, 0.75, 0.75),
               'opacity': 0.3,
               'outer_factor': 1.5,
               'shape': 'triangulation'}


def mirr_shell_table(tel_length, mirr_length, shell_thick, r_out, n_shells):

    r_out = [r_out]
    a = []
    f = []
    r_in = []
    for i in range(n_shells):
        a.append(.5 / r_out[-1] * (tel_length / r_out[-1] + np.sqrt((tel_length / r_out[-1])**2 + 1)))
        f.append(0.25 / a[-1])
        r_in.append(np.sqrt(r_out[-1]**2 - mirr_length / a[-1]))
        r_out.append(r_in[-1] - 2 * shell_thick)
    return Table({'shell': np.arange(n_shells) + 1,  # + 1 to start at 1, not 0
                  'r_out': r_out[:-1], 'a': a, 'f': f, 'r_in': r_in})


class PiLens(optics.PerfectLens):

    def __init__(self, conf, **kwargs):
        self.shells = mirr_shell_table(conf['f'], conf['mirr_length'],
                                       conf['shell_thick'], conf['mirr_rout'],
                                       conf['n_shells'])
        self.conf = conf
        super().__init__(**kwargs)

    def reflectivity(self, energy, angle):
        return 1 - 0.32  * np.rad2deg(angle) / 4.

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        # A ray through the center is not broken.
        # So, find out where a central ray would go.
        focuspoints = h2e(self.geometry['center']) + self.focallength * norm_vector(h2e(photons['dir'][intersect]))
        dir = e2h(focuspoints - h2e(interpos[intersect]), 0)
        pol = parallel_transport(photons['dir'].data[intersect, :], dir,
                                 photons['polarization'].data[intersect, :])
        # Single reflection, so get change in angle
        refl_angle = np.arccos(inner1d(norm_vector(dir),
                                       photons['dir'].data[intersect, :]))
        refl = self.reflectivity(photons['energy'][intersect], refl_angle / 2)
        r = np.sqrt(intercoos[intersect, 0]**2 + intercoos[intersect, 1]**2)
        phi = np.arctan(np.abs(intercoos[intersect, 1] /
                               intercoos[intersect, 0]))
        shell = np.zeros(intersect.sum())
        for s in self.shells:
            ind = (s['r_in'] < r) & (r < s['r_out']) & (phi < self.conf['shell_half_opening_angle'])
            shell[ind] = s['shell']
        refl[shell < 1] = 0
        #import pbd
        #pdb.set_trace()
        return {'dir': dir, 'polarization': pol, 'probability': refl,
                'refl_ang': refl_angle, 'shell': shell}



class SimpleMirror(optics.FlatStack):
    # Optics
    # Scatter as FWHM ~30 arcsec. Divide by 2.3545 to get Gaussian sigma.
    # Scatter numbers are still all wrong.
    spider_fraction = 0.95
    display = {'shape': 'cylinder',
               'color': (.7, .7, .7)}

    def refl(self):
        '''Read Ni reflectivity'''
        self.refl = Table.read(os.path.join(inputpath, 'ni_refl_1deg.txt'),
                               format='ascii.no_header', data_start=2,
                               names=['energy', 'refl'])
        self.refl['energy'] *= 1e-3  # convert eV to keV
        return self.refl

    def __init__(self, conf):
        refl = self.refl()
        kwords = [{'conf': conf, 'focallength': conf['f']},
                  {'inplanescatter': conf['inscat'],
                   'perpplanescatter': conf['perpscat']},
                  {'filterfunc': lambda x: np.ones_like(x) * self.spider_fraction,
                   'name': 'support spider'}]
        super().__init__(orientation=xyz2zxy[:3, :3],
                         position=[0, 0, conf['f']],
                         zoom=[conf['mirr_length'],
                               conf['mirr_rout'],
                               conf['mirr_module_halfwidth']],
                         elements=[PiLens,
                                   optics.RadialMirrorScatter,
                                   optics.EnergyFilter],
                         keywords=kwords)


def outsidemirror(photons):
    '''Remove photons that do not hit the active area of the mirror.
    '''
    photons['probability'][~np.isfinite(photons['shell'])] = 0.
    return photons



class PiGrid(GratingGrid):

    def elempos(self):
        '''This elempos makes a regular grid, very similar to Mark Egan's design.'''
        dx = 2 * self.conf['gratingzoom'][1] + 2 * self.conf['gratingframe'][1]
        dy = 2 * self.conf['gratingzoom'][2] + self.conf['gratingframe'][2]
        #x = np.arange(-2, 2.1) * dx
        y_pos = np.arange(self.y_in[0], self.y_in[1], dy)
        mx = []
        my = []
        for y in y_pos:
            # max x value that a ray can have at this y
            max_x = (np.abs(y) + 0.5 * self.conf['gratingzoom'][2]) * \
                np.tan(self.conf['shell_half_opening_angle'])
            # number of gratings in x direction needed to cover that
            # one grating at the center covers -.5 to .5
            n_x = max_x / dx
            mx.extend(np.arange(-np.round(n_x), np.round(n_x) + 0.1) * dx)
            my.extend([y] * int(2 * np.round(n_x) + 1))
        mx = np.array(mx)
        my = np.array(my)
        #mx, my = np.meshgrid(x, y)
        #mx = mx.flatten()
        #my = my.flatten()
        # Get z value
        # This class does not know if the ML mirror is tipped by +45 deg
        # or -45 deg. Yet, for -45 the stepping of the gratings should be
        # reversed with relation to +45 deg.
        # That's where the "-" in "-my[i]" comes from.
        z = np.array([self.z_from_xy(mx[i], -my[i]) for i in range(len(mx))])
        rg, gamma, beta = self.cart2sph(mx, my, z)

        ang_in = np.arctan2(self.mirrorpos['r_in'] - self.conf['gratingzoom'][2], self.mirrorpos['f'])
        ang_out = np.arctan2(self.mirrorpos['r_out'] + self.conf['gratingzoom'][2], self.mirrorpos['f'])

        ind = (np.abs(beta) > ang_in) & (np.abs(beta) < ang_out)
        return np.vstack([mx[ind], my[ind], z[ind], np.ones(ind.sum())]).T

    #def generate_elements(self):
    #    super().generate_elements()
    #    if self.bend:
    #        bend_gratings(self.elements, r_elem=self.bend)


class PiGratings(CATGratings):
    GratingGrid = PiGrid

    def __init__(self, conf, **kwargs):
        self.conf = conf
        super().__init__('1', conf, **kwargs)


class Detectors(simulator.Parallel):

    def __init__(self, conf):
        # 50 mn directly depositied on CCD + 30 nm on contamination filter
        al = Table.read('../inputdata/aluminuim_transmission_80nm.txt',
                        format='ascii.no_header',
                        data_start=2, names=['energy', 'transmission'])
        # 100 nm contamination filter
        poly = Table.read('../inputdata/polyimide_transmission.txt',
                          format='ascii.no_header',
                          data_start=2, names=['energy', 'transmission'])
        #qe = Table.read('../inputdata/qe.csv', format='ascii.ecsv')
        qe = Table.read('../inputdata/xgs_bi_ccdqe.dat',
                        format='ascii.no_header',
                        data_start=4,
                        names=['energy', 'qe', 'temp1', 'temp2'])
        qe['energy'] = 1e-3 * qe['energy']  # eV to keV


        al['energy'] = 1e-3 * al['energy']  # eV to keV
        poly['energy'] = 1e-3 * poly['energy']  # ev to keV


        detkwargs = {'pixsize': conf['pixsize']}
        detzoom = np.array([1, conf['pixsize'] * conf['detsize'][0] / 2,
                             conf['pixsize'] * conf['detsize'][1] / 2])

        detposlist = [affines.compose(conf['det0pos'],
                                      euler2mat(0, np.pi/2, 0, 'sxyz'),
                                      detzoom),
                      affines.compose(conf['det1pos'],
                                      euler2mat(0, 0, 0, 'sxyz'),
                                      detzoom),
        ]
        ccd_args = {'elements': (optics.FlatDetector,
                                 optics.EnergyFilter,
                                 optics.EnergyFilter,
                                 optics.EnergyFilter),
                    'keywords': (detkwargs,
                                 {'filterfunc': interp1d(al['energy'],
                                                         al['transmission']),
                                  'name': 'aluminium filter'},
                                 {'filterfunc': interp1d(poly['energy'],
                                                         poly['transmission']),
                                  'name': 'polyamide filter'},
                                 {'filterfunc': interp1d(qe['energy'],
                                                         qe['qe']),
                                  'name': 'CCD QE'},
                    ),
        }
        super().__init__(elem_class=optics.FlatStack,
                         elem_pos=detposlist,
                         elem_args=ccd_args,
                         id_col='CCD_ID')
        self.set_display()

    def set_display(self):
        self.disp = copy.deepcopy(self.elements[0].display)
        self.disp['color'] = (0., 0., 1.)
        self.disp['box-half'] = '+x'

        self.elements[0].display = self.disp
        self.elements[1].display = self.disp


def effective_baffle(photons):
    '''Only photons that hit the ML first are allowed onto the CCD 1

    Until the geometry of the baffle is specificed, I'm just running
    this with an "effective baffle" that filters out all non-wanted
    photons using columns that encapsulate the history of the photon.

    Thus should eventually be replaced by a proper geometric baffle.
    '''
    ind = photons['CCD_ID'] == 1
    ind2 = np.isfinite(photons['mlwave_nominal'])
    photons['probability'][ind & ~ind2] = 0.
    return photons


class PerfectPisox(simulator.Sequence):
    '''Intialization that onle a few lines is done here.
    When it gets longer it gets split out into a separate class.
    '''
    def init_aper(self, conf):
        x = conf['aper_rin'] - conf['mirr_rout']
        return optics.MultiAperture(elements=[
            RectangleAperture(orientation=euler2mat(0, -np.pi / 2, 0, 'sxyz'),
                              position=[x, 0, conf['aper_z']],
                              zoom=conf['aper_zoom']),
            RectangleAperture(orientation=euler2mat(0, -np.pi / 2, 0, 'sxyz'),
                              position=[-x, 0, conf['aper_z']],
                              zoom=conf['aper_zoom'])]
                                    )


    def init_ml(self, conf):
        c = conf['ML']
        datafile = os.path.join(inputpath, c['mirrorfile'])
        ml = LGMLMirror(position=c['pos'],
                        orientation=euler2mat(0, -np.pi / 4, 0, 'sxyz'),
                        zoom=c['zoom'],
                        datafile=datafile,
                        lateral_gradient = c['lateral_gradient'],
                        spacing_at_center = c['spacing_at_center'],
                        refl_theory=refl_theory)
        return ml

    def post_process(self):
        self.KeepPos = simulator.KeepCol('pos')
        return [self.KeepPos]

    def __init__(self, conf=conf, **kwargs):
        elem = [self.init_aper(conf),
                SimpleMirror(conf),
                outsidemirror,
                PiGratings(conf),
                self.init_ml(conf),
                Detectors(conf),
                effective_baffle,
        ]
        super().__init__(elements=elem, postprocess_steps=self.post_process(),
                         **kwargs)
