'''LGML (laterally graded multi-layer mirrors)

The laterally graded multi-layer mirrors are treated as two parts:
A "theoretical" mirror and a "measured correction".
The first part uses the relative reflection probability for p and s polarized
photons based on theoretical calculations, but discards the normalization of
the reflection table.
The second part scales that probability with the unpolarized reflection
probabilities which are determined in Hermann's lab.
'''

import numpy as np
from astropy.table import Table
from astropy.modeling.models import Gaussian1D
from scipy import interpolate

from marxs import optics
from marxs import energy2wave
from marxs.math.utils import norm_vector


refl_theory = {'period': np.array([24., 35., 53]),
               'lambda45': np.array([29.45, 48.80, 73.20]),
               'angle': np.deg2rad([41., 43., 45, 47, 49]),
               'rp': np.array([[7.40e-04, 5.30e-03, 9.70e-03],
                               [1.70e-04, 1.04e-03, 1.60e-03],
                               [7.00e-06, 1.60e-04, 1.30e-03],
                               [3.34e-04, 3.90e-03, 1.70e-02],
                               [1.26e-03, 1.45e-02, 6.20e-02]]),
               'rs': np.array([[0.0447, 0.295, 0.418],
                               [0.0472, 0.309, 0.464],
                               [0.0503, 0.322, 0.515],
                               [0.0537, 0.344, 0.571],
                               [0.0577, 0.378, 0.632]])
               }
# The predicted Bragg peak lambda at 45 deg differs from the simple
# lambda = 2 D cos(45).
# So we don't use the period from the theoretical data, but get the D that
# we would expect for the given lambda, since that is the quantity we use in
# the lab when we calibrate this stuff.
refl_theory['period_lab'] = refl_theory['lambda45'] / np.cos(np.pi / 4.) / 2


class LGMLMirror(optics.FlatBrewsterMirror):
    '''
    Parameters
    ----------
    datafile : string
        Path and name to a data table with lab measured reflectivity
        for unpolarized X-ray light.
    '''
    loc_coos_name = ['ml_x', 'ml_y']

    display = {'color': (1., 1., 1.),
               'shape': 'box'
    }

    def fresnel(self, photons, intersect, interpos, intercoos):
        '''The incident angle can easily be calculated from e_x and photons['dir'].'''
        d = self.D(intercoos[intersect, 0])
        dir = norm_vector(photons['dir'].data[intersect, :])
        arccosang = np.arccos(np.einsum('j,ij', -self.geometry('e_x'), dir))

        # get rs and rp from interpol of table
        rs = self.rs.ev(arccosang, d)
        rp = self.rp.ev(arccosang, d)
        scale = 2. / (rs + rp)
        return rs * scale, rp * scale

    def __init__(self, datafile, **kwargs):
        data = Table.read(datafile, format='ascii.no_header', data_start=1,
                          names=['wave', 'R', 'M', 'width'])
        self.rs = interpolate.RectBivariateSpline(refl_theory['angle'],
                                                  refl_theory['period_lab'],
                                                  refl_theory['rs'], ky=2)
        self.rp = interpolate.RectBivariateSpline(refl_theory['angle'],
                                                  refl_theory['period_lab'],
                                                  refl_theory['rp'], ky=2)

        self.amp = interpolate.interp1d(data['wave'], data['R'])
        self.width = interpolate.interp1d(data['wave'], data['width'])
        super(LGMLMirror, self).__init__(**kwargs)

    def D(self, x):
        '''Herman D(x) = 0.88 Ang/mm * x (in mm) + 26 Ang,
        where x is measured from the short wavelength end of the mirror.
        In marxs x is measured from the center, so we add 15 mm (the half-length.)
        '''
        return 0.88 * (x + 15) + 26

    def specific_process_photons(self, photons,
                                 intersect, interpoos, intercoos):
        cosang = np.dot(photons['dir'].data[intersect, :],
                        -self.geometry('e_x'))
        wave_braggpeak = 2 * self.D(intercoos[intersect, 0]) * cosang
        wave_nominal = 2 * self.D(intercoos[intersect, 0]) * 2**(-0.5)
        amp = self.amp(wave_nominal)
        width = self.width(wave_nominal)
        gaussians = Gaussian1D(amplitude=amp, mean=1.,
                               stddev=width / 2.355)
        wave = energy2wave / photons['energy'][intersect] * 1e7

        out = super(LGMLMirror, self).specific_process_photons(photons,
                                                               intersect,
                                                               interpoos,
                                                               intercoos)

        return {'probability': gaussians(wave / wave_braggpeak) * out['probability'],
                'mlwave_nominal': wave_nominal,
                'mlwave_braggpeak': wave_braggpeak,
                'mlcosang': cosang,
                'ml_x': intercoos[intersect, 0],
                'dir': out['dir'],
                'polarization': out['polarization']}
