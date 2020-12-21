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

# Herman gave me the second order table parameterised in the different way
# than the first order tables.
# I'll just hardcode it here for now.
# Also, I'll use a fixed width hardcoded below.
secondordertab='''
0.264	0.0080	6.5
0.295	0.0090	6.0
0.316	0.0100	5.6
0.340	0.0115	5.2
0.353	0.0130	5.0
0.375	0.0165	4.7
0.391	0.026	4.5
0.420	0.0017	4.2
0.465	0.0013	3.8
0.490	0.00084	3.6
0.595	0.00011	2.95
0.650	9.0e-5	2.7
0.765	6.8e-5	2.3'''

secord = Table.read(secondordertab, format='ascii.no_header', names=['en', 'R', 'D'])
secord['D'] = secord['D'] * 1e-6
secondamp = interpolate.interp1d(secord['D'], secord['R'], fill_value='extrapolate')


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
               'shape': 'box',
               'box-half': '+x',
    }

    def fresnel(self, photons, intersect, interpos, intercoos):
        '''The incident angle can easily be calculated from e_x and photons['dir'].'''
        d = self.D(intercoos[intersect, 0])
        dir = norm_vector(photons['dir'].data[intersect, :])
        arccosang = np.arccos(np.einsum('j,ij', -self.geometry['e_x'], dir))

        # get rs and rp from interpol of table
        rs = self.rs.ev(arccosang, d)
        rp = self.rp.ev(arccosang, d)
        scale = 2. / (rs + rp)
        return rs * scale, rp * scale

    def __init__(self, datafile, lateral_gradient, spacing_at_center,
                 refl_theory, **kwargs):
        self.lateral_gradient = lateral_gradient
        self.spacing_at_center = spacing_at_center
        data = Table.read(datafile, format='ascii.no_header', data_start=1,
                          names=['wave', 'R', 'M', 'width'])
        # Table is in Ang, but I use mm as unit of length
        data['wave'] = data['wave'] * 1e-7
        # second order
        # data2 = Table.read
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
        return self.lateral_gradient * x + self.spacing_at_center

    def specific_process_photons(self, photons,
                                 intersect, interpoos, intercoos):
        cosang = np.dot(photons['dir'].data[intersect, :],
                        -self.geometry['e_x'])
        D = self.D(intercoos[intersect, 0])
        wave_braggpeak = 2 * D * cosang
        wave_nominal = 2 * D * 2**(-0.5)
        amp = self.amp(wave_nominal)
        width = self.width(wave_nominal)
        gauss1order = Gaussian1D(amplitude=amp, mean=1.,
                                 stddev=width / 2.355)
        # Fixed hard-coded width
        gauss2order = Gaussian1D(amplitude=secondamp(D), mean=1.,
                                 stddev=0.02 / 2.355)
        wave = energy2wave / photons['energy'][intersect]

        out = super(LGMLMirror, self).specific_process_photons(photons,
                                                               intersect,
                                                               interpoos,
                                                               intercoos)
        g1 = gauss1order(wave / wave_braggpeak)
        g2 = gauss2order(wave / (wave_braggpeak / 2))

        return {'probability': (g1 + g2) * out['probability'],
                'mlwave_nominal': wave_nominal,
                'mlwave_braggpeak': wave_braggpeak,
                'mlcosang': cosang,
                'ml_x': intercoos[intersect, 0],
                'dir': out['dir'],
                'polarization': out['polarization']}
