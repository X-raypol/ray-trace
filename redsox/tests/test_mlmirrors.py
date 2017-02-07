import os
import numpy as np
from astropy.table import Table
from transforms3d.euler import euler2mat
from ..mlmirrors import LGMLMirror
from .. import redsox
from marxs import energy2wave


def test_total_refl_probability():
    '''The theoretical values for the s and p reflectivity shall be normed
    to match the observed reflectivity of unpolarized light.'''

    pos = np.array([[1., 0., 0., 1],
                    [1., 0., 0., 1]])
    dir = np.array([[-1., 0., 0., 0],
                    [-1., 0., 0., 0]])
    polarization = np.array([[0., 1., 0., 0],
                             [0, 0., 1., 0]])
    wave_at_x0 = 2. * (0.88 * 15 + 26) * np.cos(np.pi / 4) * 1e-7
    en_at_x0 = energy2wave / wave_at_x0
    photons = Table({'pos': pos, 'dir': dir,
                     'energy': [en_at_x0, en_at_x0],
                     'polarization': polarization,
                     'probability': [1., 1.]})
    mir = LGMLMirror(os.path.join(redsox.inputpath, 'ml_refl_2015_minimal.txt'),
                     orientation=euler2mat(-np.pi / 4, 0, 0, 'syxz'))

    photons = mir(photons)
    assert np.isclose(photons['probability'].sum(), 2. * 0.0858447606431)
