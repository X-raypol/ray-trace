'''Currently, I use a very simple mirror model that is mostly defined
in redsox.py itself.  However, for effective area calculations, I need
a scaling factor to scale from the aperture used there to the real
geometric opening.
'''
from copy import deepcopy
import numpy as np
from astropy.table import Table
from astropy import units as u
from marxs import optics

from . import xyz2zxy

# Geometric opening
# From Specifications.txt
# radii in mm
tab = '''
shell  r_out r_int r_in theta
1 228.840 222.191 202.243 1.270
2 220.615 214.205 194.971 1.224
3 212.629 206.451 187.910 1.180
4 204.875 198.922 181.055 1.137
5 197.346 191.612 174.399 1.096
6 190.036 184.515 167.937 1.055
7 182.939 177.624 161.663 1.016
8 176.048 170.933 155.571 0.978
9 169.357 164.436 149.657 0.941
'''

shells = Table.read(tab, format='ascii')
shells['A'] = np.pi * (shells['r_out']**2 - shells['r_int']**2)

Ageom = shells['A'].sum() * u.mm**2
# Reverse order from inner to outer, because I took the display
# code below from Lynx that is in that order
shells.sort('r_out')


class MultiShellAperture(optics.MultiAperture):

    def __init__(self, channels, conf, **kwargs):
        elements = [optics.CircleAperture(position=[0, 0, conf['aper_z']],
                                          zoom=[1, shell['r_out'], shell['r_out']],
                                          r_inner=shell['r_int'],
                                          id_num=shell['shell'],
                                          orientation=xyz2zxy[:3, :3])
                    for shell in shells]
        kwargs['elements'] = elements
        kwargs['id_col'] = 'shell'
        super().__init__(**kwargs)
        disp = {'color': (0.0, 0.75, 0.75),
                'opacity': 0.3,
                'shape': 'triangulation',
                'outer_factor': 1.3,
                'inner_factor': 1.}
        for i in range(len(self.elements) - 1):
            self.elements[i].display = deepcopy(disp)
            self.elements[i].display['outer_factor'] = shells['r_int'][i + 1] / shells['r_out'][i]
        self.elements[0].display['inner_factor'] = 0
        self.elements[-1].display = deepcopy(disp)
        self.elements[-1].display['outer_factor'] = 1.5
