'''Currently, I use a very simple mirror model that is mostly defined in redsox.py itself.
However, for effective area calculations, I need a scaling factor to scale from the aperture
used there to the real geometric opening.
'''
import numpy as np
from astropy.table import Table

# Geometric opening
# From Specifications.txt
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

t = Table.read(tab, format='ascii')
t['A'] = np.pi * (t['r_out']**2 - t['r_int']**2)
Ageom = t['A'].sum() / 100
