'''This is separate from run_tools to make it independent from redsox and marxs
so that I can import this from any python version even if redsox and marxs
cannot be imported there.
The underlying reason is that mayavi requires pyqt 4, while modern matplotlib requires
pyqt > 4. Thus, I cannot have both in the same conda environment and it's just convenient
to be able to get the wave for plotting without importing the rest.
'''

import numpy as np
import astropy.units as u

wave = np.arange(25., 75., 1.) * u.Angstrom
