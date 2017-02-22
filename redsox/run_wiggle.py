from __future__ import print_function

import wiggle

n_photons = 1e4
outpath = '/melkor/d1/guenther/projects/REDSoX/sims/aeff/'

### Mirror ###
tab = wiggle.maketab_independent_move_rot([-1, 0, 1], [0, .1, .2])
tab = wiggle.run_wiggle(tab, wiggle.wiggle_det)
tab.write(outpath + 'wiggle_det.fits', overwrite=True)

'''
Wiggeling other parts of REDSoX could be done the same way, but we decided
not to continue this investigation for now, since the engineers said that
it will by *easy* to stay significantly below the tolerances that Herman
calculated analytically.

I can continue this later, if needed.
'''
