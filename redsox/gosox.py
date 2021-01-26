import copy

from marxs.optics.aperture import CircleAperture

from . import xyz2zxy
from .pisox import conf as piconf
from .pisox import PerfectPisox

conf = copy.deepcopy(piconf)

# Now update conf where GoSoX differs from PiSox
conf['f'] = 1190
conf['aper_z'] = 1290
conf['aper_zoom'] = [1, 75, 55]
conf['aper_rin'] = 80  # mirrors use this number too.
conf['shell_thick'] = 0.5
conf['ML'] = {'mirrorfile': 'ml_refl_smallsat.txt',
              'zoom': [0.25, 25., 5.],
              'pos': [0, 11 + 25, 0],
              'lateral_gradient': 1.6e-7,  # Ang/mm, converted to dimensionless
              'spacing_at_center': 1.6e-7 * 36,
              }
conf['det0pos'] = [0, 0, -15]  # Shift from zero to avoid overlap of detectors
conf['det1pos'] = [15, 11 + 25, 0]  # Shift from CCD center opposite ML center
conf['channels'] = ['1', '2', '3']


class PerfectGosox(PerfectPisox):
    def init_aper(self, conf):
        return CircleAperture(position=[0, 0, conf['aper_z']],
                              orientation=xyz2zxy[:3, :3],
                              zoom=[1, conf['mirr_rout'], conf['mirr_rout']],
                              r_inner=conf['aper_rin'])

    def __init__(self, conf=conf, **kwargs):
        super().__init__(conf, **kwargs)
