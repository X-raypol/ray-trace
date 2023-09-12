from transforms3d import affines, euler
from marxs.design.tolerancing import WigglePlotter
from astropy.table import Table
import astropy.units as u
import numpy as np
from astropy import coordinates
from . import xyz2zxy


class MirrorMover():
    def __init__(self, conf):
        self.conf = conf

    def __call__(self, element, dx=0, dy=0, dz=0, rx=0., ry=0., rz=0.):
        pos = affines.compose([dx, dy, dz + self.conf['f']],
                              euler.euler2mat(rx, ry, rz, 'sxyz') @ xyz2zxy[:3, :3],
                              [self.conf['mirr_length'],
                               self.conf['mirr_rout'],
                               self.conf['mirr_rout']])
        element.geometry.pos4d[:] = pos


def offsetpoint(element, position_angle, separation):
    element.coords = element.coords.directional_offset_by(position_angle,
                                                          separation)


class MDPPlotter(WigglePlotter):
    ylabel = 'Figure of merrit'
    y2label = None

    def plot_one_line(self, ax, axt, key, g, x, Aeff_col='Aeffgrat'):
        ax.plot(x, np.abs(g['modulation'][:, 1]) * np.sqrt(g[Aeff_col]),
                    label='{:3.1f} $\AA$'.format(key[0]), lw=2 )

class ModulationPlotter(WigglePlotter):
    ylabel = 'Modulation factor (solid lines)'
    y2label = '$A_{eff}$ [cm$^2$] per channel (dotted lines)'

    def plot_one_line(self, ax, axt, key, g, x, Aeff_col='Aeffgrat'):
        ax.plot(x, np.abs(g['modulation'][:, 1]), label='{:3.1f} $\AA$'.format(key[0]), lw=1.5)
        axt.plot(x, g[Aeff_col], ':', label='{:2.0f} $\AA$'.format(key[0]), lw=2)
