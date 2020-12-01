from transforms3d import affines, euler
from marxs.design.tolerancing import select_1dof_changed
from astropy.table import Table
import astropy.units as u
import numpy as np

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
    element.coords = coords.directional_offset_by(position_angle, separation)


def plot_wiggle(tab, par, parlist, ax, axt=None,
                modfac='mod_mean', Aeff_col='Aeff_channel',
                axes_facecolor='w', plot_type=None):
    '''Plotting function for overview plot wiggeling 1 dof at the time.

    For parameters starting with "d" (e.g. "dx", "dy", "dz"), the plot axes
    will be labeled as a shift, for parameters tarting with "r" as rotation.

    Parameters
    ----------
    table : `astropy.table.Table`
        Table with wiggle results
    par : string
        Name of parameter to be plotted
    parlist : list of strings
        Name of all parameters in ``table``
    ax : `matplotlib.axes.Axes`
        Axis object to plot into.
    axt : ``None`` or  `matplotlib.axes.Axes`
        If this is ``None``, twin axis are created to show resolving power
        and effective area in one plot. Alternatively, a second axes instance
        can be given here.
    R_col : string
        Column name in ``tab`` that hold the resolving power to be plotted.
        Default is set to work with `marxs.design.tolerancing.CaptureResAeff`.
    Aeff_col : string
        Column name in ``tab`` that hold the effective area to be plotted.
    axes_facecolor : any matplotlib color specification
        Color for the background in the plot.
    '''
    import matplotlib.pyplot as plt

    t = select_1dof_changed(tab, par, parlist)
    t.sort(par)
    t_wave = t.group_by('wave')
    if (axt is None) and (plot_type is None):
        axt = ax.twinx()

    for key, g in zip(t_wave.groups.keys, t_wave.groups):
        if par[0] == 'd':
            x = g[par]
        elif par[0] == 'r':
            x = np.rad2deg(g[par].data)
        else:
            raise ValueError("Don't know how to plot {}. Parameter names should start with 'd' for shifts and 'r' for rotations.".format(par))

        if plot_type == 'MDP':
            ax.plot(x, np.abs(g['modulation'][:, 1]) * np.sqrt(g[Aeff_col]),
                    label='{:3.1f} $\AA$'.format(key[0]), lw=2 )
        elif plot_type == 'aeff':
            ax.plot(x, g[Aeff_col], label='{:2.0f} $\AA$'.format(key[0]), lw=2)
        elif plot_type == 'modulation':
             ax.plot(x, np.abs(g['modulation'][:, 1]), label='{:3.1f} $\AA$'.format(key[0]), lw=1.5)
        else:
            ax.plot(x, np.abs(g['modulation'][:, 1]), label='{:3.1f} $\AA$'.format(key[0]), lw=1.5)
            axt.plot(x, g[Aeff_col], ':', label='{:2.0f} $\AA$'.format(key[0]), lw=2)
    if plot_type == 'MDP':
        ax.set_ylabel('Figure of merrit')
        axlist = [ax]
    elif plot_type == 'aeff':
        ax.set_ylabel('$A_{eff}$ [cm$^2$] per channel')
        axlist = [ax]
    elif plot_type == 'modulation':
        ax.set_ylabel('Modulation factor')
        axlist = [ax]
    else:
        ax.set_ylabel('Modulation factor (solid lines)')
        axt.set_ylabel('$A_{eff}$ [cm$^2$] per channel (dotted lines)')
        axlist = [ax, axt]

    if par[0] == 'd':
        ax.set_xlabel('shift [mm]')
        ax.set_title('Shift along {}'.format(par[1]))
    elif par[0] == 'r':
        ax.set_xlabel('Rotation [degree]')
        ax.set_title('Rotation around {}'.format(par[1]))

    for a in axlist:
        a.set_facecolor(axes_facecolor)
        a.set_axisbelow(True)
        a.grid(axis='x', c='1.0', lw=2, ls='solid')



wiggle_plot_facecolors = {'global': '0.9',
                          'individual': (1.0, 0.9, 0.9)}
'''Default background colors for wiggle overview plots.

If the key of the dict matches part of the filename, the color listed in
the dict is applied.
'''

def load_and_plot(filename, parlist=['dx', 'dy', 'dz', 'rx', 'ry', 'rz'], **kwargs):
    '''Load a table with wiggle results and make default plot

    This is a function to generate a quicklook image with many
    hardcoded defaults for figure size, colors etc.
    In particular, this function is written for the display of
    6d plots which vary 6 degrees of freedom, one at a time.

    The color for the background in the plot is set depending on the filename
    using the ``string : color`` assignments in
    `~marxs.design.tolerancing.wiggle_plot_facecolors`. No fancy regexp based
    match is applied, this is simply a check with ``in``.

    Parameters
    ----------
    filename : string
        Path to a file with data that can be plotted by
        `~marxs.design.tolerancing.plot_wiggle`.

    parlist : list of strings
        Name of all parameters in ``table``.
        This function only plots six of them.

    kwargs :
        All other parameters are passed to
        `~marxs.design.tolerancing.plot_wiggle`.

    Returns
    -------
    tab : `astropy.table.Table`
        Table of data read from ``filename``
    fig : `matplotlib.figure.Figure`
        Figure with plot.

    '''
    import matplotlib.pyplot as plt

    tab = Table.read(filename)

    if 'axis_facecolor' not in kwargs:
        for n, c in wiggle_plot_facecolors.items():
            if n in filename:
                kwargs['axes_facecolor'] = c

    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(wspace=.6, hspace=.3)
    for i, par in enumerate(parlist):
        ax = fig.add_subplot(2, 3, i + 1)
        plot_wiggle(tab, par, parlist, ax, **kwargs)

    return tab, fig
