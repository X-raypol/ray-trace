### Missing: Offset pointing

import os
import argparse
from copy import deepcopy, copy

import numpy as np
import astropy.units as u
from astropy import table
from astropy.coordinates import SkyCoord
from marxs.simulator import Sequence
from marxs.design.tolerancing import (wiggle, moveglobal, moveindividual,
                                      moveelem,
                                      varyattribute, varyorderselector,
                                      varyperiod,
                                      run_tolerances,
                                      generate_6d_wigglelist,
                                      run_tolerances_for_energies2,
                                      oneormoreelements)
from marxs import optics
from marxs.source import JitterPointing, PointSource

import sys
sys.path.append('..')
from redsox.analysis import fractional_aeff, calculate_modulation
from redsox import xyz2zxy
from redsox import mlmirrors, gratings

parser = argparse.ArgumentParser(description='Run tolerancing simulations.')
parser.add_argument('mission', choices=['redsox', 'pisox', 'gosox'],
                    help='Select mission')
parser.add_argument('--n_photons', default=100000, type=int,
                    help='Number of photons per simulation (default 100,000')
parser.add_argument('-s',  '--scenario', action="extend", nargs="+", type=str,
                    help='Specify which scenarios to to run. If argument is not set, all test will be run.')
parser.add_argument('-e', '--exclude', action="extend", nargs="+", type=str,
                    help='Exclude specific scenarios from running')
args = parser.parse_args()


if args.mission == 'pisox':
    from redsox import pisox
    PerfectInstrum = pisox.PerfectPisox
    kwargs = {}
elif args.mission == 'gosox':
    from redsox import gosox, pisox
    PerfectInstrum = gosox.PerfectGosox
    kwargs = {}
elif args.mission == 'redsox':
    from redsox import redsox as mission_redsox
    PerfectInstrum = mission_redsox.PerfectRedsox
    kwargs = {'channels': '1'}

instrumfull = PerfectInstrum()


def filter_noCCD(photons):
    photons['probability'][~np.isfinite(photons['det_x'])] = 0
    return photons


class Instrum(PerfectInstrum):
    def post_process(self):
        return []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elements.insert(0, JitterPointing(coords=SkyCoord(30., 0.,
                                                               unit='deg'),
                                               reference_transform=xyz2zxy,
                                               jitter=0 * u.arcsec))
        self.elements.append(filter_noCCD)


def polfuncinthirds(time, energy):
    '''devide photon lists into thirds:
    first part is random orientation for calculate aeff,
    second part is angle 0, third part is angle 90 deg
    The second and third part are used to get the modulation.
    '''
    n = len(time)
    pol = np.random.uniform(0, 2 * np.pi, n) * u.rad
    pol[n // 3: (n // 3) * 2] = 0 * u.rad
    pol[(n // 3) * 2:] = np.pi / 2 * u.rad
    return pol


def analyzer(photons):
    aeff = instrumfull.elements[0].area.to(u.cm**2) * fractional_aeff(photons[:len(photons) // 3])
    modulation = calculate_modulation(photons[len(photons) // 3:])
    return {'aeff': aeff, 'modulation': modulation,
            'Aeff_channel': aeff[1]}


my_source = PointSource(coords=SkyCoord(30., 0., unit='deg'), energy=0.25 * u.keV,
                        polarization=polfuncinthirds)


## Here starts the list parameters ###
outpath = f'../run_results/{args.mission}/'

changeglobal, changeindividual = generate_6d_wigglelist([0., 0.01, .02, .1, .2, .4, .7, 1., 2., 5., 10.] * u.mm,
                                                        [0., .05, 2., 5., 10., 15., 20., 25., 30., 40., 50., 60., 120., 180.] * u.arcmin)


# in arcsec, conversion is a few lines below
scatter = np.array([0,  1., 4., 8., 16., 30., 60., 120., 180., 300, 600])
scatter = np.hstack([np.vstack([scatter, np.zeros_like(scatter)]),
                     np.vstack([np.zeros_like(scatter[1:]), scatter[1:]])])
scatter = np.deg2rad(scatter / 3600.).T

lgmlfactors = np.logspace(-4, -1, num=10)
lgmlfactors = np.hstack([1 - lgmlfactors[::-1], 1 + lgmlfactors])

# Pointing offsets
offsets = [.1, .2, .3, .4, .5, 1., 2, 5., 7., 9., 11.] * u.arcmin
racoos = [{'coords': SkyCoord(30 * u.deg + a, 0. * u.deg)} for a in np.hstack([-offsets, [0] * u.deg, offsets])]
# (0,0) is already simulated in the previous list. No need to do that again.
deccoos = [{'coords': SkyCoord(30. * u.deg, a)} for a in np.hstack([-offsets, offsets])]

if args.mission in ['pisox', 'gosox']:
    wave = [30, 40, 50] * u.Angstrom
    orders = ['all', 0, -1, -2]

    def increase_aperture_size_pisox(instrum, pars):
        '''increase size of aperture to make sure light reaches SPOs.

        In practice thermal precolimators etc.will impose further restrictions.
        '''
        if args.mission == 'pisox':
            pilens = instrum.elements_of_class(pisox.RectangleAperture)[0]
            pilens.pos4d[0, 1] += 20
        else:
            lens = instrum.elements_of_class(optics.aperture.CircleAperture)[0]
            lens.geometry._geometry['r_inner'] = lens.geometry['r_inner'] - 20
            lens.pos4d[[0, 1], [1, 2]] = lens.pos4d[0, 1] + 20

    runs = {'jitter': (JitterPointing, varyattribute,
                       [{'jitter': j} for j in np.array([0., 2., 3.5, 5., 8.,  10., 20., 30., 45.]) * u.arcmin]),
            'scatter': (optics.scatter.RadialMirrorScatter, varyattribute,
                        [{'inplanescatter': a, 'perpplanescatter':b} for a, b in scatter]),
            'CAT_global': (pisox.PiGrid, moveglobal, changeglobal),
            'CAT_individual': (pisox.PiGrid, wiggle, changeindividual),
            'CAT_period': (optics.CATGrating, varyperiod,
                           [{'period_mean': 0.0002, 'period_sigma': s} for s in np.logspace(-6, -2, 13) * 0.0002]),
            'LGML_global': (mlmirrors.LGMLMirror, moveelem, changeglobal),
            'LGML_gradient': (mlmirrors.LGMLMirror, varyattribute,
                              [{'lateral_gradient': s} for s in  lgmlfactors * 1.6e-7]),
            # Changing the value in the center is equivalent to moving the mirror along
            # the long axis, so no need to simulate that as a separate step.
            'detector_global': (pisox.Detectors, moveglobal, changeglobal),
            'offset_point': (JitterPointing, varyattribute, racoos + deccoos),
            # 'CAT_bending': (CATGrating)  -- still missing --
            'Mirrors_global': (pisox.PiLens, moveelem, changeglobal, increase_aperture_size_pisox),
            }


elif args.mission == 'redsox':
    wave = [35, 50, 65] * u.Angstrom
    orders = ['all', 0, -1, -2]

    runs = {'jitter': (JitterPointing, varyattribute,
                       [{'jitter': j} for j in np.array([0., 2., 3.5, 5., 8.,  10., 20., 30., 45.]) * u.arcmin]),
            'scatter': (optics.scatter.RadialMirrorScatter, varyattribute,
                        [{'inplanescatter': a, 'perpplanescatter':b} for a, b in scatter]),
            'CAT_global': (gratings.GratingGrid, moveglobal, changeglobal),
            'CAT_individual': (gratings.GratingGrid, wiggle, changeindividual),
            'CAT_period': (optics.CATGrating, varyperiod,
                           [{'period_mean': 0.0002, 'period_sigma': s} for s in np.logspace(-6, -2, 13) * 0.0002]),
            'LGML_global': (mlmirrors.LGMLMirror, moveelem, changeglobal),
            'LGML_gradient': (mlmirrors.LGMLMirror, varyattribute,
                              [{'lateral_gradient': s} for s in  lgmlfactors * 1.6e-7]),
            # Changing the value in the center is equivalent to moving the mirror along
            # the long axis, so no need to simulate that as a separate step.
            'detector_global': (mission_redsox.Detectors, moveglobal, changeglobal),
            'offset_point': (JitterPointing, varyattribute, racoos + deccoos),
            # 'CAT_bending': (CATGrating)  -- still missing --
            # Here: Aperture is held constant, assuming that mirrors move behind structure and into shade of
            # pre-collimators.
            'Mirrors_global': (mission_redsox.SimpleMirror, moveelem, changeglobal),
            }

energies = wave.to(u.keV, equivalencies=u.spectral())


'''
Format for the runs: dict with entries:  name: (element, wigglefunc, parameters, preparefunc)
name : string -
    Name of run, also use as filename
element : marx simulation elements
wigglefunc : function
parameters : list
preparefunc (optional) : function
    Will be excuted before the test run, use this to modify the intrument in preparation
'''

if args.scenario is None:
    scenarios = runs.keys()
else:
    scenarios = args.scenario[0]


print('Running the following scenarios:', scenarios)

for outfile in scenarios:
    pars = runs[outfile]
    instrum = Instrum(**kwargs)
    if len(pars) > 3:
        pars[3](instrum, pars)
    tab = run_tolerances_for_energies2(my_source, energies, instrum,
                                       pars[0], pars[1], pars[2],
                                       analyzer,
                                       t_source=3 * args.n_photons * u.s)
    outfull = outpath + outfile + '.fits'
    tab.write(outfull, overwrite=True)
    print('Writing {}'.format(outfull))



# Run default tolerance budget a few times
# n_budget = 50
# out = []

# conf = deepcopy(arcus.arcus.defaultconf)

# for i in range(n_budget):
#     print('Run default tolerance budget: {}/{}'.format(i, n_budget))
#     align = deepcopy(arcus.arcus.align_requirement_smith)
#     arcus.arcus.reformat_randall_errorbudget(align, globalfac=None)
#     conf['alignmentbudget'] = align
#     if i == 0:
#         arc = PerfectArcus(channels='1')
#     else:
#         arc = arcus.arcus.Arcus(channels='1', conf=conf)

#     for e in energies:
#         src.energy = e.to(u.keV).value
#         photons_in = src.generate_photons(args.n_photons)
#         photons_in = DefaultPointing()(photons_in)
#         photons = arc(photons_in)
#         # good = (photons['probability'] > 0) & (photons['CCD'] > 0)
#         # out([i, src.energy], photons[good], n_photons)
#         out.append(analyzer(photons))
#         out[-1]['energy'] = e.value
#         out[-1]['run'] = i

# tab = table.Table([{d: out[i][d].value
#                     if isinstance(out[i][d], u.Quantity) else out[i][d]
#                     for d in out[i]} for i in range(len(out))])

# tab['energy'].unit = u.keV
# tab['wave'] = tab['energy'].to(u.Angstrom, equivalencies=u.spectral())

# outfull = os.path.join(get_path('tolerances'), 'baseline_budget.fits')
# tab.write(outfull, overwrite=True)
# print('Writing {}'.format(outfull))
