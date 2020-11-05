### Missing: Offset pointing

import os
import argparse
from copy import deepcopy, copy
from transforms3d import affines, euler

import numpy as np
import astropy.units as u
from astropy import table
from astropy.coordinates import SkyCoord
from marxs.simulator import Sequence
from marxs.design.tolerancing import (wiggle, moveglobal, moveindividual,
                                      varyattribute, varyorderselector,
                                      varyperiod,
                                      run_tolerances,
                                      generate_6d_wigglelist,
                                      run_tolerances_for_energies)
from marxs.optics import CATGrating
from marxs.source import JitterPointing, PointSource

import sys
sys.path.append('..')
from redsox.pisox import PerfectPisox, PiGrid
from redsox.analysis import fractional_aeff, calculate_modulation
from redsox import xyz2zxy

parser = argparse.ArgumentParser(description='Run tolerancing simulations.')
parser.add_argument('--n_photons', default=100000, type=int)
args = parser.parse_args()

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

my_source = PointSource(coords=SkyCoord(30., 0., unit='deg'), energy=0.25 * u.keV,
                        polarization=polfuncinthirds)

wave = np.array([30., 40., 50.]) * u.Angstrom
energies = wave.to(u.keV, equivalencies=u.spectral())

changeglobal, changeindividual = generate_6d_wigglelist([0., 0.01, .02, .1, .2, .4, .7, 1., 2., 5., 10.] * u.mm,
                                                        [0., .05, 2., 5., 10., 15., 20., 25., 30., 40., 50., 60., 120., 180.] * u.arcmin)


# in arcsec, conversion is a few lines below
scatter = np.array([0,  1., 4., 8., 16., 30., 60., 120., 180., 300, 600])
scatter = np.hstack([np.vstack([scatter, np.zeros_like(scatter)]),
                     np.vstack([np.zeros_like(scatter[1:]), scatter[1:]])])
scatter = np.deg2rad(scatter / 3600.).T

instrumfull = PerfectPisox()
outpath = '../run_results/pisox/'

def analyzer(photons):
    aeff = instrumfull.elements[0].area.to(u.cm**2) * fractional_aeff(photons[:len(photons) // 3])
    modulation = calculate_modulation(photons[len(photons) // 3:])
    return {'aeff': aeff, 'modulation': modulation,
            'Aeff_channel': aeff[1]}


def run_for_energies(instrum_before, wigglefunc, wiggleparts, parameters,
                     instrum, outfile, reset=None):
    dettab = run_tolerances_for_energies(my_source, energies,
                                         instrum_before, instrum,
                                         wigglefunc, wiggleparts, parameters,
                                         analyzer, reset=reset,
                                         t_source=args.n_photons * 3 * u.s)
    # For column with dtype object
    # This happens only when the input is the orderselector, so we can special
    # special case that here
    if 'order_selector' in dettab.colnames:
        o0 = dettab['order_selector'][0]
        if hasattr(o0, 'sigma'):
            dettab['sigma'] = [o.sigma for o in dettab['order_selector']]
        elif hasattr(o0, 'tophatwidth'):
            dettab['tophatwidth'] = [o.tophatwidth for o in dettab['order_selector']]
        dettab.remove_column('order_selector')
    #if 'coord' in dettab.colnames:
    #    dettab['ra_offset'] = [o.ra.arcmin for o in dettab['coord']]
    #    dettab['dec_offset'] = [o.dec.arcmin for o in dettab['coord']]
    #    dettab.remove_column(
    outfull = outpath + outfile
    dettab.write(outfull, overwrite=True)
    print('Writing {}'.format(outfull))


def filter_noCCD(photons):
    photons['probability'][~np.isfinite(photons['det_x'])] = 0
    return photons


class Pisox(PerfectPisox):
    def post_process(self):
        return []

    def __init__(self):
        super().__init__()
        self.elements.insert(0, JitterPointing(coords=SkyCoord(30., 0., unit='deg'),
                                               reference_transform=xyz2zxy,
                                               jitter=0 * u.arcsec))
        self.elements.append(filter_noCCD)

        # Shift the center of the grating assembly to the
        # position where the axes intersect the stair
        grids = self.elements_of_class(PiGrid)
        shift = np.eye(4)
        shift[2, 3] = grids[0].z_from_xy(0, 0)
        for e in grids:
            e.move_center(shift)


reset_6d = {'dx': 0., 'dy': 0., 'dz': 0., 'rx': 0., 'ry': 0., 'rz': 0.}


def moveelem(e, dx=0, dy=0, dz=0, rx=0., ry=0., rz=0.):
    '''Move and rotate marxs element around principal axes.

    Parameters
    ----------
    e :`marxs.optics.base.OpticalElement` or list of those elements
        Elements where uncertainties will be set
    dx, dy, dz : float
        translation in x, y, z (in mm)
    rx, ry, rz : float
        Rotation around x, y, z (in rad)
    '''
    if not hasattr(e.geometry, 'pos4d_orig'):
        e.geometry.pos4d_orig = e.geometry.pos4d.copy()
    move = affines.compose([dx, dy, dz],
                           euler.euler2mat(rx, ry, rz, 'sxyz'),
                           np.ones(3))
    e.geometry.pos4d = e.geometry.pos4d_orig @ move


# jitter
def dummy(p):
    '''Function needs something here, but nothing happens'''
    return p

# instrum = Pisox()
# run_for_energies(dummy, varyattribute, instrum.elements[0],
#                  [{'jitter': j} for j in np.array([0., 2., 3.5, 5., 8.,  10., 20., 30., 45.]) * u.arcmin],
#                  instrum,
#                  'jitter.fits')

# # SPO scatter
# instrum = Pisox()
# run_for_energies(Sequence(elements=instrum.elements[:2]), varyattribute,
#                  instrum.elements[2].elements[1],
#                  [{'inplanescatter': a, 'perpplanescatter':b} for a, b in scatter],
#                  Sequence(elements=instrum.elements[2:]),
#                  'scatter.fits')

# CATs
instrum = Pisox()
run_for_energies(Sequence(elements=instrum.elements[:4]), moveglobal,
                 instrum.elements[4].elements[:2],
                 changeglobal,
                 Sequence(elements=instrum.elements[4:]),
                 'CAT_global.fits')

# instrum = Pisox()
# run_for_energies(Sequence(elements=instrum.elements[:4]), wiggle,
#                  instrum.elements_of_class(PiGrid),
#                  changeindividual,
#                  Sequence(elements=instrum.elements[4:]),
#                  'CAT_individual.fits')

# # Period Variation
# instrum = Pisox()
# run_for_energies(Sequence(elements=instrum.elements[:4]), varyperiod,
#                  instrum.elements_of_class(CATGrating),
#                  [{'period_mean': 0.0002, 'period_sigma': s} for s in np.logspace(-6, -2, 13) * 0.0002],
#                  Sequence(elements=instrum.elements[4:]),
#                  'CAT_period.fits')

# # LGMLs
# instrum = Pisox()
# run_for_energies(Sequence(elements=instrum.elements[:5]), moveelem,
#                  instrum.elements[5],
#                  changeglobal,
#                  Sequence(elements=instrum.elements[5:]),
#                  'LGML_global.fits')

# factors = np.logspace(-4, -1, num=10)
# factors = np.hstack([1 - factors[::-1], 1 + factors])

# instrum = Pisox()
# run_for_energies(Sequence(elements=instrum.elements[:5]), varyattribute,
#                  instrum.elements[5],
#                  [{'lateral_gradient': s} for s in  factors * 1.6e-7],
#                  Sequence(elements=instrum.elements[5:]),
#                  'LGML_gradient.fits')
# # Changing the value in the center is equivalent to moving the mirror along
# # the long axis, so no need to simulate that here again.

# # Detectors
# instrum = Pisox()
# run_for_energies(Sequence(elements=instrum.elements[:-2]), moveglobal,
#                  instrum.elements[-2],
#                  changeglobal,
#                  Sequence(elements=instrum.elements[-2:]),
#                  'detector_global.fits')

# # Pointing errors
# # Don't want to be at RA = 0, because then negative number wrap around to 360.
# offsets = [.1, .2, .3, .4, .5, 1., 2, 5., 7., 9., 11.] * u.arcmin
# racoos = [{'coords': SkyCoord(30 * u.deg + a, 0. * u.deg)} for a in np.hstack([-offsets, [0] * u.deg, offsets])]
# # (0,0) is already simulated in the previous list. No need to do that again.
# deccoos = [{'coords': SkyCoord(30. * u.deg, a)} for a in np.hstack([-offsets, offsets])]

# instrum = Pisox()
# run_for_energies(dummy, varyattribute,
#                  instrum.elements[0],
#                  racoos + deccoos,
#                  instrum,
#                  'offset_point.fits')




# # CAT bending

# # SPOs
# # increase size of aperture to make sure light reaches SPOs.
# # In practice thermal precolimators etc.will impose further restrictions.
# instrum = Pisox()
# instrum.elements[1].elements[0].pos4d[0, 1] += np.max(trans_steps)
# instrum.elements[1].elements[0].pos4d[1, 2] += np.max(trans_steps)

# run_for_energies(Sequence(elements=instrum.elements[:2]), moveglobal,
#                  instrum.elements[2].elements[0],
#                  changeglobal,
#                  Sequence(elements=instrum.elements[2:]),
#                  'Mirrors_global.fits')


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
