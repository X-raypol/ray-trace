import os
import argparse
import copy
import numpy as np
import astropy.units as u
from astropy import table
from astropy.coordinates import SkyCoord
from marxs.simulator import Sequence
from marxs.optics import CATGrating
from marxs.design.tolerancing import (wiggle, moveglobal, moveindividual,
                                      varyattribute,
#                                      varyorderselector,
                                      varyperiod,
                                      run_tolerances,
                                      generate_6d_wigglelist)
import arcus.tolerances as tol
from marxs.source import PointSource, FixedPointing, JitterPointing

import sys
sys.path.append('..')
from redsox.redsox import PerfectRedsox, MisalignedRedsox, conf, xyz2zxy
from redsox.analysis import fractional_aeff, calculate_modulation
from redsox.gratings import GratingGrid
from redsox.tolerances import MirrorMover, offsetpoint
from redsox.redsox import align_requirement_moritz, reformat_errorbudget


def analyzer(photons):
    aeff = fractional_aeff(photons[: args.n_photons])
    modulation = calculate_modulation(photons[args.n_photons:])
    return {'Aeff': aeff, 'modulation': modulation,
            'Aeff_channel': np.mean(aeff[1:]),
            'mod_mean': -np.mean(modulation[[1, 3]]),
            }


parser = argparse.ArgumentParser(description='Run tolerancing simulations.')
parser.add_argument('outdir', help='directory where output is written')
parser.add_argument('n_photons', default=100000, type=int)
args = parser.parse_args()


coords = SkyCoord(23., -45., unit='deg')
src = PointSource(coords=coords, energy=0.5, flux=1.)


wave = np.array([40, 55, 70.]) * u.Angstrom
energies = wave.to(u.keV, equivalencies=u.spectral())

trans_steps = [0., .1, .2, .4, .7, 1., 2., 5., 10.] * u.mm
rot_steps = [0., 2., 5., 10., 20., 40., 60., 120.] * u.arcmin
changeglobal, changeindividual = generate_6d_wigglelist(trans_steps, rot_steps)


scatter = np.array([5., 10, 20, 30, 45, 60, 120, 240]) / 2.3545
scatter = np.hstack([np.vstack([scatter, np.zeros_like(scatter)]),
                     np.vstack([np.zeros_like(scatter[1:]), scatter[1:]])])
scatter = np.deg2rad(scatter / 3600.).T


def run_for_energies(instrum_before, wigglefunc, wiggleparts, parameters,
                     instrum, outfile, reset=None):
    outtabs = []
    for i, e in enumerate(energies):
        src.energy = e.to(u.keV).value
        src.polarization = None
        p_rand = src.generate_photons(args.n_photons)
        src.polarization = 0.
        p_0 = src.generate_photons(args.n_photons)
        src.polarization = 90.
        p_90 = src.generate_photons(args.n_photons)

        photons_in = table.vstack([p_rand, p_0, p_90])

        photons_in = instrum_before(photons_in)
        data = run_tolerances(photons_in, instrum,
                              wigglefunc, wiggleparts,
                              parameters, analyzer)
        # convert tab into a table.
        # astropy.tables has problems with Quantities as input
        tab = table.Table([{d: data[i][d].value
                            if isinstance(data[i][d], u.Quantity) else data[i][d]
                            for d in data[i]} for i in range(len(data))])
        tab['energy'] = e
        tab['wave'] = wave[i]
        outtabs.append(tab)
    # Reset positions and stuff so that same instance of instrum can be used again
    if reset is not None:
        wigglefunc(wiggleparts, **reset)
    dettab = table.vstack(outtabs)
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
    outfull = os.path.join(args.outdir, outfile)
    dettab.write(outfull, overwrite=True)
    print('Writing {}'.format(outfull))


class Redsox(PerfectRedsox):
    def post_process(self):
        return []

    def __init__(self, conf=conf):
        super().__init__(conf=conf)
        self.elements.insert(0, FixedPointing(coords=coords, reference_transform=xyz2zxy))
        # Move center of rotation for each grating sector to the center of that sector
        ggrids = self.elements_of_class(GratingGrid)
        for gg in ggrids:
            shift = np.eye(4)
            shift[:, 3] = np.mean(gg.elempos(), axis=0)
            gg.move_center(shift)


# Tests are ordered by importance for the work I'm doing right now

# CATs



run_for_energies(Sequence(elements=instrum.elements[:3]), moveglobal,
                 instrum.elements_of_class(GratingGrid),
                 changeglobal,
                 Sequence(elements=instrum.elements[3:]),
                 'CAT_global.fits',
                 reset=changeglobal[0])

instrum = Redsox()
run_for_energies(Sequence(elements=instrum.elements[:3]), wiggle,
                 instrum.elements_of_class(GratingGrid),
                 changeindividual,
                 Sequence(elements=instrum.elements[3:]),
                 'CAT_individual.fits',
                 reset=changeglobal[0])


# # jitter
def dummy(p):
    '''Function needs something here, but nothing happens'''
    return p

# instrum = Redsox()
instrum.elements[0] = JitterPointing(coords=coords, reference_transform=xyz2zxy, jitter=0)
run_for_energies(dummy, varyattribute, instrum.elements[0],
                 [{'jitter': j} for j in np.array([5., 10., 30., 60., 120., 180., 300.]) * u.arcsec],
                 instrum,
                 'jitter.fits',)
instrum.elements[0] = FixedPointing(coords=coords, reference_transform=xyz2zxy)

# Mirror scatter
instrum = Redsox()
backup1 = instrum.elements[2].elements[1].inplanescatter
backup2 = instrum.elements[2].elements[1].perpplanescatter

run_for_energies(Sequence(elements=instrum.elements[:2]), varyattribute,
                 instrum.elements[2].elements[1],
                 [{'inplanescatter': a, 'perpplanescatter': b} for a, b in scatter],
                 Sequence(elements=instrum.elements[2:]),
                 'scatter.fits')

instrum.elements[2].elements[1].inplanescatter = backup1
instrum.elements[2].elements[1].perpplanescatter = backup2

# Multilayer mirrors
instrum = Redsox()
run_for_energies(Sequence(elements=instrum.elements[:4]), moveindividual,
                 instrum.elements[4],
                 changeglobal,
                 Sequence(elements=instrum.elements[4:]),
                 'mlmirror_global.fits',
                 reset=changeglobal[0])

# detectors
instrum = Redsox()
run_for_energies(Sequence(elements=instrum.elements[:5]), moveindividual,
                 instrum.elements[5].elements[1],
                 changeglobal,
                 Sequence(elements=instrum.elements[5:]),
                 'detector_global.fits',
                 reset=changeglobal[0])

# Period Variation
instrum = Redsox()
run_for_energies(Sequence(elements=instrum.elements[:3]), varyperiod,
                 instrum.elements_of_class(CATGrating),
                 [{'period_mean': 0.0002, 'period_sigma': s} for s in np.logspace(-3, -1, 4) * 0.0002],
                 Sequence(elements=instrum.elements[3:]),
                 'CAT_period.fits')


# CAT surfaceflatness
instrum = Redsox()
run_for_energies(Sequence(elements=instrum.elements[:3]), varyattribute,
                 instrum.elements_of_class(CATGrating),
                 [{'order_selector': tol.OrderSelectorWavy(wavysigma=s)} for s in np.deg2rad([0., .1, .2, .4, .6, .8, 1.])],
                 Sequence(elements=instrum.elements[3:]),
                 'CAT_flatness.fits')

# CAT buckeling
run_for_energies(Sequence(elements=instrum.elements[:3]), varyattribute,
                 instrum.elements_of_class(CATGrating),
                 [{'order_selector': tol.OrderSelectorTopHat(tophatwidth=s)} for s in np.deg2rad([0., .25, .5, .75, 1., 1.5, 2., 3., 5])],
                 Sequence(elements=instrum.elements[3:]),
                 'CAT_buckeling.fits')

offsets = []
for off in [0, 1, 2.5, 5., 7.5, 10., 15] * u.arcsec:
    for ang in [0, 90] * u.degree:
        offsets.append({'separation': off, 'position_angle': ang})

instrum = Redsox()
run_for_energies(dummy, offsetpoint,
                 instrum.elements[0],
                 offsets,
                 instrum,
                 'offset_point.fits')

move_mirror = MirrorMover(conf)

run_for_energies(Sequence(elements=instrum.elements[:2]), move_mirror,
                 instrum.elements[2],
                 changeglobal,
                 Sequence(elements=instrum.elements[2:]),
                 'mirror_global.fits',
                 reset=changeglobal[0])

# still missing: Vary r for bend gratings


## Step 2
n_budget = 100
out = []

pnt = FixedPointing(coords=coords, reference_transform=xyz2zxy)

for i in range(n_budget):
    print('Run default tolerance budget: {}/{}'.format(i, n_budget))
    if i == 0:
        instrum = PerfectRedsox()
    else:
        align_requirement = copy.deepcopy(align_requirement_moritz)
        reformat_errorbudget(align_requirement, None)
        conf['alignmentbudget'] = align_requirement

        instrum = MisalignedRedsox(conf=conf)

    for e in energies:
        src.energy = e.to(u.keV).value
        src.polarization = None
        p_rand = src.generate_photons(args.n_photons)
        src.polarization = 0.
        p_0 = src.generate_photons(args.n_photons)
        src.polarization = 90.
        p_90 = src.generate_photons(args.n_photons)
        photons_in = table.vstack([p_rand, p_0, p_90])

        photons_in = pnt(photons_in)
        photons = instrum(photons_in)
        out.append(analyzer(photons))
        out[-1]['energy'] = e.value
        out[-1]['run'] = i

    tab = table.Table([{d: out[i][d].value
                        if isinstance(out[i][d], u.Quantity) else out[i][d]
                        for d in out[i]} for i in range(len(out))])

    tab['energy'].unit = u.keV
    tab['wave'] = tab['energy'].to(u.Angstrom, equivalencies=u.spectral())

    outfull = os.path.join(args.outdir, 'moritz_budget.fits')
    tab.write(outfull, overwrite=True)
    print('Writing {}'.format(outfull))
