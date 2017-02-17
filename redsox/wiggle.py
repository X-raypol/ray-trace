from __future__ import print_function

import copy
import numpy as np
from astropy.table import Table
from transforms3d.affines import compose
from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2mat
import redsox
import run_tools


def maketab_independent_move_rot(move, rot):
    '''Make a wiggle table where each x,y,z is moves / rotated keeping
    all others fixed at 0.'''
    n = 3 * len(move) + 3 * len(rot)
    nmove  = 3 * len(move)
    t = Table()
    for i, c in enumerate('xyz'):
        t['trans_' + c] = np.zeros(n)
        t['trans_' + c][i * len(move): (i + 1) * len(move)] = move
        t['rot_' + c] = np.zeros(n)
        t['rot_' + c][nmove + i * len(rot): nmove + (i + 1) * len(rot)] = rot
    # common case is to have a row 0,0,0,0,0,0 in all sections.
    # Need that only once.
    ind = (t['trans_x'] == 0) & (t['trans_y'] == 0) & (t['trans_z'] == 0) & (t['rot_x'] == 0) & (t['rot_y'] == 0) & (t['rot_z'] == 0)
    ind = ind.nonzero()[0]
    t.remove_rows(ind[1:])  # do not remove row with index 0, because we want to keep one of them.
    return t


def affinelemframe(elem, row):
    g = elem.geometry
    trans = g('e_x')[:3] * row['trans_x'] + g('e_y')[:3] * row['trans_y'] + g('e_z')[:3] * row['trans_z']
    rot = np.dot(axangle2mat(g('e_z')[:3], row['rot_z']),
                 np.dot(axangle2mat(g('e_y')[:3], row['rot_y']),
                        axangle2mat(g('e_x')[:3], row['rot_x']))
          )
    return compose(trans, rot, np.ones(3))


def affglobal(row):
    trans = [row['trans_x'], row['trans_y'], row['trans_z']]
    rot = euler2mat(row['rot_x'], row['rot_y'], row['rot_z'], 'sxyz')
    return compose(trans, rot, np.ones(3))


def wiggle_det(row):
    mission = copy.deepcopy(redsox.redsox)
    det123 = mission.elements[-1].elements[-1]
    for e1 in det123.elements:
        for e2 in e1.elements:
            e2.pos4d = np.dot(affinelemframe(e2, row), e2.pos4d)
    return mission


def wiggle_ml(row):
    mission = copy.deepcopy(redsox.redsox)
    ml123 = mission.elements[-2]
    for e1 in ml123.elements:
        e1.pos4d = np.dot(affinelemframe(e1, row), e1.pos4d)
    return mission


def wiggle_mirror(row):
    mission = copy.deepcopy(redsox.redsox)
    mirr = mission.elements[2]
    for e1 in mirr.elements:
        e1.pos4d = np.dot(affinelemframe(e1, row), e1.pos4d)
    return mission


def wiggle_gratings(row):
    mission = copy.deepcopy(redsox.redsox)
    grat123 = mission.elements[2].elements[:3]
    for sect in grat123:
        for e in sect:
            randrow = {}
            for n in ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']:
                randrow[n] = np.random.randn() * row[n]
            e.pos4d = np.dot(affinelemframe(e, randrow), e.pos4d)
    return mission


def run_wiggle(tab, wigglefunc):
    for n in ['aeff', 'modulation']:
        tab[n] = np.nan * np.ones((len(tab), len(run_tools.energies), 4))
    for i in range(len(tab)):
        print("--- Running wiggle {0} of {1} ---".format(i + 1, len(tab)))
        mission = wigglefunc(tab[i])
        tab['aeff'][i] = run_tools.run_aeff(mission=mission)
        tab['modulation'][i] = run_tools.run_modulation(mission=mission)
    tab['merrit'] = np.sqrt(tab['aeff'].data[:, 1]) * tab['modulation'].data[:, 1]
    return tab
