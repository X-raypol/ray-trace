import os
from warnings import warn

import numpy as np
from scipy import optimize
import astropy.units as u
from astropy.table import Table
from marxs.math.geometry import Cylinder
from marxs.math.utils import h2e
from marxs.optics import OpticalElement
from marxs.simulator import ParallelCalculated
from marxs.missions.mitsnl.catgrating import (CATL1L2Stack,
                                              InterpolateEfficiencyTable)
from transforms3d.axangles import axangle2mat
from transforms3d.affines import decompose

from . import inputpath

globalorderselector = InterpolateEfficiencyTable(Table.read(os.path.join(inputpath,
                                                              'efficiency.csv'),
                                                            format='ascii.ecsv'))

# Fix color index coloring and activate in redsox again


def blazemat(blazeang):
    return axangle2mat(np.array([0, 0, 1]), blazeang.to(u.rad).value)


def bend_gratings(gratings, r_elem=1664):
    '''Bend gratings in a gas to follow the Rowland cirle

    Gratings are bend in one direction (the dispersion direction) only.

    Assumes that the focal point is at the origin of the coordinate system!

    Parameters
    ----------
    gratings : list
        List of gratings to be bend
    '''
    for e in gratings:
        t, rot, z, s = decompose(e.geometry.pos4d)
        d_phi = np.arctan(z[1] / r_elem)
        c = Cylinder({'position': t - r_elem * h2e(e.geometry['e_x']),
                      'orientation': rot,
                      'zoom': [r_elem, r_elem, z[2]],
                      'phi_lim': [-d_phi, d_phi]})
        c._geometry = e.geometry._geometry
        e.geometry = c
        e.display['shape'] = 'surface'
        for e1 in e.elements:
            # can't be the same geometry, because groove_angle is part of _geometry and that's different
            # Maybe need to get that out again and make the geometry strictly the geometry
            # But for now, make a new cylinder of each of them
            # Even now, not sure that's needed, since intersect it run by FlatStack
            c = Cylinder({'position': t - r_elem * h2e(e.geometry['e_x']),
                          'orientation': rot,
                          'zoom': [r_elem, r_elem, z[2]],
                          'phi_lim': [-d_phi, d_phi]})
            c._geometry = e1.geometry._geometry
            e1.geometry = c
            e1.display['shape'] = 'surface'


class ColoringGrating(CATL1L2Stack):
    def __init__(self, **kwargs):
        self.colorindex = kwargs.pop('color_index')
        super(ColoringGrating, self).__init__(**kwargs)

    def specific_process_photons(self, photons, intersect, interpos,
                                 intercoos):
        # Not sure if this ever gets called, but can check / debug later
        out = super(ColoringGrating, self).specific_process_photons(photons, intersect, interpos, intercoos)
        out['colorindex'] = [self.colorindex] * intersect.sum()
        return out


class GratingGrid(ParallelCalculated, OpticalElement):
    id_col = 'facet'
    elem_class = CATL1L2Stack
    order_selector = globalorderselector

    def __init__(self, channel, conf, y_in, **kwargs):
        self.G = conf['ML']['lateral_gradient']
        kwargs['pos_spec'] = self.elempos
        if 'normal_spec' not in kwargs.keys():
            kwargs['normal_spec'] = np.array([0., 0., 0., 1.])
        if 'parallel_spec' not in kwargs.keys():
            kwargs['parallel_spec'] = np.array([0., 1., 0., 0.])
        self.conf = conf
        self.bend = self.conf['bend']
        self.channel = channel
        self.y_in = y_in
        kwargs['elem_class'] = self.elem_class
        kwargs['elem_args'] = {'d': conf['d'],
                               'order_selector': self.order_selector,
                               'zoom': conf['gratingzoom'],
                               'orientation': np.dot(axangle2mat([1, 0, 0], np.pi / 2),
                                                     blazemat(conf['blazeang'])),
                               #'color_index': kwargs.pop('color_index'),
                               }
        self.mirrorpos = {'f': conf['f'],
                          'r_in': conf['aper_rin'],
                          'r_out': conf['mirr_rout'],
        }
        self.z_bracket = conf['grating_z_bracket']
        self.z_guess = sum(self.z_bracket) / 2
        super().__init__(**kwargs)
        self.elem_pos = [np.dot(conf['rotchan'][channel], e) for e in self.elem_pos]
        self.generate_elements()
        # Keywords that help position the gratings

    def elem_rg(self, gamma, beta):
        return self.conf['d']/(np.sqrt(2) * self.G * np.sin(gamma)**3 * (np.sin(beta) + np.cos(beta)))

    def xyz_from_gammabeta(self, gamma, beta):
        rg = self.elem_rg(gamma, beta)
        return rg * np.cos(gamma), rg * np.sin(gamma) * np.sin(beta), rg * np.sin(gamma) * np.cos(beta)

    @staticmethod
    def cart2sph(x, y, z):
        '''convert cartesian to spherical coordiantes'''
        hyz = np.hypot(y, z)
        beta = np.arctan2(y, z)
        gamma = np.arctan2(hyz, x)
        rg = np.hypot(hyz, x)
        return rg, gamma, beta

    def z_from_xy(self, x, y):
        def func(z):
            rg, gamma, beta = self.cart2sph(x, y, z)
            return rg - self.elem_rg(gamma, beta)
        sol = optimize.root_scalar(func, bracket=self.z_bracket, x0=self.z_guess)
        if not sol.converged:
            warn(f'Calculating z position for {x}{y} failed because: {sol.flag}')
        return sol.root

    def elempos(self):
        '''This elempos makes a regular grid, very similar to Mark Egan's design.'''
        dx = 2 * self.conf['gratingzoom'][1] + 2 * self.conf['gratingframe'][1]
        dy = 2 * self.conf['gratingzoom'][2] + 2 * self.conf['gratingframe'][2]
        x = np.arange(dx / 2, 100, dx)
        x = np.hstack([-x, x])
        y = np.arange(self.y_in[0], self.y_in[1], dy)
        mx, my = np.meshgrid(x, y)
        mx = mx.flatten()
        my = my.flatten()
        # Throw out those that are outside of a 30 deg sector
        ind = np.rad2deg(np.arctan2(np.abs(mx), np.abs(my))) < 30
        mx = mx[ind]
        my = my[ind]
        # Get z value
        z = np.array([self.z_from_xy(mx[i], my[i]) for i in range(len(mx))])
        rg, gamma, beta = self.cart2sph(mx, my, z)

        ang_in = np.arctan2(self.mirrorpos['r_in'], self.mirrorpos['f'])
        ang_out = np.arctan2(self.mirrorpos['r_out'], self.mirrorpos['f'])

        ind = (np.abs(beta) > ang_in) & (np.abs(beta) < ang_out)

        return np.vstack([mx[ind], my[ind], z[ind], np.ones(ind.sum())]).T

    def generate_elements(self):
        super().generate_elements()
        if self.bend:
            bend_gratings(self.elements, r_elem=self.bend)
