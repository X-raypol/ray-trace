import numpy as np
from scipy import optimize
import astropy.units as u
from marxs.optics import OpticalElement
from marxs.simulator import ParallelCalculated
from arcus.ralfgrating import CATGratingwithL1, globalorderselector
from transforms3d.axangles import axangle2mat


def blazemat(blazeang):
    return axangle2mat(np.array([0, 0, 1]), blazeang.to(u.rad).value)


class GratingGrid(ParallelCalculated, OpticalElement):
    id_col = 'facet'
    G = 8.8e-8   # 0.88 Ang / mm
    d_frame = 0.5
    elem_class = CATGratingwithL1
    order_selector = globalorderselector

    def beta_from_betamax(self, betamax):
        '''return beta of grating center given beta on the rightmost border.
        This assumes gratings are oriented such that the central ray is the normal.
        '''
        def bm(beta):
            rg = self.elem_rg(0., beta)
            return np.abs(beta) + np.arcsin(self.conf['gratingzoom'][1] / rg) - np.abs(betamax)
        return optimize.root(bm, betamax)['x'][0]

    def ymax_from_beta(self, beta):
        x, y, z = self.elem_center(0., beta)
        return y * np.tan(np.deg2rad(30.))

    def distribute_on_fixed_beta(self, beta):
        l = 2 * np.abs(self.ymax_from_beta(beta))
        n = np.round(l / (2 * self.conf['gratingzoom'][0] + self.d_frame))
        # y = (np.arange(n) - n/2) * (2 * self.zoom_grating[0] + self.d_frame)
        # unfinished because I realized it's not the best way to do it
        # but I keep it here as reference for later

        rg = self.elem_rg(0, beta)
        ''' Technically, rg depends on gamma and d_gamma depends on rg and beta
        so this is an iterative problem, but this approximation is good enough for now.
        '''
        d_gamma = (2 * self.conf['gratingzoom'][0] + self.d_frame) / rg
        return (np.arange(n) - n / 2) * d_gamma

    def distribute_betas(self):
        beta = []
        betalow = self.conf['beta_lim'][1]
        while betalow > self.conf['beta_lim'][0]:
            beta.append(self.beta_from_betamax(betalow))
            betalow = beta[-1] - np.arcsin((self.conf['gratingzoom'][1] + .5) / self.elem_rg(0., beta[-1]))

        betalow = -self.conf['beta_lim'][1]
        while betalow < -self.conf['beta_lim'][0]:
            beta.append(self.beta_from_betamax(betalow))
            betalow = beta[-1] + np.arcsin((self.conf['gratingzoom'][1] + .5) / self.elem_rg(0., beta[-1]))
        return beta

    def __init__(self, channel, conf, **kwargs):
        kwargs['pos_spec'] = self.elempos
        if 'normal_spec' not in kwargs.keys():
            kwargs['normal_spec'] = np.array([0., 0., 0., 1.])
        if 'parallel_spec' not in kwargs.keys():
            kwargs['parallel_spec'] = np.array([0., 1., 0., 0.])
        self.conf = conf
        self.channel = channel
        kwargs['elem_class'] = self.elem_class
        kwargs['elem_args'] = {'d': conf['d'],
                               'order_selector': self.order_selector,
                               'zoom': conf['gratingzoom'],
                               'orientation': np.dot(axangle2mat([1, 0, 0], np.pi / 2),
                                                     blazemat(conf['blazeang'])),
                               }
        super(GratingGrid, self).__init__(**kwargs)
        self.elem_pos = [np.dot(conf['rotchan'][channel], e) for e in self.elem_pos]
        self.generate_elements()

    def elem_rg(self, gamma, beta):
        return self.conf['d']/(np.sqrt(2) * self.G * np.cos(gamma)**3 * (np.sin(beta) + np.cos(beta)))

    def elem_center(self, gamma, beta):
        rg = self.elem_rg(gamma, beta)
        return rg * np.sin(gamma), rg * np.cos(gamma) * np.sin(beta), rg * np.cos(gamma) * np.cos(beta)

    def elempos(self):
        betagamma = []
        betas = self.distribute_betas()
        for b in betas:
            for g in self.distribute_on_fixed_beta(b):
                betagamma.append([b, g])

        betagamma = np.array(betagamma)
        x, y, z = self.elem_center(betagamma[:, 1], betagamma[:, 0])

        return np.vstack([x, y, z, np.ones_like(x)]).T
