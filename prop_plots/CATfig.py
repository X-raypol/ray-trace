import os
import numpy as np
import matplotlib.pyplot as plt
from settings import figureout, kwargsfig

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111, aspect='equal')

l = 20
d = 10
alpha = np.deg2rad(100)
step = np.arange(20, 120, d)
x = 100 + np.sin(alpha) * step
y = 340 + np.cos(alpha) * step
x1 = x + l * np.sin(alpha + np.pi / 2)
y1 = y + l * np.cos(alpha + np.pi / 2)

linerange = np.array([-100, 200])
gamma = np.deg2rad(115)
ax.plot(100 + np.sin(gamma) * linerange,
        345 + np.cos(gamma) * linerange, 'r', lw=2,
        label='plane perpendicular\nto incoming ray')

# Light gray bars for traditional transmission grating
xt = 100 + np.sin(gamma) * step
yt = 355 + np.cos(gamma) * step
x1t = xt + l * np.sin(gamma + np.pi / 2)
y1t = yt + l * np.cos(gamma + np.pi / 2)
ax.plot(np.vstack([xt, x1t]), np.vstack([yt, y1t]),
        '0.7', lw=5)  # , label='normal grating')


ax.plot(np.vstack([x, x1]), np.vstack([y, y1]),
        'k', lw=5)  # , label='CAT grating')
ax.plot([200, 160], [400, 320], 'b', lw=2, label='incoming ray')
ax.plot([160, 0], [320, 0], ls=':', color='b', lw=2, label='zero order ray')
ax.plot([160, 180], [320, 0], ls='--', color='b', lw=2, label='diffracted ray')
ax.plot([160, 160 + np.sin(alpha - np.pi / 2) * 80],
        [320, 320 + np.cos(alpha - np.pi / 2) * 80],
        'k', label='grating normal')
phi = np.arange(0.18, .45, .001)
ax.plot(160 + np.sin(phi) * 60,
        320 + np.cos(phi) * 60, 'k')
ax.text(168, 360, r'$\Theta_B$', size=20)
ax.set_ylim(220, 400)
ax.set_xlim(100, 400)
ax.set_ylabel('Optical axis')
ax.legend(loc='right')
ax.set_axis_off()
out = ax.set_xlabel('Dispersion direction')
#fig.savefig('/melkor/d1/guenther/projects/REDSoX/JATIS/CAT_principle.pdf')
fig.savefig('/melkor/d1/guenther/projects/REDSoX/JATIS/CAT_principle.png', dpi=300)
