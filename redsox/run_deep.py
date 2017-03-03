import run_tools
from astropy.table import Table

n_photons = 1e5
outpath = '/melkor/d1/guenther/projects/REDSoX/sims/aeff/'

aeff = []
modulation = []

aeff.append(run_tools.run_aeff(n_photons=n_photons))
modulation.append(run_tools.run_modulation(n_photons=n_photons))

outfixed = Table([aeff, modulation],
                 names=['Aeff', 'modulation'])
outfixed.write(outpath + 'deep.fits', overwrite=True)
