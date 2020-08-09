import os
import numpy as np

#redsoxbase = '/melkor/d1/guenther/Dropbox/REDSoX File Transfers'
inputpath = os.path.join(os.path.dirname(__file__), '..', 'inputdata')
xyz2zxy = np.array([[0., 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).T
