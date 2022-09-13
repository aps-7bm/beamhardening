from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import beamhardening.beamhardening as bh

corrector = bh.BeamSoftener()

with h5py.File('/data/ARL/ARL_31030_Reprocess/ARL_Diesel_31030_007.h5','r') as hdf_file:
    dark = hdf_file['/exchange/data_dark'][...]
    flat = hdf_file['/exchange/data_white'][...]
    data = hdf_file['/exchange/data'][0,...]

dark = np.median(dark, axis = 0)
flat = np.median(flat, axis = 0)
print(dark.shape)
print(flat.shape)
print(data.shape)

trans = (data - dark) / (flat - dark)
trans[trans < 0] = 1e-6
trans[trans > 1.1] = 1.1

corrector.set_geometry(36, 20. / 2.1)
corrector.add_filter('Be', 1.85, 750)
corrector.add_sample('Fe', 7.8)
corrector.add_scintillator('Lu3Al5O12', 6.73, 100)
corrector.find_angles(flat)
corrector.compute_calibration()
bob = corrector.correct_image(trans)
plt.plot(trans)
