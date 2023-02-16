# beamhardening

This is a package to correct for beam hardening in synchrotron WB imaging from bending magnet sources.

This code depends on [xraydb](github.com/xraypy/XrayDB).  The package contains two modules, beamhardening.beamhardening, which has the calculations described below, and beamhardening.material, which holds the material properties and calls xraydb.

Code to correct for beam hardening effects in imaging experiments.
The main application of this code is for synchrotron experiments with 
a bending magnet beam.  This beam is both polychromatic and has a spectrum
which varies with the vertical angle from the ring plane.  In principle,
this could be used for other polychromatic x-ray sources.

The mathematical approach is to filter the incident spectrum by a 
series of filters.  This filtered spectrum passes through a series of
thicknesses of the sample material.  For each thickness, the transmitted
spectrum illuminates the scintillator material.  The absorbed power in 
the scintillator material is computed as a function of the
sample thickness.  A univariate spline fit is then calculated
between the calculated transmission and the sample thickness for the centerline
of the BM fan.  This is then used as a lookup table to convert sample 
transmission to sample thickness, as an alternative to Beer's law.
To correct for the dependence of the spectrum on vertical angle,
at a reference transmission (0.1 by default, which works well with the APS BM
beam), the ratio between sample thickness computed with the centerline spline
fit and the actual sample thickness is computed as a correction factor. 
A second spline fit between vertical angle and correction factor is calculated,
and this is used to correct the images for the dependence of the spectrum
on the vertical angle in the fan.  

This code uses a set of text data files to define the spectral
properties of the beam.  The spectra are in text files with 
two columns.  The first column gives energy in eV, the second the spectral
power of the beam.  A series of files are used, in the form 
Psi_##urad.dat, with the ## corresponding to the vertical angle from the ring
plane in microradians.  These files were created in the BM spectrum
tool of XOP.

This code also uses a setup.cfg file, located in beam_hardening_data.
This mainly gives the options for materials, their densities, and 
the reference transmission for the angular correction factor.

## Typical Usage

'''
from beamhardening import beamhardening
beamsoftener = beamhardening.BeamCorrector()
#Add scintillator material
beamsoftener.add_scintillator('LuAG', 100) #LuAG scintillator, 100 microns thick

#Add sample material
beamsoftener.add_sample('Cu', 8.9)  #Cu sample, density = 8.9 kg/m^3
#Set geometry for vertical spectrum variations
beamsoftener.set_geometry(25, 5.1)  #25 m distance, 5.1 micron pixel size

#Add filters
beamsoftener.add_filter('Fe', 7.874, 204) #Fe filter, density 7.874 kg/m^3, 204 microns thick
beamsoftener.add_filter('Be', 1.8, 750)

#Compute the spline fits for pathlength and angle
beamsoftener.compute_calibration()

#Add flat and dark images
beamsoftener.add_flat_image(flat_image)
beamsoftener.add_dark_image(dark_image)

#Perform correction
beamsoftener.correct_image(image)
'''
