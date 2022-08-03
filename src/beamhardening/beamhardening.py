'''Code to correct for beam hardening effects in tomography experiments.
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

'''
from copy import deepcopy
from pathlib import Path, PurePath
import logging

import numpy as np
import scipy.integrate
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import convolve
from scipy.signal.windows import gaussian

from beamhardening import material


log = logging.getLogger(__name__)


data_path = Path(__file__).parent.parent / 'data'


class Spectrum:
    '''Class to hold the spectrum: energies and spectral power.'''
    def __init__(self, energies, spectral_power):
        if len(energies) != len(spectral_power):
            raise ValueError
        self.energies = energies
        self.spectral_power = spectral_power


    def integrated_power(self):
        return scipy.integrate.simps(self.spectral_power, self.energies)


    def mean_energy(self):
        total_power = self.integrated_power()
        return scipy.integrate.simps(self.spectral_power * self.energies, self.energies) / total_power
    

    def __len__(self):
        return len(energies)


class BeamSoftener():
    # Variables we need for computing LUT
    spectra_dict = None # Initialized in __init__
    scintillator_thickness = 0
    scintillator_material = None
    d_source = None
    sample_material = None
    pixel_size = None
    filters = None # Initialized in __init__
    angular_spline = None
    ref_trans = None
    threshold_trans = None
    # Variables for when we convert images
    centerline_spline = None
    angular_spline = None

    
    def __init__(self):
        """Initializes the beam hardening correction code."""
        log.info('  *** beam hardening')
        self.filters = {}        
        self.filter_densities = {}
        self.read_config_file()
        self.read_source_data()
        self.angles = None 


    def read_config_file(self, config_filename=None):
        '''Read in parameters for beam hardening corrections from file.
        Default file is in same directory as this source code.
        Users can input an alternative config file as needed.
        '''
        if config_filename:
            config_path = Path(config_filename)
            if not config_path.exists():
                raise IOError('Config file does not exist: ' + str(config_path))
        else:
            config_path = Path.joinpath(Path(__file__).cwd(), 'setup.cfg')
        with open(config_path, 'r') as config_file:
            for line in config_file.readlines():
                if line == '':
                    break
                if line.startswith('#'):
                    continue
                elif line.startswith('ref_trans'):
                    self.ref_trans = float(line.split(':')[1].strip())
                elif line.startswith('threshold_trans'):
                    self.threshold_trans = float(line.split(':')[1].strip())
                elif line.startswith('distance'):
                    self.d_source = float(line.split(':')[1].strip())
                elif line.startswith('pixel_size'):
                    self.pixel_size = float(line.split(':')[1].strip())
    

    def read_source_data(self):
        """Reads the spectral power data from files.  Data file comes from the
        BM spectrum module in XOP. Saves *self.spectra_dict*: a
        dictionary of spectra at the various psi angles from the ring
        plane.
        
        """
        self.spectra_dict = {}
        file_list = [x for x in data_path.iterdir() if x.suffix in ('.dat', '.DAT')]
        for f_path in file_list:
            if f_path.is_file() and f_path.name.startswith('Psi'):
                log.info('  *** *** source file {:s} located'.format(f_path.name))
                f_angle = float(f_path.name.split('_')[1][:2])
                spectral_data = np.genfromtxt(f_path, comments='!')
                spectral_energies = spectral_data[:,0] / 1000.
                spectral_power = spectral_data[:,1]
                self.spectra_dict[f_angle] = Spectrum(spectral_energies, spectral_power)
    

    def add_filter(self, symbol, density, thickness):
        """Add a filter of a given symbol and thickness."""
        matl = material.Material(symbol, density, spectra_dict[0].energies)
        self.filters[matl] = thickness
    

    def add_sample(self, symbol, density):
        '''Define a sample material to be used in these calculations.
        Inputs:
        symbol: chemical formula for the sample
        density: density of the sample material in g/cc
        '''
        self.sample_material = material.Material(symbol, density, spectra_dict[0].energies)


    def add_scintillator(self, symbol, density, thickness):
        '''Define a scintillator material to be used in these calculations.
        Inputs:
        symbol: chemical formula for the sample
        density: density of the sample material in g/cc
        thickness: active thickness of the scintillator
        '''
        self.scintillator_material = material.Material(symbol, density, spectra_dict[0].energies)
        self.scintillator_thickness = thickness


    def apply_filters(self, input_spectrum):
        """Computes the spectrum after all filters.
        
        Parameters
        ==========
        input_spectrum : np.ndarray
          spectral power for input spectrum, in numpy array
        
        Returns
        =======
        temp_spectrum : np.ndarray
          spectral power transmitted through the filter set.
        
        """
        temp_spectrum = deepcopy(input_spectrum)
        for filt, thickness in self.filters.items():
            temp_spectrum = filt.compute_transmitted_spectrum(thickness, temp_spectrum)
        return temp_spectrum


    def find_angles(self, input_image):
        '''Finds the brightest row of input_image.
        Filters to make sure we ignore spurious noise.
        Return:
        numpy array with angle of each image row
        '''
        vertical_slice = np.sum(input, axis=1, dtype=np.float64)
        gaussian_filter = scipy.signal.windows.gaussian(200,20)
        filtered_slice = scipy.signal.convolve(vertical_slice, gaussian_filter,
                                                mode='same')
        center_row = float(np.argmax(filtered_slice))
        self.angles = np.abs(np.arange(input_image.shape[0]) - center_row)
        self.angles *= self.pixel_size / self.d_source
    

    def compute_calibration(self):
        '''Compute the calibrations to perform beam hardening.
        Calibrate pathlength vs. transmission at each angle value.
        Compute the correction required to handle angular spectral variations
        '''
        if scintillator_material is None:
            print('Need to set scintillator material')
            raise AttributeError
        if sample_material is None:
            print('Need to define a sample material')
            raise AttributeError
        angles_urad = []
        cal_curve = []
        for angle in sorted(self.spectra_dict.keys()):
            angles_urad.append(float(angle))
            spectrum = self.spectra_dict[angle]
            #Filter the beam
            filtered_spectrum = apply_filters(self.filters, spectrum)
            #Create an interpolation function based on this
            angle_spline = self._find_calibration_one_angle(filtered_spectrum)
            if angle  == 0:
                self.centerline_spline = angle_spline
            cal_curve.append(angle_spline(self.ref_trans))
        cal_curve /= cal_curve[0]
        angular_spline = InterpolatedUnivariateSpline(angles_urad, cal_curve) 
        self.angular_correction = angular_spline(self.angles)[:,None]

    
    def _find_calibration_one_angle(self, input_spectrum):
        '''Makes a scipy interpolation function to be used to correct images.
        
        '''
        # Make an array of sample thicknesses
        sample_thicknesses = np.sort(np.concatenate((-np.logspace(1,0,21), [0], np.logspace(-1,4.5,441))))
        # For each thickness, compute the absorbed power in the scintillator
        detected_power = np.zeros_like(sample_thicknesses)
        for i in range(sample_thicknesses.size):
            sample_filtered_power = self.sample_material.compute_transmitted_spectrum(sample_thicknesses[i],
                                                                                  input_spectrum)
            detected_power[i] = self.scintillator_material.compute_absorbed_power(self.scintillator_thickness,
                                                                                   sample_filtered_power)
        # Compute an effective transmission vs. thickness
        absorbed_power = self.scintillator_material.compute_absorbed_power(self.scintillator_thickness,
                                                                            input_spectrum)
        sample_effective_trans = detected_power / absorbed_power
        # Threshold the transmission we accept to keep the spline from getting unstable
        usable_trans = sample_effective_trans[sample_effective_trans > self.threshold_trans]
        usable_thicknesses = sample_thicknesses[sample_effective_trans > self.threshold_trans]
        # Return a spline, but make sure things are sorted in ascending order
        inds = np.argsort(usable_trans)
        return InterpolatedUnivariateSpline(usable_trans[inds], usable_thicknesses[inds], ext='const')


    def correct_image(self, input_trans):
        '''Perform beam hardening corrections on an input image.
        Inputs:
        input_trans: transmission image
        Returns:
        numpy array the same shape as input_trans, but in pathlength
        '''
        return self.centerline_spline(input_trans) * self.angular_correction

        
    def correct_as_pathlength_centerline(self, input_trans):
        """Corrects for the beam hardening, assuming we are in the ring plane.

        Parameters
        ==========
        input_trans : np.ndarray
          transmission

        Returns
        =======
        pathlength : np.ndarray
          sample pathlength in microns.

        """
        return
        pathlength = mproc.distribute_jobs(input_trans, self.centerline_spline, args=(), axis=1)
        return pathlength


    def correct_vertical_spectrum(self, centerline_image):
        '''Applies the correction for the vertical position to an 
        image that has been corrected assuming the centerline spectrum.
        Returns:
        image corrected for vertical variations in spectrum
        '''
        return centerline_image * self.angular_correction
