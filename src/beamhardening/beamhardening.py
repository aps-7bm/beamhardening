'''Code to correct for beam hardening effects in imaging experiments.
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
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from beamhardening import material


log = logging.getLogger(__name__)


data_path = Path(__file__).parent


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


class BeamCorrector():
    # Variables we need for computing LUT
    spectra_dict = None # Initialized in __init__
    scintillator_thickness = 0
    scintillator_material = None
    dark_image = None
    flat_image = None
    d_source = None
    sample_material = None
    pixel_size = None
    filters = None # Initialized in __init__
    angular_spline = None
    ref_trans = None
    threshold_trans = None
    # Variables for when we convert images
    centerline_spline = None
    
    def __init__(self):
        """Initializes the beam hardening correction code."""
        log.info('  *** beam hardening')
        self.filters = {}        
        self.filter_densities = {}
        self.possible_materials = {}
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
            config_path = Path.joinpath(Path(__file__).parent, 'setup.cfg')
        with open(config_path, 'r') as config_file:
            for line in config_file.readlines():
                if line.startswith('#'):
                    continue
                elif line.startswith('symbol'):
                    symbol = line.split(',')[0].split('=')[1].strip()
                    density = float(line.split(',')[1].split('=')[1])
                    self.possible_materials[symbol] = (symbol, density)
                elif line.startswith('name'):
                    name = line.split(',')[0].split('=')[1].strip()
                    symbol = line.split(',')[1].split('=')[1].strip()
                    density = float(line.split(',')[2].split('=')[1])
                    self.possible_materials[name] = (symbol, density)
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
                spectral_energies = spectral_data[:,0]
                spectral_power = spectral_data[:,1]
                self.spectra_dict[f_angle] = Spectrum(spectral_energies, spectral_power)

    
    def add_filter(self, symbol, thickness, density = None):
        """Add a filter of a given symbol and thickness."""
        if density is None and symbol in self.possible_materials:
            density = self.possible_materials[symbol][1]
            matl = material.Material(self.possible_materials[symbol][0], density)
        else:
            matl = material.Material(symbol, density)
        self.filters[matl] = thickness
    

    def add_sample(self, symbol, density = None):
        '''Define a sample material to be used in these calculations.
        Inputs:
        symbol: chemical formula for the sample
        density: density of the sample material in g/cc
        '''
        if density is None and symbol in self.possible_materials:
            density = self.possible_materials[symbol][1]
            matl = material.Material(self.possible_materials[symbol][0], density)
        else:
            matl = material.Material(symbol, density)
        self.sample_material = matl


    def add_scintillator(self, symbol, thickness, density = None):
        '''Define a scintillator material to be used in these calculations.
        Inputs:
        symbol: chemical formula for the sample
        thickness: active thickness of the scintillator
        density: density of the sample material in g/cc
        '''
        if density is None and symbol in self.possible_materials:
            density = self.possible_materials[symbol][1]
            matl = material.Material(self.possible_materials[symbol][0], density)
        else:
            matl = material.Material(symbol, density)
        self.scintillator_material = matl
        self.scintillator_thickness = thickness


    def set_geometry(self, d_source, pixel_size):
        '''Explicitly set the geometry for computing vertical variations in spectrum.
        
        Parameters
        ----------
        d_source : float
            distance from the source to the scintillator in meters
        pixel_size: float
            size of the pixels in object space in microns
        '''
        self.d_source = d_source
        self.pixel_size = pixel_size


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
        vertical_slice = np.sum(input_image, axis=1, dtype=np.float64)
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
        self.compute_interp_values()
        self.centerline_spline = InterpolatedUnivariateSpline(
                                                    self.centerline_interp_values[0],
                                                    self.centerline_interp_values[1],
                                                    )
        angular_spline = InterpolatedUnivariateSpline(
                                                    self.angular_interp_values[0],
                                                    self.angular_interp_valures[1],
                                                    )
        self.angular_correction = angular_spline(self.angles)[:,None]

    
    def _find_calibration_one_angle(self, input_spectrum):
        '''Makes a scipy interpolation function to be used to correct images.
        
        '''
        usable_ext_l, usable_thicknesses = self._find_interp_values_one_angle(input_spectrum)
        return InterpolatedUnivariateSpline(usable_ext_l, usable_thicknesses, ext='const')


    def _find_interp_values_one_angle(self, input_spectrum):
        '''Makes a scipy interpolation function to be used to correct images.
        
        '''
        # Make an array of sample thicknesses
        sample_thicknesses = np.sort(np.concatenate((-np.logspace(1,-1,41), [0], np.logspace(-1,4.5,111))))
        # For each thickness, compute the absorbed power in the scintillator
        detected_power = np.zeros_like(sample_thicknesses)
        sample_ext_lengths = self.sample_material.return_ext_lengths_total(1, input_spectrum.energies)
        scint_ext_lengths = self.scintillator_material.return_ext_lengths_abs(self.scintillator_thickness, input_spectrum.energies)
        scint_abs_spectrum = 1 - np.exp(-scint_ext_lengths)
        for i in range(sample_thicknesses.size):
            trans = np.exp(-sample_ext_lengths * sample_thicknesses[i])
            sample_filtered_spectral_power = input_spectrum.spectral_power * trans
            scint_abs_spectral_power = sample_filtered_spectral_power * scint_abs_spectrum
            detected_power[i] = scipy.integrate.simps(scint_abs_spectral_power, input_spectrum.energies)
        # Compute an effective transmission vs. thickness
        scint_spectral_power = input_spectrum.spectral_power * scint_abs_spectrum
        absorbed_power = scipy.integrate.simps(scint_spectral_power, input_spectrum.energies)
        sample_effective_trans = detected_power / absorbed_power
        # Threshold the transmission we accept to keep the spline from getting unstable
        usable_trans = sample_effective_trans[sample_effective_trans > self.threshold_trans]
        usable_thicknesses = sample_thicknesses[sample_effective_trans > self.threshold_trans]
        # Return a spline, but make sure things are sorted in ascending order
        usable_ext_l = -np.log(usable_trans)
        inds = np.argsort(usable_ext_l)
        return (usable_ext_l[inds], usable_thicknesses[inds])


    def compute_interp_values(self):
        '''Compute the calibrations to perform beam hardening.
        Calibrate pathlength vs. transmission at each angle value.
        Compute the correction required to handle angular spectral variations
        '''
        if self.scintillator_material is None:
            print('Need to set scintillator material')
            raise AttributeError
        if self.sample_material is None:
            print('Need to define a sample material')
            raise AttributeError
        angles_urad = []
        cal_curve = []
        for angle in sorted(self.spectra_dict.keys()):
            print(angle)
            angles_urad.append(float(angle))
            spectrum = self.spectra_dict[angle]
            #Filter the beam
            filtered_spectrum = self.apply_filters(spectrum)
            #Create an interpolation function based on this
            angle_interp_values = self._find_interp_values_one_angle(filtered_spectrum)
            if angle  == 0:
                self.centerline_interp_values = angle_interp_values
            cal_curve.append(np.interp(
                                        self.ref_trans,
                                        angle_interp_values[0],
                                        angle_interp_values[1],
                                        ))
        cal_curve /= cal_curve[0]
        self.angular_interp_values = (angles_urad, cal_curve)

    
    def correct_image(self, input_image):
        '''Perform beam hardening corrections on an input image.
        Inputs:
        input_image: image to be corrected
        Returns:
        numpy array the same shape as input_trans, but in pathlength
        '''
        trans = (input_image - self.dark_image) / (self.flat_image - self.dark_image)
        return self.centerline_spline(-np.log(trans)) * self.angular_correction

        
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
        pathlength = mproc.distribute_jobs(-np.log(input_trans), self.centerline_spline, args=(), axis=1)
        return pathlength


    def add_dark_image(self, dark_frame):
        """Add dark image to the object."""
        self.dark_image = dark_frame


    def add_flat_image(self, flat_frame, exp_ratio = 1):
        """Add flat image, including processing for vertical fan center.
        
        Parameters
        ----------
        flat_frame : Numpy array
            array containing the flatfield data
        exp_ratio : float
            ratio of data exposure to flat exposure (default 1)
        """
        self.flat_image = flat_frame.astype(np.float64) * exp_ratio
        self.find_angles(self.flat_image) 
