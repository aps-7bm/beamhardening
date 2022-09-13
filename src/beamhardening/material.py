'''Class to define a material and its x-ray interactions.
'''
from copy import deepcopy
import numpy as np
import xraydb

class Material:
    '''Class that defines the absorption and attenuation properties of a material.
    Data taken from xraydb
    
    '''
    def __init__(self, name, density):
        self.name = name
        self.density = density
 

    def __repr__(self):
        return f"Material({self.name}, density {self.density} g/cm^3)"

    
    def compute_proj_density(self, thickness):
        '''Computes projected density from thickness and material density.
        Input: thickness in um
        Output: projected density in g/cm^2
        '''
        self.proj_density = thickness /1e4 * self.density
    

    def return_ext_lengths_total(self, thickness, energies):
        attenuation_array = xraydb.material_mu(
                                            self.name,
                                            energies,
                                            self.density,
                                            'total'
                                            )
        return attenuation_array * thickness * 1e-4
        

    def return_ext_lengths_abs(self, thickness, energies):
        attenuation_array = xraydb.material_mu(
                                            self.name,
                                            energies,
                                            self.density,
                                            'photo'
                                            )
        return attenuation_array * thickness * 1e-4


    def compute_transmitted_spectrum(self, thickness, input_spectrum):
        '''Computes the transmitted spectral power through a filter.
        Inputs:
        thickness: the thickness of the filter in um
        input_spectrum: Spectrum object for incident spectrum
        Output:
        Spectrum object for transmitted intensity
        '''
        output_spectrum = deepcopy(input_spectrum)
        #Find the spectral transmission using Beer-Lambert law
        attenuation_array = xraydb.material_mu(
                                            self.name,
                                            input_spectrum.energies,
                                            self.density,
                                            'total'
                                            )
        attenuation_array *= thickness * 1e-4
        output_spectrum.spectral_power *= np.exp(-attenuation_array)
        return output_spectrum
    
    def compute_absorbed_spectrum(self, thickness, input_spectrum):
        '''Computes the absorbed power of a filter.
        Inputs:
        thickness: the thickness of the filter in um
        input_spectrum: Spectrum object for incident beam
        Output:
        Spectrum objection for absorbed spectrum
        '''
        output_spectrum = deepcopy(input_spectrum)
        #Find the spectral transmission using Beer-Lambert law
        absorption_array = xraydb.material_mu(
                                            self.name,
                                            input_spectrum.energies,
                                            self.density,
                                            'photo'
                                            )
        absorption_array *= thickness * 1e-4
        output_spectrum.spectral_power *= (1.0 - np.exp(-absorption_array))
        return output_spectrum
    

    def compute_absorbed_power(self, thickness, input_spectrum):
        '''Computes the absorbed power of a filter.
        Inputs:
        material: the Material object for the filter
        thickness: the thickness of the filter in um
        input_energies: Spectrum object for incident beam 
        Output:
        absorbed power
        '''
        return self.compute_absorbed_spectrum(thickness,input_spectrum).integrated_power()
