import pytest
import numpy as np
from beamhardening import material, beamhardening

def test_material_init(sample_mat):
    assert(sample_mat.name == 'Fe')
    assert(sample_mat.density == 7.874)

def test_transmitted_spectrum(sample_mat, simple_spectrum):
    assert(np.allclose(sample_mat.compute_transmitted_spectrum(12.5, simple_spectrum).spectral_power[0],
                0.186369))           
    
def test_absorbed_spectrum(sample_mat, simple_spectrum):
    assert(np.allclose(sample_mat.compute_absorbed_spectrum(12.5, simple_spectrum).spectral_power[0],
                0.811256))           

def test_absorbed_power(sample_mat, uniform_spectrum):
    assert(np.allclose(
    
@pytest.fixture
def sample_mat():
    return material.Material('Fe', 7.874)

@pytest.fixture
def simple_spectrum():
    return beamhardening.Spectrum([1e4], [1.0])

@pytest.fixture
def peak_spectrum():
    return beamhardening.Spectrum([9e3, 1e4, 1.1e4], [0., 1.0, 0.])
'''
    def compute_transmitted_spectrum(self,  input_spectrum):
    def compute_absorbed_spectrum(self, thickness, input_spectrum):
    def compute_absorbed_power(self, thickness, input_spectrum):
'''
