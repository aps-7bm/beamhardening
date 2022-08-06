import pytest
import numpy as np

from beamhardening import beamhardening, material

@pytest.fixture
def beamsoftener():
    return beamhardening.BeamSoftener()

@pytest.fixture
def simple_spectrum():
    return beamhardening.Spectrum([1e4], [1.0])

@pytest.fixture
def top_hat_spectrum():
    energies = np.linspace(9000, 13000, 1001)
    powers = np.ones_like(energies)
    powers -= np.abs(energies - 11000.) / 1000.
    powers[powers < 0] = 0
    return beamhardening.Spectrum(energies, powers)

def test_set_geometry(beamsoftener):
    beamsoftener.set_geometry(25, 5.1)
    assert(np.allclose(beamsoftener.d_source, 25))
    assert(np.allclose(beamsoftener.pixel_size, 5.1))

def test_add_filter(beamsoftener):
    beamsoftener.add_filter('Fe', 7.874, 204)
    beamsoftener.add_filter('Be', 1.8, 750)
    assert(len(list(beamsoftener.filters.keys())) == 2)

def test_apply_filters(beamsoftener, simple_spectrum):
    beamsoftener.add_filter('Fe', 7.874, 204)
    beamsoftener.add_filter('Be', 1.8, 750)
    filtered = beamsoftener.apply_filters(simple_spectrum)
    assert(np.allclose(filtered.spectral_power[0], 1.1339292e-12))
        
def test_find_angles(beamsoftener):
    test_image = np.zeros((500, 1))
    test_image[232,0] = 100.
    beamsoftener.set_geometry(25, 5.1)
    beamsoftener.find_angles(test_image)
    pixels_from_center = np.abs(np.arange(500) - 232)
    ref_angle = pixels_from_center * 5.1 / 25
    assert(np.allclose(ref_angle, beamsoftener.angles))

def test_spectrum_integrated_power(top_hat_spectrum):
    assert(np.allclose(top_hat_spectrum.integrated_power(), 1000.))

def test_spectrum_mean_energy(top_hat_spectrum):
    assert(np.allclose(top_hat_spectrum.mean_energy(), 11000.))
