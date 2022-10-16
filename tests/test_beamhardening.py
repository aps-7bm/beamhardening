import pytest
import numpy as np

from beamhardening import beamhardening, material

@pytest.fixture
def beamsoftener():
    return beamhardening.BeamCorrector()

@pytest.fixture
def simple_spectrum():
    return beamhardening.Spectrum([1e4], [1.0])

@pytest.fixture
def top_hat_spectrum():
    energies = np.linspace(9000, 13000, 1001)
    powers = np.ones_like(energies)
    powers -= np.abs(energies - 11000.) * 10.
    powers[powers < 0] = 0
    return beamhardening.Spectrum(energies, powers)

@pytest.fixture
def dual_energy_spectrum():
    energies = np.linspace(9000, 33000, 4001)
    powers = np.ones_like(energies)
    powers_10keV = powers - np.abs(energies - 10000.) * 10.
    powers_10keV[powers_10keV < 0] = 0
    powers_30keV = powers - np.abs(energies - 30000.) * 10.
    powers_30keV[powers_30keV < 0] = 0
    return beamhardening.Spectrum(energies, powers_10keV + powers_30keV)
    
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
    assert(np.allclose(top_hat_spectrum.integrated_power(), 8./3.))

def test_spectrum_mean_energy(top_hat_spectrum):
    assert(np.allclose(top_hat_spectrum.mean_energy(), 11000.))

def test__find_calibration_one_angle(beamsoftener, top_hat_spectrum, dual_energy_spectrum):
    beamsoftener.add_scintillator('LuAG', 100)
    beamsoftener.add_sample('Fe')
    beamsoftener.add_filter('Cu', 50)
    #Test with the top hat spectrum
    top_hat_spline = beamsoftener._find_calibration_one_angle(top_hat_spectrum)
    Fe_matl = material.Material('Fe', 7.87)
    test_thickness = 32.4
    mono_ext_l = Fe_matl.return_ext_lengths_total(test_thickness, 11000)
    assert(np.allclose(top_hat_spline(mono_ext_l), test_thickness))
    #Now test with the dual energy spectrum
    dual_energy_spline = beamsoftener._find_calibration_one_angle(dual_energy_spectrum)
    scint_matl = material.Material('Lu3Al3O12', 6.73)
    Cu_matl = material.Material('Cu', 8.96)
    test_thicknesses = np.random.rand(10) * 1000
    for test_thickness in test_thicknesses:
        scint_10keV_ext_l = scint_matl.return_ext_lengths_total(100, 10000)
        scint_30keV_ext_l = scint_matl.return_ext_lengths_total(100, 30000)
        Cu_10keV_ext_l = Cu_matl.return_ext_lengths_total(50, 10000)
        Cu_30keV_ext_l = Cu_matl.return_ext_lengths_total(50, 30000)
        Fe_10keV_ext_l = Fe_matl.return_ext_lengths_total(test_thickness, 10000)
        Fe_30keV_ext_l = Fe_matl.return_ext_lengths_total(test_thickness, 30000)
        ext_to_sample_10keV = scint_10keV_ext_l + Cu_10keV_ext_l
        ext_to_sample_30keV = scint_30keV_ext_l + Cu_30keV_ext_l
        input_to_sample = 0.5 * (np.exp(-ext_to_sample_10keV) + np.exp(-ext_to_sample_30keV))
        total_ext_l_10keV = ext_to_sample_10keV + Fe_10keV_ext_l
        total_ext_l_30keV = ext_to_sample_30keV + Fe_30keV_ext_l
        trans = (0.5 * (np.exp(-total_ext_l_10keV) + np.exp(-total_ext_l_30keV))) / input_to_sample
        dual_ext_l = -np.log(trans)
        assert(np.allclose(dual_energy_spline(dual_ext_l), test_thickness))
