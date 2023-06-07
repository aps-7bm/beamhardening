import pytest
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

from beamhardening import beamhardening, material

@pytest.fixture
def beamsoftener_lookup():
    return beamhardening.BeamCorrector()

@pytest.fixture
def centerline_read_spectrum():
    data_path = Path.cwd().joinpath('data')
    text_data = np.genfromtxt(data_path.joinpath('Psi_00urad.dat'))
    return beamhardening.Spectrum(text_data[:,0], text_data[:,1])


@pytest.fixture
def urad40_read_spectrum():
    data_path = Path.cwd().joinpath('data')
    text_data = np.genfromtxt(data_path.joinpath('Psi_40urad.dat'))
    return beamhardening.Spectrum(text_data[:,0], text_data[:,1])

@pytest.fixture
def beamsoftener_calc():
    return beamhardening.BeamCorrector(
                                    calculate_source = "standard",
                                    E_storage_ring = 7,
                                    B_storage_ring = 0.6,
                                    minimum_E = 1000,
                                    maximum_E = 2e5,
                                    step_E = 100,
                                    minimum_psi_urad = 0,
                                    maximum_psi_urad = 40,
                                    )

def test_calc_is_done(beamsoftener_calc, centerline_read_spectrum):
    ''' Test to make sure calculations are actually done.
    '''
    spec = beamsoftener_calc.spectra_dict[0]
    assert not np.allclose(spec.spectral_power, centerline_read_spectrum.spectral_power)
    

def test_calc_propotional_to_centerline(beamsoftener_calc, centerline_read_spectrum):
    ''' See if the calculations match data from XOP
    '''
    spec = beamsoftener_calc.spectra_dict[0]
    calc_spectrum_interp = scipy.interpolate.interp1d(
                                                    spec.energies,
                                                    spec.spectral_power,
                                                    )
    calc_match_file = calc_spectrum_interp(centerline_read_spectrum.energies)
    ratio = calc_match_file / centerline_read_spectrum.spectral_power
    ratio /= np.mean(ratio)
    print(f'Range of ratio between calc and file is {np.ptp(ratio)}')
    assert(np.allclose(ratio / np.mean(ratio), np.ones_like(ratio)))


def test_calc_propotional_to_40urad(beamsoftener_calc, urad40_read_spectrum):
    ''' See if the calculations match data from XOP
    '''
    spec = beamsoftener_calc.spectra_dict[40]
    calc_spectrum_interp = scipy.interpolate.interp1d(
                                                    spec.energies,
                                                    spec.spectral_power,
                                                    )
    calc_match_file = calc_spectrum_interp(urad40_read_spectrum.energies)
    ratio = calc_match_file / urad40_read_spectrum.spectral_power
    ratio /= np.mean(ratio)
    print(f'Range of ratio between calc and file is {np.ptp(ratio)}')
    assert(np.allclose(ratio / np.mean(ratio), np.ones_like(ratio)))
