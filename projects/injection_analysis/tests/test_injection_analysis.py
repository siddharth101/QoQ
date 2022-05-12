import os
import shutil
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from gwpy.spectrogram import Spectrogram
from injection_analysis import main

TEST_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def out_dir():
    data_dir = "tmp"
    os.makedirs(data_dir, exist_ok=True)
    yield Path(data_dir)
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture(params=[["H1", "L1"], ["H1"]])
def ifos(request):
    return request.param


@pytest.fixture(params=[5, 8])
def m1_cut_low(request):
    return request.param


@pytest.fixture(params=[5, 8])
def m2_cut_low(request):
    return request.param


@pytest.fixture(params=[5, 8])
def snr_cut_low(request):
    return request.param


@pytest.fixture(params=[1e20])
def snr_cut_high(request):
    return request.param


@pytest.fixture(params=[1, 2])
def window(request):
    return request.param


@pytest.fixture(params=["injections.hdf5"])
def injection_file(request):
    return str(TEST_DIR / "data" / request.param)


def test_main_produces_expected_shape(injection_file, ifos, window, out_dir):

    # load in xml file of injections
    events = h5py.File(injection_file, "r")["events"][()]
    n_events = len(events)

    # q transform / pixel occ parameters
    sample_rate = 2048
    threshold = 50
    fmin = 10
    fmax = sample_rate / 2
    t_windows = [0.5, 1.0, 2.0]
    f_windows = [100, 512, 1024]
    fres = 0.05
    tres = 0.01

    # calculate expected q transform output shape
    n_freqs = int((fmax - fmin) / fres)
    n_times = int(window / tres)

    # create fake q data as return value for mock call
    fake_q_data = Spectrogram(
        np.zeros((n_times, n_freqs)) + 60, df=fres, dt=tres, f0=fmin
    )

    # create fake pixel occ as return value for mock call
    fake_pixel_occ = np.zeros(9)

    # create mock calls to query_q_data
    # and calc_pixel_occupancy so we don't have to
    # actually query data and calc pixel occs during tests.
    # these functions should have tests in their own library

    mock_query_q_data = patch(
        "injection_analysis.query_q_data", return_value=fake_q_data
    )
    mock_calc_pixel_occ = patch(
        "injection_analysis.calc_pixel_occupancy", return_value=fake_pixel_occ
    )
    mock_check_state_vector = patch(
        "injection_analysis.check_state_vector", return_value=True
    )  # don't skip any events for any ifo

    # no cuts
    with mock_query_q_data, mock_calc_pixel_occ, mock_check_state_vector:
        out_file = main(
            injection_file,
            ifos,
            63744000,  # offset
            0,  # snr cut low
            1e20,  # snr cut high
            0,  # m1 cut low
            0,  # m1 cut high
            100,  # dummy science mode bitmask
            window,
            "GDS-CALIB_STRAIN_INJ1_O3Replay",  # channel
            "O3ReplayMDC_llhoft",  # frame type
            sample_rate,
            "GDS-CALIB_STATE_VECTOR",  # state channel
            fmin,
            fres,
            tres,
            f_windows,
            t_windows,
            threshold,
            False,  # store raw
            True,  # store pixel occ
            out_dir,
            50,  # logging cadence
        )

    expected_shape = (n_events, len(t_windows) * len(f_windows))
    with h5py.File(out_file) as f:

        for ifo in ifos:
            assert np.shape(f[ifo]["pixel_occ"]) == expected_shape


def test_main_produces_events_with_correct_cuts(
    injection_file,
    ifos,
    window,
    snr_cut_low,
    snr_cut_high,
    m1_cut_low,
    m2_cut_low,
    out_dir,
):

    # q transform / pixel occ parameters
    sample_rate = 2048
    threshold = 50
    fmin = 10
    fmax = sample_rate / 2
    t_windows = [0.5, 1.0, 2.0]
    f_windows = [100, 512, 1024]
    fres = 0.05
    tres = 0.01

    # calculate expected q transfrom output shape
    n_freqs = int((fmax - fmin) / fres)
    n_times = int(window / tres)

    # create fake q data
    fake_q_data = Spectrogram(
        np.zeros((n_times, n_freqs)) + 60, df=fres, dt=tres, f0=fmin
    )
    fake_pixel_occ = np.zeros(9)

    # create mock call to query_q_data so we don't have to
    # actually query data during tests

    mock_query_q_data = patch(
        "injection_analysis.query_q_data", return_value=fake_q_data
    )
    mock_calc_pixel_occ = patch(
        "injection_analysis.calc_pixel_occupancy", return_value=fake_pixel_occ
    )
    mock_check_state_vector = patch(
        "injection_analysis.check_state_vector", return_value=True
    )  # don't skip any events for any ifo

    with mock_query_q_data, mock_calc_pixel_occ, mock_check_state_vector:
        out_file = main(
            injection_file,
            ifos,
            63744000,
            snr_cut_low,
            snr_cut_high,
            m1_cut_low,
            m2_cut_low,
            100,
            window,
            "GDS-CALIB_STRAIN_INJ1_O3Replay",  # channel
            "O3ReplayMDC_llhoft",  # frame type
            sample_rate,
            "GDS-CALIB_STATE_VECTOR",  # state channel
            fmin,
            fres,
            tres,
            f_windows,
            t_windows,
            threshold,
            False,  # store raw
            True,  # store pixel occ
            out_dir,
            50,  # logging cadence
        )

    with h5py.File(out_file) as f:
        for ifo in ifos:
            assert ifo in f.keys()
            assert all(f[ifo]["event parameters"]["mass1"] > m1_cut_low)
            assert all(f[ifo]["event parameters"]["mass2"] > m2_cut_low)
            assert all(
                f[ifo]["event parameters"][f'snr_{ifo.strip("1")}']
                > snr_cut_low
            )
