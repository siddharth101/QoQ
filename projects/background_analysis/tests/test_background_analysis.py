import os
import shutil
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from background_analysis import process_one_pycbc_file
from gwpy.spectrogram import Spectrogram

TEST_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def out_dir():
    data_dir = "tmp"
    os.makedirs(data_dir, exist_ok=True)
    yield Path(data_dir)
    shutil.rmtree(data_dir)


@pytest.fixture(params=[["H1", "L1"], ["H1"]])
def ifos(request):
    return request.param


@pytest.fixture(params=[5, 8])
def m1_cut_low(request):
    return request.param


@pytest.fixture(params=[5, 8])
def m2_cut_low(request):
    return request.param


@pytest.fixture(params=[50, 100])
def ifar_thresh(request):
    return request.param


@pytest.fixture(params=[1, 2])
def window(request):
    return request.param


@pytest.fixture(params=["triggers.h5"])
def trigger_file(request):
    return str(TEST_DIR / "data" / request.param)


@pytest.fixture(params=["templates.h5"])
def template_file(request):
    return str(TEST_DIR / "data" / request.param)


def test_process_one_pycbc_file_produces_correct_event_shape(
    trigger_file, template_file, ifos, window, out_dir
):
    # get number of events
    with h5py.File(trigger_file) as f:
        n_events = len(f["background_exc"]["ifar"][()])

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

    # create mock calls to query_q_data and
    # calc_pixel_occupancy so we don't have to
    # actually query data and calc pixel occs during tests.
    # these functions should have tests in their own library

    mock_query_q_data = patch(
        "background_analysis.query_q_data", return_value=fake_q_data
    )
    mock_calc_pixel_occ = patch(
        "background_analysis.calc_pixel_occupancy", return_value=fake_pixel_occ
    )

    with mock_query_q_data, mock_calc_pixel_occ:
        (
            pixel_occupancies,
            q_data,
            times,
            m1s,
            m2s,
            ifars,
        ) = process_one_pycbc_file(
            ifos,
            0,  # no cuts on m1, m2 or ifar so all events make it
            0,
            0,
            window,
            "DCS-CALIB_STRAIN_CLEAN_C01",
            "HOFT_C01",
            sample_rate,
            fmin,
            fres,
            tres,
            f_windows,
            t_windows,
            threshold,
            100,
            trigger_file,
            template_file,
        )

    expected_shape = (n_events, len(t_windows) * len(f_windows))

    for ifo in ifos:
        assert np.shape(pixel_occupancies[ifo]) == expected_shape


def test_process_one_pycbc_file_produces_events_with_correct_cuts(
    trigger_file,
    template_file,
    ifos,
    m1_cut_low,
    m2_cut_low,
    ifar_thresh,
    window,
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

    # create mock call to query_q_data so we don't have to
    # actually query data during tests

    mock_query_q_data = patch(
        "background_analysis.query_q_data", return_value=fake_q_data
    )

    with mock_query_q_data:
        (
            pixel_occupancies,
            q_data,
            times,
            m1s,
            m2s,
            ifars,
        ) = process_one_pycbc_file(
            ifos,
            m1_cut_low,
            m2_cut_low,
            ifar_thresh,
            window,
            "DCS-CALIB_STRAIN_CLEAN_C01",
            "HOFT_C01",
            sample_rate,
            fmin,
            fres,
            tres,
            f_windows,
            t_windows,
            threshold,
            100,
            trigger_file,
            template_file,
        )

    for ifo in ifos:
        assert ifo in pixel_occupancies.keys()
        assert ifo in q_data.keys()
        assert ifo in times.keys()

    assert all(m1s > m1_cut_low)
    assert all(m2s > m2_cut_low)
    assert all(ifars > ifar_thresh)

    assert len(m1s) == len(m2s)
    assert len(m1s) == len(ifars)
    assert len(m2s) == len(ifars)
