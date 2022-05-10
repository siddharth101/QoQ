from unittest.mock import patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries
from qoq.core import query_q_data


@pytest.fixture(params=["H1", "L1"])
def ifo(request):
    return request.param


@pytest.fixture(params=[125600000])
def time(request):
    return request.param


@pytest.fixture(params=[2])
def window(request):
    return request.param


@pytest.fixture(params=[2048])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[32])
def fmin(request):
    return request.param


@pytest.fixture(params=[0.01])
def tres(request):
    return request.param


@pytest.fixture(params=[0.05])
def fres(request):
    return request.param


def test_query_q_data_produces_expected_shape(
    ifo,
    time,
    window,
    sample_rate,
    fmin,
    tres,
    fres,
):

    # doesnt matter as datafind is mocked
    channel = None
    frame_type = None

    # create dummy sine wave
    ts_length = 12  # arbitrary, will be cropped
    f = 800  # frequency in Hz

    t = np.linspace(-ts_length / 2, ts_length / 2, ts_length * sample_rate)
    waveform = np.sin(2 * np.pi * t * f)

    sine_wave = TimeSeries(
        waveform, t0=time - ts_length / 2, sample_rate=sample_rate
    )

    # define mocks
    mock_ts = patch("gwpy.timeseries.TimeSeries.read", return_value=sine_wave)
    mock_find_urls = patch("qoq.core.core.find_urls", return_value=None)

    # calc fmax from sample rate
    fmax = sample_rate / 2

    # determine expected shape
    expected_shape = (int(2 * window / tres), int((fmax - fmin) / fres))

    with mock_ts, mock_find_urls:
        q_data = query_q_data(
            ifo,
            time,
            window,
            channel,
            frame_type,
            sample_rate,
            fmin,
            tres,
            fres,
        )

        assert np.shape(q_data) == expected_shape


"""
def test_calc_pixel_occupancy(
    fmin,
    fres,
    window,
    threshold,


):
    shape =
    q_data = np.zeros(())
    calc_pixel_occupancy(
        q_data,
        fmin,
        fres,
        window,
        threshold,
        f_windows,
        t_windows
    )
"""
