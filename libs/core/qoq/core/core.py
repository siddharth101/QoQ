from typing import List

import numpy as np
import pandas as pd
from gwdatafind import find_urls
from gwpy.segments import Segment
from gwpy.timeseries import TimeSeries


def query_q_data(
    ifo: str,
    time: float,
    window: float,
    channel: str,
    frame_type: str,
    sample_rate: float,
    fmin: float,
    tres: float,
    fres: float,
):
    """Queries strain data around trigger time and performs a q transform

    Args:
        ifo: Which interferometer to query data for
        time: trigger time
        window: half window around trigger time to query
        channel: name of strain channel to query
        sample_rate: sampling rate
        fmin: minimum frequency
        tres: time resolution of q transform
        fres: frequency resolution of q transform

    Returns:
        q_data: 2d array of q transform data

    """

    fmax = sample_rate / 2
    frange = (fmin, fmax)

    # TODO: un hardcode the value 6 used for querying window
    # create frame cache using gwdatafind
    frames = find_urls(
        site=ifo.strip("1"),
        frametype=f"{ifo}_{frame_type}",
        gpsstart=int(time - 7),
        gpsend=int(time + 7),
        urltype="file",
        on_gaps="ignore",
    )

    # read strain from frames cache
    t = TimeSeries.read(
        frames,
        channel=f"{ifo}:{channel}",
        start=time - 6,
        end=time + 6,
        pad=0,
        nproc=3,
    )

    t.times = t.times.value - time
    print(t)
    # perform q transform
    q_data = t.q_transform(
        qrange=(4, 64),
        frange=frange,
        outseg=Segment(-window, window),
        tres=tres,
        fres=fres,
        whiten=True,
    )
    return q_data


def frac_above_threshold(df: pd.DataFrame, threshold: float):
    frac = np.sum(df[df > threshold].count()) / np.sum(df.count())
    return frac


def calc_pixel_occupancy(
    q_data: np.ndarray,
    fmin: float,
    fres: float,
    window: float,
    threshold: float,
    f_windows: List[float] = [100, 512, 1024],
    t_windows: List[float] = [0.5, 1, 2],
):

    """Calculates saturation percentages

    Args:
        q_data:
            2d np.ndarray of q transform data
        fmin:
            minimum frequency of q_data
        fres:
            frequency resolution of q_data
        window:
            time half window
        threshold:
            Threshold to declare pixel "saturated"
        plot:
            bool to create plot
        savedir:
            where to save data
        f_windows:
            frequency values for creating windows
        t_windows:
            time lengths for creating windows
    """

    # unpack frequency and time windows
    f1, f2, f3 = f_windows
    t1, t2, t3 = t_windows

    # dict to store final pixel occupancy values
    pixel_occupancy = []

    df = pd.DataFrame(q_data)

    # dividing rows and columns of the dataframe

    # get the center index of time dimension
    center_time_idx = df.shape[0] // 2

    # number of indices in one second
    len_one_sec = int(
        df.shape[0] / (2 * window)
    )  # window is actually a half window

    # get index values of frequency windows
    f1_index, f2_index, f3_index = (
        int((f1 - fmin) / fres),
        int((f2 - fmin) / fres),
        int((f3 - fmin) / fres),
    )

    row_ind = [
        int(len_one_sec * t1),
        int(len_one_sec * t2),
        int(len_one_sec * t3),
    ]
    freq_ind = [0, f1_index, f2_index, f3_index]

    # create tuple of indices around center for each time window
    row_inds = [(center_time_idx - i, center_time_idx + i) for i in row_ind]

    # create tuple of indices for frequency windows
    freq_inds = [
        (freq_ind[i], freq_ind[i + 1]) for i in range(len(freq_ind) - 1)
    ]

    above_thresh = []

    # for each time, frequency window pair
    for j in freq_inds:
        for i in row_inds:
            vals = frac_above_threshold(
                df.iloc[i[0] : i[1], j[0] : j[1]], threshold
            )

            above_thresh.append(100 * vals)

    dftf = pd.DataFrame(
        np.reshape(above_thresh, (3, 3)),
        columns=["t1", "t2", "t3"],
        index=["f1", "f2", "f3"],
    )

    # flatten df values
    pixel_occupancy = dftf.values.flatten()

    return pixel_occupancy