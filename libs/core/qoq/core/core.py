from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
from gwdatafind import find_urls
from gwpy.spectrogram import Spectrogram
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

def read_h5_data(filepath, tres=0.01, fres=0.05, fmin=10, ifo='L1'):
	f = h5py.File(filepath, 'r')
	datafr = pd.DataFrame(list(f['{}_q_data'.format(ifo)]))
	spec = Spectrogram(datafr, dt=tres, df=fres, f0=fmin)
	dur = np.round(spec.times.value[-1]) # duration
	win = dur//2
	len_one_sec = int(1/tres)
	spec.times = np.linspace(-win, win, int(dur)*(len_one_sec)) 

	return spec, win

def frac_above_threshold(df: pd.DataFrame, threshold: float):
    frac = np.sum(df[df > threshold].count()) / np.sum(df.count())
    return frac


def calc_pixel_occupancy(
    q_data: np.ndarray,
    fmin: float,
    fres: float,
    window: float,
    threshold: float,
    mut_exc=False,
    f_windows: List[float]=[100, 512, 1024],
    t_windows: List[float]=[0.5, 1, 2],
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
    # Mutual Exclusive
    if mut_exc:
        ### dividing rows and columns of the dataframe
        a = df.shape[0]//2  # center index
        len_one_sec = int(df.shape[0]/(2*window))  # int(1/0.01) 1/tres
        b = window*len_one_sec  # window*1/0.01
        c = int((b - len_one_sec//2))
        t_inds = np.arange(0, c, len_one_sec)
        central_inds = [a - len_one_sec//2, a + len_one_sec//2]
        row_ind = [central_inds[0] - j for j in t_inds[::-1]] + [j + central_inds[1] for j in t_inds]
        row_ind.insert(0, 0)
        row_ind.append(df.shape[0])
        row_inds = [(row_ind[i], row_ind[i+1]) for i in range(len(row_ind)-1)]
        f1_index, f2_index, f3_index = int((f1 - 10)*1/fres), int((f2 - 10)*1/fres), int((f3 - 10)*1/fres)
        freq_ind = [0, f1_index, f2_index, f3_index]
        freq_inds = [(freq_ind[i], freq_ind[i+1]) for i in range(len(freq_ind)-1)]

    # dividing rows and columns of the dataframe
    else:
    # get the center index of time dimension
        center_time_idx = df.shape[0] // 2

    # number of indices in one second
        len_one_sec = int(df.shape[0] / (2 * window))  # window is actually a half window

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
                df.iloc[i[0]:i[1], j[0]:j[1]], threshold
            )

            above_thresh.append(100 * vals)

    #dftf = pd.DataFrame(
   #     np.reshape(above_thresh, (3, 3)),
   #     columns=["t1", "t2", "t3"],
   #     index=["f1", "f2", "f3"],
 #   )
    dftf = pd.DataFrame(np.reshape(above_thresh, (len(freq_inds), len(row_inds))),
                                     columns=['t'+'{}'.format(i) for i in range(1, len(row_inds)+1)],
                          index=['f1', 'f2', 'f3'])
    # flatten df values
    pixel_occupancy = dftf.values.flatten()

    return pixel_occupancy


def raw_data_dir_to_pixel_occ(
    raw_data_dir: Path,
    out_dir: Path,
    out_file_label: str,
    f_windows: List[float],
    t_windows: List[float],
    threshold: float,
):
    """
    Takes a path to a data directory of individual
    q data files produced by projects, and calculates pixel occupancy
    values given frequency and time windows. Stores in one h5 file with
    all relevant info about the events
    """

    out_file = out_dir / out_file_label

    pixel_occs = []
    # get all the files
    files = raw_data_dir.glob("*")

    # for each file
    for file_path in files:
        with h5py.File(file_path) as f:
            # get q data
            q_data = f["q_data"]
            fmin = f.attrs["fmin"]
            fres = f.attrs["fres"]
            tres = f.attrs["tres"]
            window = f.attrs["window"]

        # calculate pixel occupancy values
        pixel_occupancy = calc_pixel_occupancy(
            q_data,
            fmin,
            fres,
            window,
            threshold,
            f_windows,
            t_windows,
        )

        pixel_occs.append(pixel_occupancy)

    with h5py.File(out_file, "w") as f:
        f.create_dataset("pixel_occs", data=pixel_occs)
        f.attrs.update(
            {
                "f_windows": f_windows,
                "t_windows": t_windows,
                "fres": fres,
                "threshold": threshold,
                "window": window,
                "fmin": fmin,
                "tres": tres,
            }
        )
