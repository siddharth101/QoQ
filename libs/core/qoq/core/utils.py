from gwdatafind import find_urls
from gwpy.timeseries import StateVector


def check_state_vector(
    ifo: str,
    channel: str,
    time: float,
    window: float,
    bitmask: int,
    frame_type: str,
):
    """Checks state vector to ensure data being analyzed is in science mode.
       If any sample is not in science mode, data is flagged as bad

    Args:
        ifo: Which interferometer to query data for
        channel: StateVector channel to query
        time: trigger time
        window: half window around trigger time to query
        bitmask: bitmask that declares science mode data

    Returns:
        bool: True if bitmask is set (i.e. good data)
    """

    # create frame cache using gwdatafind
    frames = find_urls(
        site=ifo.strip("1"),
        frametype=f"{ifo}_{frame_type}",
        gpsstart=int(time - window - 1),
        gpsend=int(time + window + 1),
        urltype="file",
        on_gaps="ignore",
    )

    # read state vector from frames cache
    state = StateVector.read(
        frames,
        channel=f"{ifo}:{channel}",
        start=time - window,
        end=time + window,
    )

    # if any bitmask is not in science mode, flag as not good data
    for value in state.value:

        # if given bitmask is set, check next bit
        if ((int(value) & bitmask) == bitmask) and (int(value) >= 0):
            continue

        # otherwise declare abd data
        else:
            return False

    # if all states passed, we have good data
    return True


'''
below function is deprecated


def load_raw_data(filepath: str, ifo: str):
    """Reads filepath produced by one of the projects
       in the repo, loading in the raw q data

    Args:
        filepath: path to h5 file
        ifo: which ifo to read data

    Returns:
        gwpy.Spectrogram of data
    """

    try:
        with h5py.File(filepath, "r") as f:
            data = f["{}_q_data".format(ifo)]
    except KeyError:
        raise ValueError(f"raw q data for {ifo} not stored in this file")

    data_frame = pd.DataFrame(list(data))
    f = h5py.File(filepath, "r")
    fres = f.attrs["fres"]
    tres = f.attrs["tres"]
    fmin = f.attrs["fmin"]
    window = f.attrs["window"]

    n_times = window / tres

    spec = Spectrogram(data_frame.values, dt=tres, df=fres, f0=fmin)
    spec.times = np.linspace(-window, window, n_times)

    return spec

'''
