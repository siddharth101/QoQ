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
