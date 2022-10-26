from gwpy.timeseries import StateVector


def check_state_vector(
    state_vector: StateVector,
    bitmask: int,
):
    """Checks state vector to ensure data being analyzed is in science mode.
       If any sample is not in science mode, data is flagged as bad

    Args:
        state_vector: gwpy StateVector object to check
        bitmask: bitmask that declares science mode data

    Returns:
        bool:
            True if required bitmask is set
            for all times in StateVector (i.e. good data)
    """

    # if any bitmask is not in science mode, flag as not good data
    for value in state_vector.value:

        # if given bitmask is set, check next bit
        if ((int(value) & bitmask) == bitmask) and (int(value) >= 0):
            continue

        # otherwise declare abd data
        else:
            return False

    # if all states passed, we have good data
    return True
