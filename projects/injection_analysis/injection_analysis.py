import logging
import os
from typing import List

import h5py
import numpy as np
from hermes.typeo import typeo

from retraction_scripts.utils import (
    calc_pixel_occupancy,
    check_state_vector,
    query_q_data,
)


@typeo
def main(
    injection_xml_path: str,
    ifos: List,
    offset: float,
    snr_cut_low: float,
    snr_cut_high: float,
    m1_cut_low: float,
    m2_cut_low: float,
    science_mode_bitmask: int,
    window: float,
    strain_channel: str,
    frame_type: str,
    sample_rate: float,
    state_channel: str,
    fmin: float,
    fres: float,
    tres: float,
    f_windows: List[float],
    t_windows: List[float],
    threshold: float,
    store_raw: bool,
    out_dir: str,
):
    """Generate q_transforms for O3 replay mdc injections.

    Args:
        injection_xml_path:
            Path to the xml file used to generate the mdc's.
            See https://git.ligo.org/shaon.ghosh/injection_campaign_studies/
            For the xml file used for O3 replay MDC
        ifos: Which ifos to analyze
        offset:
            offset in seconds between true injection time (i.e. O3 time),
            and time of replay.
            For the list of offsets used during the replay,
            see https://wiki.ligo.org/Computing/DASWG/O3EndToEndReplay
        snr_cut_low: minimum single IFO snr of events
        snr_cut_high: maximum single IFO snr of events
        m1_cut_low: minimum m1 to consider
        m2_cut_low: minimum m2 to consider
        science_mode_bitmask: bitmask for science mode
        window: half window of data to query for q transform
        strain_channel: strain channel to query data for
        frame_type: frame type for querying data w/ gwdatafind
        state_channel: state channel to check science mode
        fmin: minimum frequency for calculating q transform
        fres: frequency res for calculating q transform
        tres: time res for calcualting q transform
    """
    # load in xml file of injections
    events = h5py.File(injection_xml_path, "r")["events"][()]

    # apply snr and mass cuts
    events = events[
        np.logical_and(
            events["snr_H"] > snr_cut_low, events["snr_H"] < snr_cut_high
        )
    ]

    events = events[
        np.logical_and(
            events["snr_L"] > snr_cut_low, events["snr_L"] < snr_cut_high
        )
    ]
    events = events[
        np.logical_and(
            events["mass1"] > m1_cut_low, events["mass2"] > m2_cut_low
        )
    ]

    for ifo in ifos:
        os.makedirs(out_dir, exist_ok=True)

    events = events[:2]
    # loop over events
    for event in events:

        # list to store ifos in science mode
        good_ifos = []

        # dict to store data
        data = {}
        for ifo in ifos:

            # convert to replay time
            time = event[f'time_{ifo.strip("1")}'] + offset
            # check if we have good data for this ifo
            try:
                good_data_bool = check_state_vector(
                    ifo,
                    state_channel,
                    time,
                    window,
                    science_mode_bitmask,
                    frame_type,
                )
            except Exception as e:
                logging.error(e)
                continue

            data[ifo] = None
            # if this ifo is in science mode
            if good_data_bool:

                try:
                    # make q gram
                    q_data = query_q_data(
                        ifo,
                        time,
                        window,
                        strain_channel,
                        frame_type,
                        sample_rate,
                        fmin,
                        tres,
                        fres,
                    )
                    # store in dict
                    data[ifo] = q_data.value
                    # append ifo
                    good_ifos.append(ifo)

                except Exception as e:
                    logging.error(e)
                    continue

            # if we arent in science mode check next ifo
            elif not good_data_bool:
                continue

        # if no science mode ifos continue
        if len(good_ifos) == 0:
            continue

        # concat science mode ifos into str for file name
        ifo_str = "".join(good_ifos)

        out_file = os.path.join(out_dir, f"{int(time)}_{ifo_str}.h5")

        # now, calculate pixel occupancy values
        pixel_occupancy = calc_pixel_occupancy(
            data,
            fmin,
            fres,
            window,
            threshold,
            good_ifos,
            f_windows,
            t_windows,
        )

        # for each ifo in science mode store data
        with h5py.File(out_file, "w") as f:
            for ifo in good_ifos:

                # if we want to store raw q data
                if store_raw:
                    f.create_dataset(f"{ifo}_q_data", data=data[ifo])

                # store pixel occupancy values
                f.create_dataset(f"{ifo}_pixel_occ", data=pixel_occupancy[ifo])

            # store info to reproduce results
            f.attrs.update(
                {
                    "t_windows": t_windows,
                    "f_windows": f_windows,
                    "window": window,
                    "replay time": time,
                    "true time": time - offset,
                    "event": event,
                    "fres": 0.05,
                    "tres": 0.01,
                    "fmin": fmin,
                }
            )


if __name__ == "__main__":
    main()
