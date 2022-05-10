import logging
import os
from typing import List

import h5py
import numpy as np
from hermes.typeo import typeo
from qoq.core import calc_pixel_occupancy, check_state_vector, query_q_data
from qoq.logging import configure_logging


@typeo
def main(
    injection_file: str,
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
    """Generate q transforms and calculate pixel occupancy values
    for O3 replay mdc injections. Produces one output file.
    Checks state vector for each ifo to ensure the detector is in science mode.
    If in science mode, pixel occupancy values will be stored for that ifo.
    This means that there will be an uneven number of H1 and L1 events
    depending on how many are injected during science time for each ifo.

    Args:
        injection_file:
            Path to the h5 file used to generate the mdc's.
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

    # configure logging
    configure_logging(filename=os.path.join(out_dir, "log.log"))

    # load in xml file of injections
    events = h5py.File(injection_file, "r")["events"][()]

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

    os.makedirs(out_dir, exist_ok=True)

    q_data = {}
    pixel_occupancies = {}
    true_times = {}
    replay_times = {}
    event_info = {}

    for ifo in ifos:
        q_data[ifo] = []
        pixel_occupancies[ifo] = []
        replay_times[ifo] = []
        true_times[ifo] = []
        event_info[ifo] = []

    logging_cadence = 50
    out_file = os.path.join(out_dir, "injections.h5")

    # loop over events
    for i, event in enumerate(events):

        if i % logging_cadence == 0:
            logging.info(f"Completed analysis for {i} events")

        for ifo in ifos:

            # convert to replay time
            true_time = event[f'time_{ifo.strip("1")}']
            time = true_time + offset

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

            # if this ifo is in science mode
            if good_data_bool:

                try:
                    # make q gram
                    data = query_q_data(
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

                    # calculate pixel occupancy values
                    pixel_occupancy = calc_pixel_occupancy(
                        data,
                        fmin,
                        fres,
                        window,
                        threshold,
                        f_windows,
                        t_windows,
                    )

                    # store q_data in master dict
                    q_data[ifo].append(data.value)

                    # store pixel_occ in master dict
                    pixel_occupancies[ifo].append(pixel_occupancy)

                    # store true time, replay time, and event info
                    true_times[ifo].append(true_time)
                    replay_times[ifo].append(time)
                    event_info[ifo].append(event)

                except Exception as e:
                    logging.error(e)
                    continue

            # if we arent in science mode check next ifo
            elif not good_data_bool:
                continue

    logging.info("Saving data to h5 file")
    # for each ifo in science mode store data
    with h5py.File(out_file, "w") as f:
        for ifo in ifos:
            ifo_gr = f.create_group(ifo)

            # if we want to store raw q data
            if store_raw:
                ifo_gr.create_dataset("q_data", data=q_data[ifo])

            # store pixel occupancy values
            ifo_gr.create_dataset("pixel_occ", data=pixel_occupancies[ifo])

            # store replay time, true time, and event info
            # i.e. (injection parameters)
            ifo_gr.create_dataset("replay time", data=replay_times[ifo])
            ifo_gr.create_dataset("true time", data=true_times[ifo])
            ifo_gr.create_dataset("event parameters", data=event_info[ifo])

        # store q transform / q pixel occ info to reproduce results
        f.attrs.update(
            {
                "t_windows": t_windows,
                "f_windows": f_windows,
                "window": window,
                "fres": fres,
                "tres": tres,
                "fmin": fmin,
            }
        )

    return out_file


if __name__ == "__main__":
    main()
