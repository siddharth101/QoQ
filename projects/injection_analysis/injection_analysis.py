import logging
import uuid
from pathlib import Path
from typing import List

import h5py
import numpy as np
from gwpy.segments import Segment
from gwpy.timeseries import StateVector, TimeSeries
from hermes.typeo import typeo
from qoq.core import calc_pixel_occupancy, check_state_vector, query_mdc_frames
from qoq.logging import configure_logging


@typeo
def main(
    injection_file: str,
    ifos: List[str],
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
    f_windows: List[float] = None,
    t_windows: List[float] = None,
    threshold: float = 60,
    store_raw: bool = True,
    store_pixel_occ: bool = False,
    out_dir: Path = "./data",
    frame_dir: Path = Path(
        "/home/shaon.ghosh/ANALYSIS/O4/analysis/DMTGen_stuff/frame_outputs"
    ),
    logging_cadence: int = 50,
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
        f_windows: frequency windows with which to calculate pixel occ
        t_windows: time windows with which to calculate pixel occ

        calc_pixel_occ: whether to calculate and store pixel occupancy values
        out_dir: where to store data
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # configure logging
    configure_logging(filename=out_dir.joinpath("log.log"), verbose=False)

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

    # infer frange of qscan from
    # passed parameters
    fmax = sample_rate / 2
    frange = (fmin, fmax)

    # create dictionaries to store data
    # initializing a list for each ifo
    pixel_occupancies = {}
    event_info = {}

    if store_raw:
        raw_data_dir = out_dir.joinpath("raw")
        raw_data_dir.mkdir(parents=True, exist_ok=True)

    for ifo in ifos:

        pixel_occupancies[ifo] = []
        event_info[ifo] = []

    out_file = out_dir.joinpath("injections.h5")

    logging.info(f"Starting analysis for total of {len(events)}")

    # loop over events
    for i, event in enumerate(events):

        # cadence at which to log progress
        if i % logging_cadence == 0:
            logging.info(f"Completed analysis for {i} events")

        # one file per event
        file_label = str(uuid.uuid4()) + ".h5"

        for ifo in ifos:

            # get event time
            time = event[f'time_{ifo.strip("1")}']

            # query mdc frames
            frames = query_mdc_frames(ifo, time, window, frame_dir)

            # read in statevector around event
            state_vector = StateVector.read(
                frames,
                channel=f"{ifo}:{state_channel}",
                start=time - window,
                end=time + window,
            )

            # check for good data quality / science mode
            good_data_bool = check_state_vector(
                state_vector,
                science_mode_bitmask,
            )

            # if this ifo is in science mode
            if good_data_bool:

                # read in strain data
                ts = TimeSeries.read(
                    frames,
                    channel=f"{ifo}:{strain_channel}",
                    start=time - window - 4,
                    end=time + window + 4,
                    pad=0,
                    nproc=3,
                )
                ts.times = ts.times.value - time

                # perform q transform
                data = ts.q_transform(
                    qrange=(4, 64),
                    frange=frange,
                    outseg=Segment(-window, window),
                    tres=tres,
                    fres=fres,
                    whiten=True,
                )

                # if we want to calc pixel occ for specified windows
                if store_pixel_occ:

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

                    # store pixel_occ and
                    # other information in master dict
                    pixel_occupancies[ifo].append(pixel_occupancy)
                    event_info[ifo].append(event)

                # append raw q data for this ifo
                # to event file
                if store_raw:
                    with h5py.File(
                        raw_data_dir.joinpath(file_label), "a"
                    ) as f:
                        f.create_dataset(f"q_data_{ifo}", data=data.value)

            # if we arent in science mode check next ifo
            elif not good_data_bool:
                logging.info(
                    f"Bad data for event {i} at {time} in {ifo}. Skipping"
                )
                continue

        if store_raw:
            # update attributes with q transform info
            with h5py.File(raw_data_dir.joinpath(file_label), "a") as f:
                # and event info
                f.attrs.update(
                    {
                        "ifos": ifos,
                        "event": event,
                        "fres": fres,
                        "tres": tres,
                        "fmin": fmin,
                        "window": window,
                    }
                )

    if store_pixel_occ:
        logging.info("Saving all pixel occ data to single h5 file")
        # for each ifo in science mode store data
        with h5py.File(out_file, "w") as f:
            for ifo in ifos:
                ifo_gr = f.create_group(ifo)

                # store pixel occupancy values
                ifo_gr.create_dataset("pixel_occ", data=pixel_occupancies[ifo])

                # store replay time, true time, and event info
                # i.e. (injection parameters)

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
