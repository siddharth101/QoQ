import glob
import logging
import os
from typing import List

import h5py
import numpy as np
from hermes.typeo import typeo

from retraction_scripts.utils import calc_pixel_occupancy, query_q_data


def process_one_pycbc_file(
    trigger_file: str,
    template_file: str,
    ifos: List,
    m1_cut_low: float,
    m2_cut_low: float,
    ifar_thresh: float,
    window: float,
    strain_channel: str,
    frame_type: str,
    sample_rate: float,
    fmin: float,
    fres: float,
    tres: float,
    f_windows: List[float],
    t_windows: List[float],
    threshold: float,
    store_raw: bool,
    out_dir: str,
):
    """Generates q transforms and pixel occupancy for
        pycbc background trigger file

    Args:
        trigger_file:
            Path to a pycbc file of background timeslides
        template_file:
            Path to mass template file corresponding to trigger file
        ifos: Which ifos to analyze
        snr_cut_low: minimum single IFO snr of events
        snr_cut_high: maximum single IFO snr of events
        m1_cut_low: minimum m1 to consider
        m2_cut_low: minimum m2 to consider
        window: half window of data to query for q transform
        strain_channel: strain channel to query data for
        frame_type: frame type for querying data w/ gwdatafind
        fmin: minimum frequency for calculating q transform
        fres: frequency res for calculating q transform
        tres: time res for calcualting q transform
    """

    os.makedirs(out_dir, exist_ok=True)

    ifo_str = "".join(ifos)
    trigger_data = h5py.File(trigger_file)
    template_data = h5py.File(template_file)

    # get ifars
    ifars = trigger_data["background_exc"]["ifar"][()]

    # indices that pass ifar threshold
    idxs = np.where(ifars > ifar_thresh)[0]

    # get template ids for all triggers
    template_ids = trigger_data["background_exc"]["template_id"]

    # number of events
    n_events = len(idxs)

    logging.info(
        f"Possibly generating q data for {n_events}, "
        " assuming all pass mass cuts"
    )

    # loop over idxs of events that pass far thresh
    for idx in idxs:

        # get template_id for this event
        template_id = template_ids[idx]

        # get m1 and m2 for template id
        m1 = template_data["mass1"][template_id]
        m2 = template_data["mass2"][template_id]

        # check that masses pass cuts
        if m1 < m1_cut_low or m2 < m2_cut_low:
            continue

        # dict to store data
        data = {}
        for ifo in ifos:

            time = trigger_data["background_exc"][ifo]["time"][idx]

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

        # hard code file label as L1 time
        # for some reason, the trigger_id column
        # is different for H1 and L1
        # not sure better option to use for now
        file_label = round(
            trigger_data["background_exc"]["L1"]["time"][idx], 2
        )
        out_file = os.path.join(out_dir, f"{file_label}_{ifo_str}.h5")

        # now, calculate pixel occupancy values
        pixel_occupancy = calc_pixel_occupancy(
            data,
            fmin,
            fres,
            window,
            threshold,
            ifos,
            f_windows,
            t_windows,
        )

        # for each ifo in science mode store data
        with h5py.File(out_file, "w") as f:
            for ifo in ifos:

                # if we want to store raw q data
                if store_raw:
                    f.create_dataset(f"{ifo}_q_data", data=data[ifo])

                # store pixel occupancy values
                f.create_dataset(f"{ifo}_pixel_occ", data=pixel_occupancy[ifo])

                # store trigger time for ifo
                f.attrs.update(
                    {
                        f"{ifo} time": trigger_data["background_exc"][ifo][
                            "time"
                        ][idx]
                    }
                )

            # store info to reproduce results
            f.attrs.update(
                {
                    "m1": m1,
                    "m2": m2,
                    "ifar": ifars[idx],
                    "t_windows": t_windows,
                    "f_windows": f_windows,
                    "window": window,
                    "fres": 0.05,
                    "tres": 0.01,
                    "fmin": fmin,
                }
            )


@typeo
def main(
    pycbc_data_dir: str,
    pycbc_template_dir: str,
    ifos: List,
    snr_cut_low: float,
    snr_cut_high: float,
    m1_cut_low: float,
    m2_cut_low: float,
    ifar_thresh: float,
    window: float,
    strain_channel: str,
    frame_type: str,
    sample_rate: float,
    fmin: float,
    fres: float,
    tres: float,
    f_windows: List[float],
    t_windows: List[float],
    threshold: float,
    store_raw: bool,
    out_dir: str,
):
    """Generates q transform data and pixel occupancy values
       for pycbc triggers

    Args:
        pycbc_data_dir:
            Path to directory where pycbc trigger files are stored
        pycbc_template_dir:
            Path to directory where corresponding
            pycbc mass templates are stroed
        ifos: Which ifos to analyze
        snr_cut_low: minimum single IFO snr of events
        snr_cut_high: maximum single IFO snr of events
        m1_cut_low: minimum m1 to consider
        m2_cut_low: minimum m2 to consider
        window: half window of data to query for q transform
        strain_channel: strain channel to query data for
        frame_type: frame type for querying data w/ gwdatafind
        fmin: minimum frequency for calculating q transform
        fres: frequency res for calculating q transform
        tres: time res for calcualting q transform
    """

    ifo_str = "".join(ifos)

    trigger_files = glob.glob(pycbc_data_dir + ifo_str + "-EXCLUDE_ZEROLAG*")
    template_files = glob.glob(pycbc_template_dir + ifo_str + "*")

    trigger_files = np.sort(trigger_files)
    template_files = np.sort(template_files)

    if len(trigger_files) != len(template_files):
        raise ValueError(
            f" '{len(template_files)}' template files do not match the "
            f" '{len(trigger_files)}' trigger_files"
        )

    for trigger_file, template_file in zip(trigger_files, template_files):
        process_one_pycbc_file(
            trigger_file,
            template_file,
            ifos,
            m1_cut_low,
            m2_cut_low,
            ifar_thresh,
            window,
            strain_channel,
            frame_type,
            sample_rate,
            fmin,
            fres,
            tres,
            f_windows,
            t_windows,
            threshold,
            store_raw,
            out_dir,
        )


if __name__ == "__main__":
    main()
