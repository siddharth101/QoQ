import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List

import h5py
import numpy as np
from hermes.typeo import typeo
from qoq.core import calc_pixel_occupancy, query_q_data
from qoq.logging import configure_logging


def process_one_pycbc_file(
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
    logging_cadence: int,
    trigger_file: str,
    template_file: str,
):
    """Generates q transforms and pixel occupancy for
        pycbc background trigger file

    Args:
        trigger_file:
            Path to a pycbc file of background timeslides
        template_file:
            Path to mass template file corresponding to trigger file
        ifos: Which ifos to analyze
        m1_cut_low: minimum m1 to consider
        m2_cut_low: minimum m2 to consider
        window: half window of data to query for q transform
        strain_channel: strain channel to query data for
        frame_type: frame type for querying data w/ gwdatafind
        fmin: minimum frequency for calculating q transform
        fres: frequency res for calculating q transform
        tres: time res for calcualting q transform
    """

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

    # initiate arrays to store m1s, m2s, and ifars
    m1s = []
    m2s = []
    ifars_out = []

    # dict to store pixel occs;
    # key is ifo
    pixel_occupancies = {}

    # dict to store q_data;
    # key is ifo
    q_data = {}

    # dict to store trigger times
    times = {}

    # for each ifo inititate array
    for ifo in ifos:
        q_data[ifo] = []
        pixel_occupancies[ifo] = []
        times[ifo] = []

    # loop over idxs of events that pass far thresh
    for i, idx in enumerate(idxs):

        if i % logging_cadence == 0:
            logging.info(f"Completed analysis of {i} events")

        # get template_id for this event
        template_id = template_ids[idx]

        # get m1 and m2 for template id
        m1 = template_data["mass1"][template_id]
        m2 = template_data["mass2"][template_id]

        # check that masses pass cuts
        if m1 < m1_cut_low or m2 < m2_cut_low:
            continue

        # append masses and ifar of event if all cuts are passed
        m1s.append(m1)
        m2s.append(m2)
        ifars_out.append(ifars[idx])

        for ifo in ifos:

            # get trigger time for this ifo
            time = trigger_data["background_exc"][ifo]["time"][idx]

            # append time
            times[ifo].append(time)

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

            # store in dict
            q_data[ifo].append(data.value)
            # calc pixel occ for this ifo
            pixel_occupancy = calc_pixel_occupancy(
                data,
                fmin,
                fres,
                window,
                threshold,
                f_windows,
                t_windows,
            )

            # append pixel occ information
            pixel_occupancies[ifo].append(pixel_occupancy)

    for ifo in ifos:
        pixel_occupancies[ifo] = np.array(pixel_occupancies[ifo])
        times[ifo] = np.array(times[ifo])

    m1s = np.array(m1s)
    m2s = np.array(m2s)
    ifars_out = np.array(ifars_out)

    return pixel_occupancies, q_data, times, m1s, m2s, ifars_out


@typeo
def main(
    pycbc_data_dir: str,
    pycbc_template_dir: str,
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
    logging_cadence: int,
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

    # configure logging
    configure_logging(filename=os.path.join(out_dir, "log.log"))

    ifo_str = "".join(ifos)

    # load in trigger files and template files from pycbc dirs
    trigger_files = glob.glob(pycbc_data_dir + ifo_str + "-EXCLUDE_ZEROLAG*")
    template_files = glob.glob(pycbc_template_dir + ifo_str + "*")

    # sort trigger and template files (by time)
    # so the first trigger file corresponds to first template file
    trigger_files = np.sort(trigger_files)
    template_files = np.sort(template_files)

    if len(trigger_files) != len(template_files):
        raise ValueError(
            f" '{len(template_files)}' template files do not match the "
            f" '{len(trigger_files)}' trigger_files"
        )

    # initiate output data arrays/dicts
    pixel_occupancies = {}
    q_data = {}
    times = {}

    for ifo in ifos:
        pixel_occupancies[ifo] = []
        q_data[ifo] = []
        times[ifo] = []

    m1s = []
    m2s = []
    ifars = []

    # define function to pass to map
    partial_process_func = partial(
        process_one_pycbc_file,
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
        logging_cadence,
    )

    logging.info("Submitting proceses to the pool")

    # create pool as context manager
    with ProcessPoolExecutor() as executor:

        # loop over files, appending output to master arrays/dicts
        for result in executor.map(
            partial_process_func, trigger_files, template_files
        ):

            # unpack the result
            (
                pixel_occupancies_tmp,
                q_data_tmp,
                times_tmp,
                m1s_tmp,
                m2s_tmp,
                ifars_tmp,
            ) = result

            m1s.extend(m1s_tmp)
            m2s.extend(m2s_tmp)
            ifars.extend(ifars_tmp)

            for ifo in ifos:
                pixel_occupancies[ifo].extend(pixel_occupancies_tmp[ifo])
                q_data[ifo].extend(q_data_tmp[ifo])
                times[ifo].extend(times_tmp[ifo])

    out_file = os.path.join(out_dir, "background.h5")

    logging.info("Saving data to h5 file")

    # for each ifo in science mode store data
    with h5py.File(out_file, "w") as f:
        for ifo in ifos:

            ifo_gr = f.create_group(ifo)

            # if we want to store raw q data
            if store_raw:
                ifo_gr.create_dataset("q_data", data=np.array(q_data[ifo]))

            # store pixel occupancy values
            ifo_gr.create_dataset(
                "pixel_occ", data=np.array(pixel_occupancies[ifo])
            )

            # store trigger time
            ifo_gr.create_dataset("times", data=times[ifo])

        # store m1, m2, ifar
        f.create_dataset("m1", data=m1s)
        f.create_dataset("m2", data=m2s)
        f.create_dataset("ifar", data=ifars)

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


if __name__ == "__main__":
    main()
