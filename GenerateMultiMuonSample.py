

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import random
import os
import h5py
import numpy as np

IMAGE_SHAPE = (40, 40, 38)
PMT_LABELS = "PMTlabelSheet3.csv"

def count_events(files):
    # Because we want to remove events with 0 hits, 
    # we need to count the events beforehand (to create the h5 file).
    # This function counts and indexes the events with more than 0 hits.
    # Files need to be iterated in the same order to use the indexes.
    """ This is where we manually specify the file"""
    num_events = 0
    nonzero_file_events = []
    for file_index, f in enumerate(files):
        data = np.load(f, allow_pickle=True)
        nonzero_file_events.append([])
        hits = data['digi_hit_pmt']
        for i in range(len(hits)):
            if len(hits[i]) != 0:
                nonzero_file_events[file_index].append(i)
                num_events += 1
    return num_events, nonzero_file_events

def GenMapping(csv_file):
    mPMT_to_index = {}
    with open(csv_file) as f:
        rows = f.readline().split(",")[1:]
        rows = [int(r.strip()) for r in rows]

        for line in f:
            line_split = line.split(",")
            col = int(line_split[0].strip())
            for row, value in zip(rows, line_split[1:]):
                value = value.strip()
                if value:  # If the value is not empty
                    mPMT_to_index[int(value)] = [col, row]
    npmap = np.zeros((max(mPMT_to_index) + 1, 2), dtype=np.int)
    for k, v in mPMT_to_index.items():
        npmap[k] = v
    return npmap

def SumEvents(event_numbers, time_offsets, datafile, only_noise=False):
    """
    This function sums the events in the list of event indices into the datafile.
    
    Inputs:
    event_numbers is list of indices into events in datfile to sum be summed
    time_offsets is list of time offests for each event (only used for >1 event)
    if only_noise true, then only return the noise hits from the first event in the list.
    
    Return tuple of ( tube, charge, time)
    tube is np.array of tubes (unique without duplicates)
    charge is np.array of charges (summed from all events hitting each corresponding tube)
    time is np.array of times (earliest from all events hitting each corresponding tube)
    
    Note: only take the noise hits from the first event in the list of event numbers.  
    """

    tube = np.array([], dtype=np.int32)
    charge = np.array([])
    time = np.array([])

    nev = len(event_numbers)
    if nev == 0:
        return (tube, charge, time)

        # Start with first event in list
    ievnum = event_numbers[0]
    if only_noise:
        noisepmts, signalpmts = GetNoise_and_Signal_PMTLists(ievnum, datafile)
        tube, charge, time = PMTChargeTime_in_list(ievnum, noisepmts, datafile)
    else:
        tube = datafile['digi_hit_pmt'][ievnum]
        charge = datafile['digi_hit_charge'][ievnum]
        time = datafile['digi_hit_time'][ievnum]

        # For remaining events only look at signal events
    for iev in range(1, len(event_numbers)):
        ievnum = event_numbers[iev]
        toffset = time_offsets[iev]
        noisepmts, signalpmts = GetNoise_and_Signal_PMTLists(ievnum, datafile)
        curtube, curcharge, curtime = PMTChargeTime_in_list(ievnum, signalpmts, datafile)
        for idx, pmt in enumerate(curtube):
            if np.isin(pmt, tube):
                for jdx, jpmt in enumerate(tube):
                    if jpmt == pmt:
                        charge[jdx] += curcharge[idx]
                        if curtime[idx] + toffset < time[jdx]:
                            time[jdx] = curtime[idx] + toffset
            else:
                # new pmt not in list yet... append
                tube = np.append(tube, pmt)
                charge = np.append(charge, curcharge[idx])
                time = np.append(time, curtime[idx] + toffset)

    return (tube, charge, time)

def GetNoise_and_Signal_PMTLists(evnum, data):
    """
    Inputs: 
        evnum == index into data for the event to get indices for
        data == npz datafile
        geo == npz geometry file
    
    Return two-tuple np.arrays:  one with the noise hits, and one with the non-noise hits
    """

    truehit_pmts = data['true_hit_pmt'][evnum]
    truehit_parent = data['true_hit_parent'][evnum]
    noisehit_pmts = truehit_pmts[truehit_parent == -1]
    signalhit_pmts = truehit_pmts[truehit_parent != -1]
    return (noisehit_pmts, signalhit_pmts)
    
def PMTChargeTime_in_list(evnum, pmtlist, data):
    """
    Return a tuple containing:
        np.array of pmt-numbers from evnum in data, that is in list of pmts
        np.array of digi-charges from evnum in data, that is in list of pmts
        np.array of digi-times from evnum in data, that is in list of pmts
    """
    allpmts = data['digi_hit_pmt'][evnum]
    allcharges = data['digi_hit_charge'][evnum]
    alltimes = data['digi_hit_time'][evnum]

    s_hit_pmts = np.array([], dtype=np.int32)
    s_hit_charges = np.array([])
    s_hit_times = np.array([])

    for idx, pmt in enumerate(allpmts):
        if np.isin(pmt, pmtlist):
            s_hit_pmts = np.append(s_hit_pmts, pmt)
            s_hit_charges = np.append(s_hit_charges, allcharges[idx])
            s_hit_times = np.append(s_hit_times, alltimes[idx])

    return (s_hit_pmts, s_hit_charges, s_hit_times)

def GenerateMultiMuonSample_h5(avg_mu_per_ev=2.5, sigma_time_offset=21.2):
    """
    Inputs: 
     avg_mu_per_ev == Poisson distribution mean for number of muons in each spill
     sigma_time_offset == Width of spill (Gaussian) in nanoseconds
    """
    files = ['event998.npz']

    # Remove whitespace 
    files = [x.strip() for x in files]

    # Check that files were provided
    if len(files) == 0:
        raise ValueError("No files provided!!")
    print("Merging " + str(len(files)) + " files")

    # Start merging
    num_nonzero_events, nonzero_event_indexes = count_events(files)
    print(num_nonzero_events)

    # np.random.poisson( avg_mu_per_ev, number_of_throws )
    num_muons = np.random.poisson(avg_mu_per_ev, number_of_events - 1)

    #

    dtype_events = np.dtype(np.float32)
    dtype_labels = np.dtype(np.int32)
    dtype_energies = np.dtype(np.float32)
    dtype_positions = np.dtype(np.float32)
    dtype_IDX = np.dtype(np.int32)
    dtype_PATHS = h5py.special_dtype(vlen=str)
    dtype_angles = np.dtype(np.float32)
    h5_file = h5py.File('multimuonfile.h5', 'w')
    dset_event_data = h5_file.create_dataset("event_data",
                                             shape=(num_nonzero_events,) + IMAGE_SHAPE,
                                             dtype=dtype_events)
    dset_labels = h5_file.create_dataset("labels",
                                         shape=(num_nonzero_events,),
                                         dtype=dtype_labels)
    dset_energies = h5_file.create_dataset("energies",
                                           shape=(num_nonzero_events, 1),
                                           dtype=dtype_energies)
    dset_positions = h5_file.create_dataset("positions",
                                            shape=(num_nonzero_events, 1, 3),
                                            dtype=dtype_positions)
    dset_IDX = h5_file.create_dataset("event_ids",
                                      shape=(num_nonzero_events,),
                                      dtype=dtype_IDX)
    dset_PATHS = h5_file.create_dataset("root_files",
                                        shape=(num_nonzero_events,),
                                        dtype=dtype_PATHS)
    dset_angles = h5_file.create_dataset("angles",
                                         shape=(num_nonzero_events, 2),
                                         dtype=dtype_angles)

    # 22 -> gamma, 11 -> electron, 13 -> muon
    # corresponds to labelling used in CNN with only barrel
    # IWCDmPMT_4pi_full_tank_gamma_E0to1000MeV_unif-pos-R371-y521cm_4pi-dir_3000evts_329.npz has an event
    # with pid 11 though....
    # pid_to_label = {22:0, 11:1, 13:2}

    offset = 0
    offset_next = 0
    mPMT_to_index = GenMapping(PMT_LABELS)
    # Loop over files
    for file_index, filename in enumerate(files):
        data = np.load(filename, allow_pickle=True)
        nonzero_events_in_file = len(nonzero_event_indexes[file_index])
        x_data = np.zeros((nonzero_events_in_file,) + IMAGE_SHAPE,
                          dtype=dtype_events)
        digi_hit_pmt = data['digi_hit_pmt']
        # digi_hit_charge = data['digi_hit_charge']
        # digi_hit_time = data['digi_hit_time']
        # digi_hit_trigger = data['digi_hit_trigger']
        # trigger_time = data['trigger_time']
        delay = 0
        # Loop over events in file
        # Loop over number of muons in each event
        event_id = np.array([], dtype=np.int32)
        root_file = np.array([], dtype=np.str)
        pid = np.array([])
        position = np.array([])
        direction = np.array([])
        energy = np.array([])
        labels = np.array([])

        for i, nmu in enumerate(num_muons):
            print("processing output entry ", i, " with ", nmu, " muons")
            indices = np.random.randint(0, len(digi_hit_pmt), max(1, nmu))
            time_offs = [0.]
            if nmu > 1:
                time_offs = np.append(time_offs, np.random.normal(0., sigma_time_offset, nmu - 1))
            hit_pmts, charge, time = SumEvents(indices, time_offs, data, nmu == 0)
            hit_mpmts = hit_pmts // 19
            pmt_channels = hit_pmts % 19
            rows = mPMT_to_index[hit_mpmts, 0]
            cols = mPMT_to_index[hit_mpmts, 1]
            x_data[i - delay, rows, cols, pmt_channels] = charge
            x_data[i - delay, rows, cols, pmt_channels + 19] = time

            # fix below!!!
            idx0 = indices[0]
            event_id = np.append(event_id, data['event_id'][idx0])
            root_file = np.append(root_file, data['root_file'][idx0])
            pid = np.append(pid, data['pid'][idx0])
            position = np.append(position, data['position'][idx0])
            direction = np.append(direction, data['direction'][idx0])
            energy = np.append(energy, np.sum(data['energy'][indices]))
            labels = np.append(labels, nmu)

        offset_next += nonzero_events_in_file

        file_indices = nonzero_event_indexes[file_index]

        dset_IDX[offset:offset_next] = event_id[file_indices]
        dset_PATHS[offset:offset_next] = root_file[file_indices]
        dset_energies[offset:offset_next, :] = energy[file_indices].reshape(-1, 1)
        dset_positions[offset:offset_next, :, :] = position[file_indices].reshape(-1, 1, 3)
        dset_labels[offset:offset_next] = labels[file_indices]

        direction = direction[file_indices]
        polar = np.arccos(direction[:, 1])
        azimuth = np.arctan2(direction[:, 2], direction[:, 0])
        dset_angles[offset:offset_next, :] = np.hstack((polar.reshape(-1, 1), azimuth.reshape(-1, 1)))
        dset_event_data[offset:offset_next, :] = x_data

        offset = offset_next
        print("Offset: ", offset, " Offset_next: ", offset_next, " File_indices: ", file_indices, " dset_ID: ", dset_ID, " event_id: ", event_id)
        print("Finished file: {}".format(filename))

    print("Saving")
    h5_file.close()
    print("Finished")


# In[ ]:


GenerateMultiMuonSample_h5(avg_mu_per_ev=2.5, sigma_time_offset=21.2)