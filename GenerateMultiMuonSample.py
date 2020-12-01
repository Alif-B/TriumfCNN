import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import random
import os
import sys
import h5py
import cProfile
import re
import pstats
import io

datafile = np.load('event998.npz', allow_pickle=True)
geofile = np.load('mpmt_full_geo.npz', allow_pickle=True)
tubes = geofile['tube_no']
tube_xyz = geofile['position']
tube_x = tube_xyz[:, 0]
tube_y = tube_xyz[:, 1]
tube_z = tube_xyz[:, 2]
R = (tube_x.max() - tube_x.min()) / 2.0
H = tube_y.max() - tube_y.min()
print("R=", R, "H=", H)
print("min_x=", tube_x.min(), "max_x=", tube_x.max(), "diameter=", tube_x.max() - tube_x.min())
print("min_z=", tube_z.min(), "max_z=", tube_z.max(), "diameter=", tube_z.max() - tube_z.min())
print("min_y=", tube_y.min(), "max_y=", tube_y.max(), "height=", tube_y.max() - tube_y.min())

evno = 0

def GetEvent(event_no):
    dtubes = datafile['digi_hit_pmt'][event_no]
    dcharges = datafile['digi_hit_charge'][event_no]
    dtimes = datafile['digi_hit_time'][event_no]
    return (dtubes, dcharges, dtimes)

digitubes, digicharges, digitimes = GetEvent(evno)
number_of_events = len(datafile['digi_hit_pmt'])

def GetNoise_and_Signal_PMTLists(evnum, data):
    truehit_pmts = data['true_hit_pmt'][evnum]
    truehit_parent = data['true_hit_parent'][evnum]
    noisehit_pmts = truehit_pmts[truehit_parent == -1]
    signalhit_pmts = truehit_pmts[truehit_parent != -1]
    return (noisehit_pmts, signalhit_pmts)

def PMTChargeTime_in_list(evnum, pmtlist, data):
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

def SumEvents(event_numbers, time_offsets, datafile, only_noise=False):
    tube = np.array([], dtype=np.int32)
    charge = np.array([])
    time = np.array([])

    nev = len(event_numbers)
    if nev == 0:
        return (tube, charge, time)
    ievnum = event_numbers[0]
    if only_noise:
        noisepmts, signalpmts = GetNoise_and_Signal_PMTLists(ievnum, datafile)
        tube, charge, time = PMTChargeTime_in_list(ievnum, noisepmts, datafile)
    else:
        tube = datafile['digi_hit_pmt'][ievnum]
        charge = datafile['digi_hit_charge'][ievnum]
        time = datafile['digi_hit_time'][ievnum]
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
                tube = np.append(tube, pmt)
                charge = np.append(charge, curcharge[idx])
                time = np.append(time, curtime[idx] + toffset)
    return (tube, charge, time)

IMAGE_SHAPE = (40, 40, 38)
PMT_LABELS = "PMTlabelSheet3.csv"

def count_events(files):
    print("Counting Events")
    num_events = 0
    nonzero_file_events = []
    for file_index, f in enumerate(files):
        data = np.load(f, allow_pickle=True)
        nonzero_file_events.append([])
        hits = data['digi_hit_pmt']
        for i in range(len(hits) -2990):
            if len(hits[i]) != 0:
                nonzero_file_events[file_index].append(i)
                num_events += 1
    return num_events, nonzero_file_events

def GenMapping(csv_file):
    print("GenMapping")
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

def GenerateMultiMuonSample_h5(avg_mu_per_ev=2.5, sigma_time_offset=21.2):
    files = ['event998.npz']

    files = [x.strip() for x in files]
    if len(files) == 0:
        raise ValueError("No files provided!!")
    print("Merging " + str(len(files)) + " files")

    num_nonzero_events, nonzero_event_indexes = count_events(files)
    print(num_nonzero_events)
    num_muons = np.random.poisson(avg_mu_per_ev, num_nonzero_events)

    dtype_events = np.dtype(np.float32)
    dtype_labels = np.dtype(np.int32)
    dtype_energies = np.dtype(np.float32)
    dtype_positions = np.dtype(np.float32)
    dtype_IDX = np.dtype(np.int32)
    dtype_PATHS = h5py.special_dtype(vlen=str)
    dtype_angles = np.dtype(np.float32)
    h5_file = h5py.File('multimuonfile(2).h5', 'w')
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

    offset = 0
    offset_next = 0
    mPMT_to_index = GenMapping(PMT_LABELS)
    for file_index, filename in enumerate(files):
        data = np.load(filename, allow_pickle=True)
        nonzero_events_in_file = len(nonzero_event_indexes[file_index])
        x_data = np.zeros((nonzero_events_in_file,) + IMAGE_SHAPE,
                          dtype=dtype_events)
        digi_hit_pmt = data['digi_hit_pmt']
        delay = 0
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

            idx0 = indices[0]
            event_id = np.append(event_id, data['event_id'][idx0])
            root_file = np.append(root_file, data['root_file'][idx0])
            pid = np.append(pid, data['pid'][idx0])
            position = np.append(position, data['position'][idx0])
            print(position)
            direction = np.append(direction, data['direction'][idx0])
            energy = np.append(energy, np.sum(data['energy'][indices]))
            labels = np.append(labels, nmu)

        offset_next += nonzero_events_in_file

        file_indices = nonzero_event_indexes[file_index]
        print(file_indices)
        x = dset_IDX[offset:offset_next]
        y = event_id[file_indices]
        dset_IDX[offset:offset_next] = event_id[file_indices]
        dset_PATHS[offset:offset_next] = root_file[file_indices]
        dset_energies[offset:offset_next, :] = energy[file_indices].reshape(-1, 1)
        dset_positions[offset:offset_next, :, :] = position[file_indices].reshape(-1, 1, 1)
        dset_labels[offset:offset_next] = labels[file_indices]

        direction = direction[file_indices]
        print(direction)
        print(type(direction))
        polar = np.arccos(direction[:])
        azimuth = np.arctan2(direction[:], direction[:])
        dset_angles[offset:offset_next, :] = np.hstack((polar.reshape(-1, 1), azimuth.reshape(-1, 1)))
        dset_event_data[offset:offset_next, :] = x_data

        offset = offset_next
        print(offset)
        print("Finished file: {}".format(filename))

    print("Saving")
    print("Finished")

pr = cProfile.Profile()
pr.enable()
GenerateMultiMuonSample_h5(avg_mu_per_ev=2.5, sigma_time_offset=21.2)
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('test.txt', 'w+') as f:
    f.write(s.getvalue())
