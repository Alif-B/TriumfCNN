#!/usr/bin/env python
# coding: utf-8
# # EventDisplay -- to display mPMT events in new npz file format
# Edit to input the full geometry file, and npz data file that your are interested in.
# Authors: Blair Jamieson, Connor Boubard
# June 2020

# In[138]:
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import random

# In[2]:
datafile = np.load('event998.npz', allow_pickle=True)
geofile = np.load('mpmt_full_geo.npz', allow_pickle=True)

evno = 0


def GetEvent(event_no):
    dtubes = datafile['digi_hit_pmt'][event_no]
    dcharges = datafile['digi_hit_charge'][event_no]
    dtimes = datafile['digi_hit_time'][event_no]
    return (dtubes, dcharges, dtimes)


digitubes, digicharges, digitimes = GetEvent(evno)

# In[14]:


number_of_events = len(datafile['digi_hit_pmt'])


def EventDisplayHist(times, charges, title='Event Charge versus Time', cutrange=[[-1, -1], [-1, -1]]):
    fig = plt.figure(figsize=[12, 12])
    tmin = times.min()
    tmax = times.max()
    qmin = charges.min()
    qmax = charges.max()

    if cutrange[0][0] != cutrange[0][1]:
        tmin = cutrange[0][0]
        tmax = cutrange[0][1]
    if cutrange[1][0] != cutrange[1][1]:
        qmin = cutrange[1][0]
        qmax = cutrange[1][1]

    plt.hist2d(times, charges, [100, 100], [[tmin, tmax], [qmin, qmax]])
    fig.suptitle(title, fontsize=20)
    plt.xlabel('Time (ns)', fontsize=18)
    plt.ylabel('Charge (pe)', fontsize=16)
    # plt.set_cmap('gist_heat_r')
    plt.set_cmap('cubehelix_r')
    plt.colorbar()

EventDisplayHist(digitimes, digicharges)

EventDisplayHist(digitimes, digicharges, 'QT', [[940, 1040], [0, 20]])