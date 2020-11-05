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
import os
import sys


# In[2]:
datafile = np.load('event998.npz', allow_pickle=True)
geofile = np.load('mpmt_full_geo.npz', allow_pickle=True)
# # First let's explore the geometry file
# Make sure we can find the phototube locations, and build a mapping from the three dimensional locations of the PMTs.


# In[3]:
geofile.files

# In[4]:
tubes = geofile['tube_no']
tubes

# In[5]:
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

# In[6]:
tube_dir = geofile['orientation']

# In[7]:
fig = plt.figure(figsize=[15, 15])
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tube_x, tube_y, tube_z, marker='.')
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')
ax.view_init(elev=45.0, azim=45.0)
plt.show()

evno = 0

def GetEvent(event_no):
    dtubes = datafile['digi_hit_pmt'][event_no]
    dcharges = datafile['digi_hit_charge'][event_no]
    dtimes = datafile['digi_hit_time'][event_no]
    return (dtubes, dcharges, dtimes)


digitubes, digicharges, digitimes = GetEvent(evno)

# In[14]:


number_of_events = len(datafile['digi_hit_pmt'])
number_of_events

# In[15]:


digicharges

# In[16]:


digitubes

# In[17]:


len(digitubes)

# In[18]:


len(digicharges)


# In[19]:


def ChargeTimeHist(times, charges, title='Event Charge versus Time', cutrange=[[-1, -1], [-1, -1]]):
    """
    Makes a 2d histogram of charge versus time.
    inputs:
    times is an np.array of times of PMT hits
    charges is an np.array of charges of PMT hits
    title is the title of the histogram
    cutrange has two ranges, one in x and one in y [ [tmin, tmax], [qmin,qmax] ]
    """
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


# In[27]:


ChargeTimeHist(digitimes, digicharges)

# In[28]:


ChargeTimeHist(digitimes, digicharges, 'QT', [[940, 1040], [0, 20]])


# # Pick a random event to display

# In[36]:
