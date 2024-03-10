# -*- coding: utf-8 -*-
"""
Fix pos files to deal with dodgy old polhemus system.

@author: ppysc6
"""

import numpy as np
import os.path as op
import pandas as pd
from matplotlib import pyplot as plt

# set up path
pos_path = r"R:\DRS-mTBI\Seb\mTBI_predict\pos_files"
subject = "2001"
session = "03N"
pos_fname = op.join(pos_path, subject + "_" + session + ".pos")
pos_new_fname = op.join(pos_path, subject + "_" + session + "_fixed.pos")

# load pos file into np array
df = pd.read_table(pos_fname, names=['point','x','y','z'], delim_whitespace=True)
pos = df.drop(df.index[0]).to_numpy()
npoints = np.shape(pos)[0]

# chop out fiducial points
official_fids = pos[-3:, :]
fid_cluster1 = pos[0:15, 1:]
fid_cluster2 = pos[-18:-3, 1:]

# plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[:,1], pos[:,2], pos[:,3], alpha=0.1)
ax.scatter(fid_cluster1[:,0], fid_cluster1[:,1], fid_cluster1[:,2], c="red")
ax.scatter(fid_cluster2[:,0], fid_cluster2[:,1], fid_cluster2[:,2], c="green")

# create new official fids by averaging fiducial clusters
nas1 = np.mean(fid_cluster1[0:5,:], 0)
left1 = np.mean(fid_cluster1[5:10,:], 0)
right1 = np.mean(fid_cluster1[10:15,:], 0)

nas2 = np.mean(fid_cluster2[0:5,:], 0)
left2 = np.mean(fid_cluster2[5:10,:], 0)
right2 = np.mean(fid_cluster2[10:15,:], 0)

nas_mean = np.mean([nas1,nas2], 0)
left_mean = np.mean([left1,left2], 0)
right_mean = np.mean([right1,right2], 0)

# insert new fiducial positions into copy of pos
pos_new = pos.copy()
pos_new[-3, 1:] = nas1
pos_new[-2, 1:] = left1
pos_new[-1, 1:] = right1

# plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos_new[:,1], pos_new[:,2], pos_new[:,3], alpha=0.1)
ax.scatter(fid_cluster2[:,0], fid_cluster2[:,1], fid_cluster2[:,2], c="green")

# save
np.savetxt(pos_new_fname, pos_new, fmt='%s')

