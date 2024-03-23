# -*- coding: utf-8 -*-
"""
Calculate source-space MEG "microstates" based on dynamic connectivity.
Uses K-means clustering.

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os.path as op
import os

import numpy as np
import mne
from mne_bids import BIDSPath
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from sklearn.cluster import KMeans
from scipy.stats import zscore

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'

# scanning session info
subjects = ['2001', '2003', '2008', '2009', '2014']
task = 'CRT'  # name of the task
run = '01'
suffix = 'meg'

source_epochs = []
session_list = []

for s, subject in enumerate(subjects):
    
    # load data for each session
    sessions = os.listdir(op.join(deriv_root, "sub-" + subject))
    sessions = [sessions[i][4:] for i in range(len(sessions))]

    for s, session in enumerate(sessions):
        
        # bids paths
        bids_path = BIDSPath(subject=subject, session=session,
        task=task, run=run, suffix=suffix, root=bids_root)
        
        deriv_path = BIDSPath(subject=subject, session=session,
        task=task, run=run, suffix=suffix, root=deriv_root)
        
        # load in continuous source data
        data_fname = deriv_path.basename + "-source_epochs.fif"
        epochs = mne.read_epochs(op.join(deriv_path.directory, data_fname),
                             preload=True)
        source_epochs.append(epochs)
        session_info = [subject + "_" + session, len(epochs.times)*len(epochs)]
        session_list.append(session_info)
        del epochs               
    
#%% create source raw

source_raw = []
for source_epoch in source_epochs:
    df = source_epoch.to_data_frame()
    names = df.columns.to_list()[3:]
    raw_data = df.iloc[:,3:].to_numpy().transpose()
    raw_info = mne.create_info(ch_names=names, 
                               sfreq=source_epochs[0].info["sfreq"], 
                               ch_types='misc')
    source_raw.append(mne.io.RawArray(raw_data, raw_info))

del source_epochs

#%% filter 

source_filter = []
for raw_inst in source_raw:
    source_filter.append(raw_inst.copy().filter(1, 30, picks="all")) 
    
del source_raw
    
#%% z-score

source_zscore = []
for raw_inst in source_filter:
    data = raw_inst.get_data()
    data_zscore = zscore(data, axis=1)
    source_zscore.append(mne.io.RawArray(data_zscore, raw_info))
    del data, data_zscore
    
del source_filter
    
#%% take absolute

source_abs = []
for raw_inst in source_zscore:
    data = raw_inst.get_data()
    data_abs = np.abs(data)
    source_abs.append(mne.io.RawArray(data_abs, raw_info))
    del data, data_abs
    
del source_zscore

#%% convert to single data matrix

X_list = [source_abs[i].get_data().transpose() for i in range(len(session_list))]
del source_abs
    
#%% kmeans to calculate "microstates"

X_concat = np.concatenate(X_list, axis=0)
del X_list
k = 8
kmeans = KMeans(n_clusters=k).fit(X_concat)
centroids = kmeans.cluster_centers_
idx = kmeans.fit_predict(X_concat)

centroids_fname = op.join(deriv_root, "microstates_group", 
                          "group_" + str(k) + "k_centroids")
np.save(centroids_fname + ".npy", centroids)

idx_fname = op.join(deriv_root, "microstates_group", 
                          "group_" + str(k) + "k_idx")
np.save(idx_fname + ".npy", idx)

info_fname = op.join(deriv_root, "microstates_group", 
                          "group_" + str(k) + "k_session_info")
np.save(info_fname + ".npy", session_list)
