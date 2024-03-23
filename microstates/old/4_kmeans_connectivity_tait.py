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
import scipy

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'

# scanning session info
subject = '2001'
task = 'CRT'  # name of the task
run = '01'
suffix = 'meg'

# check session numbers
sessions = os.listdir(op.join(deriv_root, "sub-" + subject))
sessions = [sessions[i][4:] for i in range(len(sessions))]

#%% load data for each session

source_epochs = []
for s, session in enumerate(sessions):
    
    # bids paths
    bids_path = BIDSPath(subject=subject, session=session,
    task=task, run=run, suffix=suffix, root=bids_root)
    
    deriv_path = BIDSPath(subject=subject, session=session,
    task=task, run=run, suffix=suffix, root=deriv_root)
    
    # load in continuous source data
    data_fname = deriv_path.basename + "-source_epochs.fif"
    source_epochs.append(mne.read_epochs(op.join(deriv_path.directory, data_fname),
                         preload=True))
    
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

#%% filter 

source_filter = []
for raw_inst in source_raw:
    source_filter.append(raw_inst.copy().filter(1, 30, picks="all")) 
    
#%% calculate global field power and find peaks

all_peaks = []
X_list = []
for session in range(len(sessions)):
    X = source_filter[session].get_data().transpose()
    N = np.shape(X)[1]
    X_zscore = zscore(X, axis=0)
    GFP = np.sqrt( (1/(N-1)) * np.sum(X_zscore**2, axis=1))
    peaks = scipy.signal.find_peaks(GFP)
    all_peaks.append(peaks[0])
    X_peaks = np.abs(X_zscore[peaks[0]])
    X_list.append(X_peaks)   
    
#%% kmeans to calculate "microstates"

X_concat = np.concatenate(X_list, axis=0)
k = 8
kmeans = KMeans(n_clusters=k).fit(X_concat)
centroids = kmeans.cluster_centers_
idx = kmeans.fit_predict(X_concat)

centroids_fname = op.join(deriv_root, "kmeans_networks_all_sessions", 
                          subject + "_" + str(k) + "k_centroids")
np.save(centroids_fname + ".npy", centroids)

idx_fname = op.join(deriv_root, "kmeans_networks_all_sessions", 
                          subject + "_" + str(k) + "k_idx")
np.save(idx_fname + ".npy", idx)

#%% plot microstate centroids (spatial maps)

fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
parc = "aparc"
labels = mne.read_labels_from_annot("fsaverage", parc=parc, subjects_dir=subjects_dir)
labels = labels[:-1]

for state in range(k):
    stc = mne.labels_to_stc(labels, centroids[state,:].transpose())
    
    stc.plot(
        clim=dict(kind="percent", lims=[75, 85, 95]),
        colormap="Reds",
        subjects_dir=subjects_dir,
        views=["lat", "med"],
        size=600,
        hemi="split",
        smoothing_steps=10,
        time_viewer=False,
        show_traces=False,
        colorbar=True,
    )


#%% create epochs object out of microstate index timecourses

sfreq = source_epochs[0].info["sfreq"]

cluster_timecourses = np.zeros((k, len(idx)))
for state in range(k):
    cluster_timecourses[state,:] = idx==state
    
cluster_names=["cluster_" + str(i) for i in range(k)]
cluster_info = mne.create_info(ch_names=cluster_names, sfreq=sfreq, ch_types='misc')
cluster_raw = mne.io.RawArray(cluster_timecourses, cluster_info)

duration = 2 + 1/sfreq
events = mne.make_fixed_length_events(cluster_raw, duration=duration)

event_id = 1
tmin, tmax = 0, 2
cluster_epochs = mne.Epochs(
    cluster_raw,
    events,
    event_id,
    tmin,
    tmax,
    baseline=None,
    preload=True)

#%% plot timecourse of each cluster

cluster_evoked = cluster_epochs.average("all")

for state in range(k):
    cluster_evoked.plot("cluster_" + str(state))

