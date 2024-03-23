# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:27:22 2024

@author: ppysc6
"""

import os.path as op
import os
import numpy as np
import mne
from matplotlib import pyplot as plt
from mne_bids import BIDSPath, read_raw_bids

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'

#%% load reaction times

RTs = []
# scanning session info
for subject in ['2001', '2003', '2008', '2009', '2014']:
    
    # each session
    sessions = os.listdir(op.join(deriv_root, "sub-" + subject))
    sessions = [sessions[i][4:] for i in range(len(sessions))]
    
    for session in sessions:
        
        task = 'CRT'  # name of the task
        run = '01'
        suffix = 'meg'
        
        bids_path = BIDSPath(subject=subject, session=session,
        task=task, run=run, suffix=suffix, root=bids_root)
        
        deriv_path = BIDSPath(subject=subject, session=session,
        task=task, run=run, suffix=suffix, root=deriv_root)
        
        RT_fname = deriv_path.basename + "-avg_RT.npy"
        RTs.append(np.load(op.join(deriv_path.directory, RT_fname)))
        
RTs = np.array(RTs)

#%% load kmeans outputs

k = 8
centroids_fname = op.join(deriv_root, "microstates_group", 
                          "group_" + str(k) + "k_centroids.npy")
idx_fname = op.join(deriv_root, "microstates_group", 
                          "group_" + str(k) + "k_idx.npy")
info_fname = op.join(deriv_root, "microstates_group", 
                          "group_" + str(k) + "k_session_info.npy")
centroids = np.load(centroids_fname)
idx = np.load(idx_fname)
info = np.load(info_fname)

#%% get cluster probability timecourses

sfreq = 600

cluster_timecourses = np.zeros((k, len(idx)))
for state in range(k):
    cluster_timecourses[state,:] = idx==state
    
#%% separate by session

i = 0
cluster_mean = np.zeros((k, len(info[:,0])))
for s, session in enumerate(info[:,0]):
    npoints = info[s,1].astype(int)
    sub_cluster = cluster_timecourses[:,i:i+npoints]
    i += npoints
    
    # make epochs object for subject
    cluster_names=["cluster_" + str(i) for i in range(k)]
    cluster_info = mne.create_info(ch_names=cluster_names, sfreq=sfreq, ch_types='misc')
    cluster_raw = mne.io.RawArray(sub_cluster, cluster_info)
    
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
    window = (RTs[s]-0.1, RTs[s]+0.1)
    window_i = [(window[0]+0.8)*sfreq, (window[1]+0.8)*sfreq]
    cluster_evoked = cluster_epochs.average("all").get_data()
    cluster_mean[:, s] = np.mean(cluster_evoked[:, int(window_i[0]):int(window_i[1])], 1)


#%% correlate with RT

import statsmodels.api as sm
import pandas as pd
import seaborn as sns

def slr_plot(X, Y, plot=True):
    results = sm.OLS(Y, sm.add_constant(X)).fit()
    c, m = results.params
    y_model = (m * X) + c
    p = results.f_pvalue
    rsq = results.rsquared
    
    if plot:
        fig, ax = plt.subplots(figsize = ( 4 , 3.5 ))
        df = pd.DataFrame()
        df['X'] = X
        df['Y'] = Y
        df['y_model'] = y_model
        sns.scatterplot(df, x='X', y='Y')
        sns.lineplot(df, x='X', y='y_model', color='r')
        ax.set(title='R^2 = ' + str("%.2f" % rsq) + ' , p = ' + str("%.2f" % p))
        plt.show()
    else:
        fig = []
        ax = []
    return rsq, p, fig, ax

for state in range(k):
    x = cluster_mean[state,:]
    y = RTs.copy()
    sns.set()
    rsq, p, fig, ax = slr_plot(x, y, True)
    ax.set(ylabel='Reaction Times (s)', xlabel='State Probability')
    plt.tight_layout()

