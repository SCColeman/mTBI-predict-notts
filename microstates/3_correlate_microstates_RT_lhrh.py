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

RTs_lh = []
RTs_rh = []
resp_lh = []
resp_rh = []
# scanning session info
for subject in ['2001', '2003', '2008', '2009', '2014']:
    
    # each session
    sessions = os.listdir(op.join(deriv_root, "sub-" + subject))
    sessions = [sessions[i][4:] for i in range(len(sessions))]
    
    #if subject=='2014':
    #    sessions = sessions[1:]
    
    for session in sessions:
        
        task = 'CRT'  # name of the task
        run = '01'
        suffix = 'meg'
        
        bids_path = BIDSPath(subject=subject, session=session,
        task=task, run=run, suffix=suffix, root=bids_root)
        
        deriv_path = BIDSPath(subject=subject, session=session,
        task=task, run=run, suffix=suffix, root=deriv_root)
        
        RT_fname = deriv_path.basename + "-avg_RT_lh.npy"
        RTs_lh.append(np.load(op.join(deriv_path.directory, RT_fname)))
        RT_fname = deriv_path.basename + "-avg_RT_rh.npy"
        RTs_rh.append(np.load(op.join(deriv_path.directory, RT_fname)))
        
        resp_fname = deriv_path.basename + "-lh_stim.npy"
        resp_lh.append(np.load(op.join(deriv_path.directory, resp_fname)))
        resp_fname = deriv_path.basename + "-rh_stim.npy"
        resp_rh.append(np.load(op.join(deriv_path.directory, resp_fname)))
        
RTs_lh = np.array(RTs_lh)
RTs_rh = np.array(RTs_rh)

RTs_lh[16] = RTs_rh[16] # JUST TO MAKE THINGS WORK, DELETE THIS SESSION LATER

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

#%% get microstate probability timecourses

sfreq = 600

microstate_timecourses = np.zeros((k, len(idx)))
for state in range(k):
    microstate_timecourses[state,:] = idx==state
    
#%% separate by session

i = 0
microstate_mean_lh = np.zeros((k, len(info[:,0])))
microstate_mean_rh = np.zeros((k, len(info[:,0])))
for s, session in enumerate(info[:,0]):
    npoints = info[s,1].astype(int)
    sub_microstate = microstate_timecourses[:,i:i+npoints]
    i += npoints
    
    # make epochs object for subject
    microstate_names=["microstate_" + str(i) for i in range(k)]
    microstate_info = mne.create_info(ch_names=microstate_names, sfreq=sfreq, ch_types='misc')
    microstate_raw = mne.io.RawArray(sub_microstate, microstate_info)
    
    duration = 2 + 1/sfreq
    events = mne.make_fixed_length_events(microstate_raw, duration=duration)
    
    event_id = 1
    tmin, tmax = 0, 2
    microstate_epochs = mne.Epochs(
        microstate_raw,
        events,
        event_id,
        tmin,
        tmax,
        baseline=None,
        preload=True)
    window_lh = (RTs_lh[s]-0.1, RTs_lh[s]+0.1)
    window_lh_i = [(window_lh[0]+0.8)*sfreq, (window_lh[1]+0.8)*sfreq]
    window_rh = (RTs_lh[s]-0.1, RTs_lh[s]+0.1)
    window_rh_i = [(window_lh[0]+0.8)*sfreq, (window_lh[1]+0.8)*sfreq]
    microstate_evoked = microstate_epochs.average("all").get_data()
    microstate_mean_lh[:, s] = np.mean(microstate_evoked[:, int(window_lh_i[0]):int(window_lh_i[1])], 1)
    microstate_mean_rh[:, s] = np.mean(microstate_evoked[:, int(window_rh_i[0]):int(window_rh_i[1])], 1)

microstate_mean_lh = np.delete(microstate_mean_lh, 16, 1)
microstate_mean_rh = np.delete(microstate_mean_rh, 16, 1)
RTs_lh = np.delete(RTs_lh, 16)
RTs_rh = np.delete(RTs_rh, 16)

info = np.delete(info, 16, 0)

#%% correlate with RT

import statsmodels.api as sm
import pandas as pd
import seaborn as sns

def slr_plot(X, Y, subs, plot=True):
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
        df['group'] = subs
        sns.scatterplot(df, x='X', y='Y', hue='group')
        sns.regplot(df, x='X', y='Y', scatter=False, 
                    line_kws=dict(color="purple"))
        sns.set(style="darkgrid")
        ax.set(title='R^2 = ' + str("%.2f" % rsq) + ' , p = ' + str("%.2f" % p))
        plt.show()
    else:
        fig = []
        ax = []
    return rsq, p, fig, ax

subnums = info[:,0]
subnums = [subnums[i][:4] for i in range(len(subnums))]
for state in range(k):
    x = microstate_mean_lh[state,:]
    y = RTs_lh.copy()
    sns.set()
    rsq, p, fig, ax = slr_plot(x, y, subnums, True)
    ax.set(ylabel='LH Reaction Times (s)', xlabel='State Probability')
    plt.tight_layout()
    
    x = microstate_mean_rh[state,:]
    y = RTs_rh.copy()
    sns.set()
    rsq, p, fig, ax = slr_plot(x, y, subnums, True)
    ax.set(ylabel='RH Reaction Times (s)', xlabel='State Probability')
    plt.tight_layout()

