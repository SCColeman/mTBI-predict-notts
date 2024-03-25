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
import pandas as pd

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'

#%% load microstates
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
microstate_epochs = []
for s, session in enumerate(info[:,0]):
    npoints = info[s,1].astype(int)
    sub_microstate = microstate_timecourses[:,i:i+npoints]
    i += npoints
    
    # make epochs object for subject
    names=["microstate_" + str(i) for i in range(k)]
    microstate_info = mne.create_info(ch_names=names, sfreq=sfreq, ch_types='misc')
    microstate_raw = mne.io.RawArray(sub_microstate, microstate_info)
    
    duration = 2 + 1/sfreq
    events = mne.make_fixed_length_events(microstate_raw, duration=duration)
    
    event_id = 1
    tmin, tmax = 0, 2
    microstate_epochs.append(mne.Epochs(
        microstate_raw,
        events,
        event_id,
        tmin,
        tmax,
        baseline=None,
        preload=True))

#%% get trial-wise average of microstates during time window

win = (int(1*sfreq), int(1.3*sfreq))

state_means = []

for i, inst in enumerate(microstate_epochs):
    
    sub_state_means = []
    for e, epoch in enumerate(inst):
        
        epoch_avg = np.mean(epoch[:,win[0]:win[1]], axis=1)
        sub_state_means.append(epoch_avg)
        
    state_means.append(sub_state_means)
    
#%% get source epochs

# scanning session info
subjects = ['2001', '2003', '2008', '2009', '2014']
task = 'CRT'  # name of the task
run = '01'
suffix = 'meg'

source_epochs = []

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
        del epochs               
    
#%% get trial-wise reaction times

RTs_lh = []
RTs_rh = []
for sub_epochs in source_epochs:

    sub_epochs.pick("stim")
    sub_RTs_lh = []
    sub_RTs_rh = []
    for e in range(len(sub_epochs)):
        
        epoch = sub_epochs[e]
        # find whether left or right stim
        lh_check = np.sum(epoch.copy().pick(["stim_left"]).get_data(copy=False))
        rh_check = np.sum(epoch.copy().pick(["stim_right"]).get_data(copy=False))
        
        if lh_check:
            stim_data = epoch.copy().pick(["stim_left"]).get_data(copy=False)[0,0,:]
            resp_data = epoch.copy().pick(["resp_left"]).get_data(copy=False)[0,0,:]
            if np.sum(resp_data) > 0:
                stim_i = np.argwhere(stim_data)[0][0]
                resp_i = np.argwhere(resp_data)[0][0]
                if resp_i > stim_i:
                    sub_RTs_lh.append((resp_i-stim_i)/sfreq)
                    sub_RTs_rh.append(0)
                else:
                    sub_RTs_lh.append(0)
                    sub_RTs_rh.append(0)
            else:
                sub_RTs_lh.append(0)
                sub_RTs_rh.append(0)
                
        if rh_check:
            stim_data = epoch.copy().pick(["stim_right"]).get_data(copy=False)[0,0,:]
            resp_data = epoch.copy().pick(["resp_right"]).get_data(copy=False)[0,0,:]
            if np.sum(resp_data) > 0:
                stim_i = np.argwhere(stim_data)[0][0]
                resp_i = np.argwhere(resp_data)[0][0]
                if resp_i > stim_i:
                    sub_RTs_rh.append((resp_i-stim_i)/sfreq)
                    sub_RTs_lh.append(0)
                else:
                    sub_RTs_rh.append(0)
                    sub_RTs_lh.append(0)
            else:
                sub_RTs_rh.append(0)
                sub_RTs_lh.append(0)
                
    RTs_lh.append(sub_RTs_lh)
    RTs_rh.append(sub_RTs_rh)
    
#%% make left hand and right hand dataframe

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
        sns.regplot(df, x='X', y='Y', line_kws=dict(color="purple"))
        sns.set(style="darkgrid")
        ax.set(title='R^2 = ' + str("%.2f" % rsq) + ' , p = ' + str("%.2f" % p))
        plt.show()
    else:
        fig = []
        ax = []
    return rsq, p, fig, ax

lh_state = 5
rh_state = 3

for ses in range(len(state_means)):
    
    signal_lh = np.array(state_means[ses])[:,lh_state]
    signal_rh = np.array(state_means[ses])[:,rh_state]
    RTs_lh_ses = np.array(RTs_lh[ses])
    RTs_rh_ses = np.array(RTs_rh[ses])
    
    lh_keep = RTs_lh_ses > 0
    rh_keep = RTs_rh_ses > 0
    
    lh_data = np.array([[ses]*len(signal_lh[lh_keep]), signal_lh[lh_keep], RTs_lh_ses[lh_keep]]).transpose()
    rh_data = np.array([[ses]*len(signal_rh[rh_keep]), signal_rh[rh_keep], RTs_rh_ses[rh_keep]]).transpose()
    
    df_ses_lh = pd.DataFrame(data=lh_data, columns=["ses", "signal", "RT"])
    df_ses_rh = pd.DataFrame(data=rh_data, columns=["ses", "signal", "RT"])
    
    rsq, p, fig, ax = slr_plot(rh_data[:,1], rh_data[:,2], True)
    sns.set(style="darkgrid")
    
    if ses==0:
        df_left = df_ses_lh.copy()
        df_right = df_ses_rh.copy()
    else:
        df_left = pd.concat([df_left, df_ses_lh], ignore_index=True)
        df_right = pd.concat([df_right, df_ses_rh], ignore_index=True)


x = df_left.iloc[:,1].to_numpy()
y = df_left.iloc[:,2].to_numpy()
rsq, p, fig, ax = slr_plot(x, y, True)

#%% stats    

import statsmodels.api as sm
import statsmodels.formula.api as smf

md = smf.mixedlm("RT ~ signal", df_right, groups=df_right["ses"])

mdf = md.fit()

print(mdf.summary())   