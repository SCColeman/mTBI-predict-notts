# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:34:00 2024

@author: ppysc6
"""

import os.path as op
import os
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids


#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'

# scanning session info
for subject in ["2001", "2003", "2008", "2009", "2014"]:
    
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
        
        if not op.exists(deriv_path.directory):
            deriv_path.mkdir()
            
        #%% load dataset and empty room noise
        
        # load data
        data = read_raw_bids(bids_path=bids_path, verbose=False)
        print(data.info)
        sfreq = data.info["sfreq"]
        
        #%% load triggers
        
        events1 = mne.find_events(data, stim_channel="UPPT001")   # for button press
        events2 = mne.find_events(data, stim_channel="UPPT002")   # for stims
        events2[events2[:,2]==1,2] = 2
        events = np.concatenate((events1, events2))
        
        #%% get relevant triggers
        
        # 1 (left) and 32 (right) are button presses, 102 (left) and 101 (right) 
        btn_left = events[events[:,2]==32, 0]
        btn_right = events[events[:,2]==1, 0]
        stim_left = events[events[:,2]==102, 0]
        stim_right = events[events[:,2]==101, 0]
        
        accept = 1*sfreq
        all_RTs = []
        for stim in stim_left:
            RTs = btn_left - stim
            correct = np.logical_and(RTs > 0, RTs < accept)
            if np.any(correct):
                all_RTs.append(np.min(RTs[correct]).astype(float))
        for stim in stim_right:
            RTs = btn_right - stim
            correct = np.logical_and(RTs > 0, RTs < accept)
            if np.any(correct):
                all_RTs.append(np.min(RTs[correct]).astype(float))
                
        avg_RT = np.mean(np.array(all_RTs)/sfreq)
        
        RT_fname = deriv_path.basename + "-avg_RT.npy"
        np.save(op.join(deriv_path.directory, RT_fname), avg_RT)
        print(avg_RT)
        
        #%% only left side 
        
        RTs_left = []
        for stim in stim_left:
            RTs = btn_left - stim
            correct = np.logical_and(RTs > 0, RTs < accept)
            if np.any(correct):
                RTs_left.append(np.min(RTs[correct]).astype(float))
                
        avg_RT_lh = np.mean(np.array(RTs_left)/sfreq)
        
        RT_fname = deriv_path.basename + "-avg_RT_lh.npy"
        np.save(op.join(deriv_path.directory, RT_fname), avg_RT_lh)
        print(avg_RT_lh)
        
        #%% only right side 
        
        RTs_right = []
        for stim in stim_right:
            RTs = btn_right - stim
            correct = np.logical_and(RTs > 0, RTs < accept)
            if np.any(correct):
                RTs_right.append(np.min(RTs[correct]).astype(float))
                
        avg_RT_rh = np.mean(np.array(RTs_right)/sfreq)
        
        RT_fname = deriv_path.basename + "-avg_RT_rh.npy"
        np.save(op.join(deriv_path.directory, RT_fname), avg_RT_rh)
        print(avg_RT_rh)
