# -*- coding: utf-8 -*-
"""
Create LCMV beamformer for all subjects and save out.
Uses outputs from 1_forward_model and 2_preprocessing.

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os.path as op
import os
import mne
from mne_bids import BIDSPath

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'

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
            
        #%% load in data
        
        data = mne.io.Raw(op.join(deriv_path.directory, deriv_path.basename + "-raw.fif"))
        data.pick("mag")
        events = mne.read_events(op.join(deriv_path.directory, deriv_path.basename + "-events.fif"))
        src = op.join(deriv_path.directory, deriv_path.basename + "-src.fif")
        bem = op.join(deriv_path.directory, deriv_path.basename + "-bem.fif")
        trans = op.join(deriv_path.directory, deriv_path.basename + "-trans.fif")
        sfreq = data.info["sfreq"]
        
        #%% epoch based on trigger
        
        event_id = [101, 102]  # trigger of interest
        tmin, tmax = -0.8, 1.2
        epochs = mne.Epochs(
            data,
            events,
            event_id,
            tmin,
            tmax,
            baseline=(-0.4, -0.1),
            preload=True,
            reject=dict(mag=4e-12),
            reject_by_annotation=True)
        
        #%% compute covariance of all data, unfiltered
        
        cov = mne.compute_covariance(epochs)
        
        #%% make forward and inverse solution 
        
        src = mne.read_source_spaces(src)
        fwd = mne.make_forward_solution(data.info, trans, src, bem, verbose=True)
        filters = mne.beamformer.make_lcmv(
            data.info,
            fwd,
            cov,
            reg=0.05,
            noise_cov=None,
            pick_ori="max-power",
            weight_norm="unit-noise-gain",
            rank=None)
        
        filters_fname = deriv_path.basename + "-filters.fif"
        filters.save(op.join(deriv_path.directory, filters_fname), overwrite=True)