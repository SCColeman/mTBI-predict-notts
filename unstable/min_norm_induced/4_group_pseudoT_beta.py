# -*- coding: utf-8 -*-
"""
Plot group pseudo T activation map.

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os.path as op
import os

import numpy as np
import mne
from mne_bids import BIDSPath

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'
pseudoT_path = op.join(deriv_root, 'min_norm_induced')

# scanning session info
subjects = ['2001', '2003', '2008', '2009', '2014']
task = 'CRT'  # name of the task
run = '01'
suffix = 'meg'

# get template details
subjects_dir = r'R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS'
fs_subject = "fsaverage"
fname_fsaverage_src = op.join(subjects_dir, fs_subject, "bem", "fsaverage-ico-5-src.fif")
src_to = mne.read_source_spaces(fname_fsaverage_src)

#%% collect subject pseudoT

stc_left = []
stc_right = []

# load in STCs and morph
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
        stc_left_fname = deriv_path.basename + "-left_stim_beta"
        stc_right_fname = deriv_path.basename + "-right_stim_beta"
        
        # morph left to fsaverage
        stc = mne.read_source_estimate(op.join(pseudoT_path, stc_left_fname))
        morph = mne.compute_source_morph(
            stc,
            subject_from="sub-" + subject,
            subject_to="fsaverage",
            src_to=src_to,
            subjects_dir=subjects_dir,
        )
        stc_morphed = morph.apply(stc)
        stc_left.append(stc_morphed)
        
        # morph right to fsaverage
        stc = mne.read_source_estimate(op.join(pseudoT_path, stc_right_fname))
        morph = mne.compute_source_morph(
            stc,
            subject_from="sub-" + subject,
            subject_to="fsaverage",
            src_to=src_to,
            subjects_dir=subjects_dir,
        )
        stc_morphed = morph.apply(stc)
        stc_right.append(stc_morphed)

#%% plot one on fsaverage

brain = stc_right[-1].plot(subject="fsaverage", subjects_dir=subjects_dir)    

#%% average stc

for stcs in [stc_left, stc_right]:
    stc_avg = sum(stcs) / len(stcs)
    stc_avg.subject = 'fsaverage'     
    
    kwargs = dict(subject="fsaverage",
                    subjects_dir=subjects_dir,
                    surface="inflated",
                    views=["lat", "med"],
                    size=700,
                    hemi="split",
                    smoothing_steps=10,
                    time_viewer=False,
                    show_traces=False,
                    colorbar=True)
     
    stc_avg.plot(**kwargs)
           