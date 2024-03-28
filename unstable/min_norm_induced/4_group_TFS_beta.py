# -*- coding: utf-8 -*-
"""
Get TFS at peak beta location.

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
source_path = op.join(deriv_root, 'min_norm_induced')

# scanning session info
subjects = ['2001', '2003', '2008', '2009', '2014']
task = 'CRT'  # name of the task
run = '01'
suffix = 'meg'

#%% collect subject peak source epochs

TFR_left = []
TFR_right = []

# load in source epochs
for s, subject in enumerate(subjects):
    
    # load data for each session
    sessions = os.listdir(op.join(deriv_root, "sub-" + subject))
    sessions = [sessions[i][4:] for i in range(len(sessions))]

    for s, session in enumerate(sessions):
        
        deriv_path = BIDSPath(subject=subject, session=session,
        task=task, run=run, suffix=suffix, root=deriv_root)
        
        # load in continuous source data
        source_left_fname = deriv_path.basename + "-lh_beta_source.fif"
        source_right_fname = deriv_path.basename + "-rh_beta_source.fif"
        
        epochs_left = mne.read_epochs(op.join(source_path, source_left_fname))
        epochs_right = mne.read_epochs(op.join(source_path, source_right_fname))
        
        # create TFR for each
        baseline = (-0.5, -0.2)
        freqs = np.arange(2,35)
        n_cycles = freqs/2
        power_left = mne.time_frequency.tfr_morlet(epochs_left, freqs=freqs, n_cycles=n_cycles,
                                                   use_fft=True, picks="all"
                                                   )
        power_left[0].plot(picks="all", baseline=baseline)