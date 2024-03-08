# -*- coding: utf-8 -*-
"""
Preprocess CRT data, taken on CTF MEG system. Can easily be automated. 

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os
import os.path as op

import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids, inspect_dataset
from matplotlib import pyplot as plt

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-PSR\Seb\mTBI_testing\derivatives'

# scanning session info
subject = '2011'
session = '03N'
task = 'CRT'  # name of the task
run = '01'
suffix = 'meg'

bids_path = BIDSPath(subject=subject, session=session,
task=task, run=run, suffix=suffix, root=bids_root)

deriv_path = BIDSPath(subject=subject, session=session,
task=task, run=run, suffix=suffix, root=deriv_root)

if not op.exists(deriv_path.directory):
    deriv_path.mkdir()
    
#%% load dataset

data = read_raw_bids(bids_path=bids_path, verbose=False)
data_raw = data.copy()  # these lines allow for easier post-hoc debugging
print(data.info)

#%% load triggers

events1 = mne.find_events(data, stim_channel="UPPT001")   # for button press
events2 = mne.find_events(data, stim_channel="UPPT002")   # for stims
events2[events2[:,2]==1,2] = 2
events = np.concatenate((events1, events2))
mne.viz.plot_events(events)

#%% third order gradiometer

data.apply_gradient_compensation(grade=3)
data_3rd = data.copy()

#%% downsample

orig_freq = 600
sfreq = 250
data.resample(sfreq=sfreq)
data_ds = data.copy()

### downsample events to match
events[:,0] = np.round(events[:,0] * (sfreq/orig_freq))

#%% broadband filter

data.filter(l_freq=1, h_freq=45)
data_bband = data.copy()

#%% plot psd

data.plot_psd(fmax=45, picks='mag').show()

#%% Automated ICA removal

EOG_channels = ["UADC007-4123", "UADC010-4123"]
ECG_channels = ["UADC009-4123"]
blink_channels = ['MLT31-4123', 'MRT31-4123'];

ica = mne.preprocessing.ICA(n_components=30)
ica.fit(data.copy().pick_types(meg=True))

eog_indices, eog_scores = ica.find_bads_eog(data, EOG_channels[0])
blink_indices, blink_scores = ica.find_bads_eog(data, blink_channels)
ecg_indices, ecg_scores = ica.find_bads_ecg(data, ECG_channels[0])

ica.exclude = eog_indices + blink_indices + ecg_indices

# plot diagnostics
ica.plot_properties(data, picks=ica.exclude)

# apply
ica.apply(data)
data_ica = data.copy()

#%% save out preprocessed data

preproc_fname = deriv_path.basename + "-raw.fif"
events_fname = deriv_path.basename + "-events.fif"
data.save(op.join(deriv_path.directory, preproc_fname), overwrite=True)
mne.write_events(op.join(deriv_path.directory, events_fname), events, overwrite=True)



