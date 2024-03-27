# -*- coding: utf-8 -*-
"""
Preprocess CRT data, taken on CTF MEG system. Can easily be automated. 

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os.path as op
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'

# scanning session info
subject = '2001'
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

# load data
data = read_raw_bids(bids_path=bids_path, verbose=False)
#data_raw = data.copy()  # these lines allow for easier post-hoc debugging
print(data.info)
sfreq = data.info["sfreq"]

#%% load triggers

events1 = mne.find_events(data, stim_channel="UPPT001")   # for button press
events2 = mne.find_events(data, stim_channel="UPPT002")   # for stims
events2[events2[:,2]==1,2] = 2
events = np.concatenate((events1, events2))
mne.viz.plot_events(events)

#%% third order gradiometer

data.apply_gradient_compensation(grade=3)
#data_3rd = data.copy()

#%% broadband filter

data.load_data().filter(l_freq=1, h_freq=45)
#data_bband = data.copy()

#%% get movement parameters and annotate high movement

# get hpi info
chpi_locs = mne.chpi.extract_chpi_locs_ctf(data, verbose=False)
head_pos = mne.chpi.compute_head_pos(data.info, chpi_locs, verbose=False)
head_movement_fig = mne.viz.plot_head_positions(head_pos, mode="traces")

# calculate gradient of head movement
head_pos_grad = head_pos.copy()
head_pos_grad[:,1:] = np.gradient(head_pos[:,1:], axis=0)
head_movement_fig = mne.viz.plot_head_positions(head_pos_grad, mode="traces")

# mark bad head movements greater than 2mm / s
bad_head_bool = head_pos_grad[:,1:4] > 0.001
bad_head_bool = np.sum(bad_head_bool, axis=1) > 0

# create annotation
hpi_sfreq = head_pos[1,0] - head_pos[0,0]
bad_head_duration = 5*hpi_sfreq
bad_head_onset = head_pos[bad_head_bool,0] - 2*hpi_sfreq
bad_head_description = "BAD_head"
bad_head_annot = mne.Annotations(bad_head_onset, bad_head_duration, 
                                 bad_head_description, 
                                 orig_time=data.info['meas_date'])

#%% mark SQUID resets

squid_annot, bad_chan = mne.preprocessing.annotate_amplitude(
                                    data, peak=dict(mag=2e-12), picks='meg',
                                    bad_percent=5, min_duration=0.005)
squid_annot.onset = squid_annot.onset - 2
squid_annot.duration = [4] * len(squid_annot.duration)
data.info["bads"].extend(bad_chan)

#%% annotate smaller muscle artefacts etc (DONT USE ZSCORE HERE)

muscle_annot, bad_chan = mne.preprocessing.annotate_amplitude(
                                    data, peak=dict(mag=5e-13), picks='meg',
                                    bad_percent=5, min_duration=1/600)
muscle_annot.onset = muscle_annot.onset - 0.2
muscle_annot.duration = muscle_annot.duration + 0.4

#%% add up annotations

data.set_annotations(bad_head_annot + squid_annot + muscle_annot)
#data_annot = data.copy()
data.plot()

#%% plot psd

data.plot_psd(fmax=45, picks='mag').show()

#%% Automated ICA removal

EOG_channels = ["UADC007-4123", "UADC010-4123"]
ECG_channels = ["UADC009-4123"]
blink_channels = ['MLT31-4123', 'MRT31-4123'];

ica = mne.preprocessing.ICA(n_components=30)
ica.fit(data.copy().pick("mag"), reject_by_annotation=True)

eog_indices, eog_scores = ica.find_bads_eog(data, EOG_channels[0],
                                            reject_by_annotation=True)
blink_indices, blink_scores = ica.find_bads_eog(data, blink_channels,
                                                reject_by_annotation=True)
ecg_indices, ecg_scores = ica.find_bads_ecg(data, ECG_channels[0],
                                            reject_by_annotation=True)

ica.exclude = list(dict.fromkeys(eog_indices + blink_indices + ecg_indices))

# plot diagnostics
ica.plot_properties(data, picks=ica.exclude)

ica.plot_sources(data)

# apply
ica.apply(data.pick("mag"))
#data_ica = data.copy()

#%% save out preprocessed data

preproc_fname = deriv_path.basename + "-raw.fif"
events_fname = deriv_path.basename + "-events.fif"
data.save(op.join(deriv_path.directory, preproc_fname), overwrite=True)
mne.write_events(op.join(deriv_path.directory, events_fname), events, overwrite=True)




