# -*- coding: utf-8 -*-
"""
Extract epoched source data for CRT task, taken with CTF MEG system. 
Uses minimum norm inverse solution.

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os.path as op

import numpy as np
import mne
from mne_bids import BIDSPath
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, apply_inverse_cov
from mne.beamformer import make_lcmv, apply_lcmv_epochs

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'

# scanning session info
subject = '2001'
session = '05N'
task = 'CRT'  # name of the task
run = '01'
suffix = 'meg'

bids_path = BIDSPath(subject=subject, session=session,
task=task, run=run, suffix=suffix, root=bids_root)

deriv_path = BIDSPath(subject=subject, session=session,
task=task, run=run, suffix=suffix, root=deriv_root)
    
#%% load in data

data = mne.io.Raw(op.join(deriv_path.directory, deriv_path.basename + "-raw.fif"))
noise = mne.io.Raw(op.join(deriv_path.directory, deriv_path.basename + "-noise.fif"))
data.pick("mag")
noise.pick("mag")
events = mne.read_events(op.join(deriv_path.directory, deriv_path.basename + "-events.fif"))
src = op.join(deriv_path.directory, deriv_path.basename + "-src.fif")
bem = op.join(deriv_path.directory, deriv_path.basename + "-bem.fif")
trans = op.join(deriv_path.directory, deriv_path.basename + "-trans.fif")
sfreq = data.info["sfreq"]

#%% calculate noise covariance

noise_cov = mne.compute_raw_covariance(noise)
noise_cov.plot(data.info)

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

#%% make forward model and inverse from files

cov = mne.compute_covariance(epochs)
src = mne.read_source_spaces(src)
fwd = mne.make_forward_solution(data.info, trans, src, bem, verbose=True)
filters = make_lcmv(
    epochs.info,
    fwd,
    data_cov=cov,
    reg=0.05, # regularisation, 5% is fairly standard
    noise_cov=noise_cov,
    pick_ori="max-power",
    weight_norm="nai",
    rank=dict({'mag': 264}))

#%% test beamformer by plotting pseudo T

subjects_dir = r'R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS'
fs_subject = 'sub-' + subject

# filter epochs
fband = [8, 13]   # alpha
epochs_filt = epochs.copy().filter(fband[0], fband[1])

# compute active and control covariance of filtered data        
act_min, act_max = 0, 0.2
con_min, con_max = 1, 1.2

active_cov = mne.compute_covariance(epochs_filt, tmin=act_min, 
                                    tmax=act_max, method="shrunk")
control_cov= mne.compute_covariance(epochs_filt, tmin=con_min, 
                                    tmax=con_max, method="shrunk")

stc_active = mne.beamformer.apply_lcmv_cov(active_cov, filters)
stc_control = mne.beamformer.apply_lcmv_cov(control_cov, filters)
pseudoT = (stc_active - stc_control) / (stc_active + stc_control)

# plot
pseudoT.plot(src=src, subject=fs_subject,
            subjects_dir=subjects_dir,
            surface="inflated",
            views=["lat", "med"],
            size=600,
            hemi="split",
            smoothing_steps=10,
            time_viewer=False,
            show_traces=False,
            colorbar=True)

#%% parcellation beamformer

# get labels from parcellation
subjects_dir = r'R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS'
fs_subject = 'sub-' + subject
parc = "HCPMMP1_combined"
labels = mne.read_labels_from_annot("fsaverage", parc=parc, subjects_dir=subjects_dir)
labels = labels[2:]
labels = mne.morph_labels(labels, fs_subject, "fsaverage", subjects_dir)

stcs = apply_lcmv_epochs(
    epochs, filters, return_generator=True)

label_ts = mne.extract_label_time_course(
    stcs, labels, src, return_generator=False
)
del stcs

#%% create source epochs object
 
n_epochs = (len(epochs))
epoch_len = np.shape(epochs[0])[2]
source_epochs_data = np.zeros((len(epochs), len(labels), np.shape(epochs[0])[2]))
for s, stc_epoch in enumerate(label_ts):
    source_epochs_data[s,:,:] = stc_epoch

#fake epoch object
ch_names=[labels[i].name for i in range(len(labels))]
epochs_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='misc')
source_epochs = mne.EpochsArray(source_epochs_data, epochs_info)

source_epochs.plot("all")

#%% save source epochs

source_epochs_fname = deriv_path.basename + "-source_epochs.fif"
source_epochs.save(op.join(deriv_path.directory, source_epochs_fname), overwrite=True)
