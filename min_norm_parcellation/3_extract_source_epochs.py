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
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

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
    
#%% load in data

data = mne.io.Raw(op.join(deriv_path.directory, deriv_path.basename + "-raw.fif"))
data.pick("mag")
events = mne.read_events(op.join(deriv_path.directory, deriv_path.basename + "-events.fif"))
src = op.join(deriv_path.directory, deriv_path.basename + "-src.fif")
bem = op.join(deriv_path.directory, deriv_path.basename + "-bem.fif")
trans = op.join(deriv_path.directory, deriv_path.basename + "-trans.fif")
sfreq = data.info["sfreq"]

#%% compute covariance

cov = mne.compute_raw_covariance(data)

#%% make forward model from files

src = mne.read_source_spaces(src)
fwd = mne.make_forward_solution(data.info, trans, src, bem, verbose=True)
del src
inv = make_inverse_operator(data.info, fwd, cov)
del fwd

#%% epoch based on trigger

event_id = [1, 32]   # trigger of interest, [1 31] -> btn press, [101, 102] -> stim
tmin, tmax = -0.5, 1
epochs = mne.Epochs(
    data,
    events,
    event_id,
    tmin,
    tmax,
    baseline=None,
    preload=True,
    reject=dict(mag=4e-12),
    reject_by_annotation=True)

#%% parcellation beamformer

# get labels from parcellation
subjects_dir = r'R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS'
fs_subject = 'sub-' + subject
parc = "aparc.DKTatlas"
labels = mne.read_labels_from_annot(fs_subject, parc=parc, subjects_dir=subjects_dir)

stcs = apply_inverse_epochs(
    epochs, inv, lambda2=1.0 / 9.0, pick_ori="normal", return_generator=True,
    method="MNE"
)

label_ts = mne.extract_label_time_course(
    stcs, labels, inv["src"], return_generator=False
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