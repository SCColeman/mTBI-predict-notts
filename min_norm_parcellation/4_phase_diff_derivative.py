# -*- coding: utf-8 -*-
"""
Extract continuous source data for CRT task, taken with CTF MEG system. 
Uses minimum norm inverse solution.

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os.path as op
import os
import numpy as np
import mne
from mne_bids import BIDSPath
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, apply_inverse_cov
from scipy.stats import zscore, rayleigh
from mne_connectivity import symmetric_orth

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'
out_path = op.join(deriv_root, 'HMM', 'data')
subjects = ['2001', '2003', '2008', '2009', '2014']

# scanning session info
#for s, subject in enumerate(subjects):
subject = '2014'
# load data for each session
#sessions = os.listdir(op.join(deriv_root, "sub-" + subject))
#sessions = [sessions[i][4:] for i in range(len(sessions))]

#for s, session in enumerate(sessions):
session = '03N'
task = 'CRT'  # name of the task
run = '01'
suffix = 'meg'

bids_path = BIDSPath(subject=subject, session=session,
task=task, run=run, suffix=suffix, root=bids_root)

deriv_path = BIDSPath(subject=subject, session=session,
task=task, run=run, suffix=suffix, root=deriv_root)
    
#%% load in data

source_raw = mne.io.Raw(op.join(deriv_path.directory, deriv_path.basename + "-source_raw.fif"), 
                        preload=True)

#%% orthogonalise

stim_data = source_raw.copy().pick('stim').get_data()
source_data = source_raw.copy().pick('misc').get_data()
source_ortho = symmetric_orth(source_data)
source_ortho_raw = mne.io.RawArray(np.concatenate((source_ortho, stim_data), 0), source_raw.info)

#%% calculate instantaneous unwrapped phase

source_hilb = source_ortho_raw.copy().filter(4, 8, picks="misc").apply_hilbert("misc")
source_phase = mne.io.RawArray(np.unwrap(np.angle(source_hilb.get_data()), axis=1), source_raw.info)

#%% get data and take PDD with seed region

names = source_phase.pick('misc').ch_names
seed_reg = 0  # somatomotor

phase_data = source_phase.get_data()
seed = phase_data[seed_reg,:]
PDD = np.zeros(np.shape(phase_data))

for region in range(len(names)):
    
    phase_diff = np.diff(np.abs(phase_data[region,:] - seed), prepend=0)
    PDD[region,:] = np.exp(-np.abs(phase_diff))

#%% make object out of PDD

source_PDD = mne.io.RawArray(np.concatenate((PDD, stim_data), 0), source_raw.info)

events = mne.find_events(source_PDD, stim_channel=["stim_right"])
event_id = 1
tmin, tmax = -0.8, 1.2
epochs_PDD = mne.Epochs(
    source_PDD,
    events,
    event_id,
    tmin,
    tmax,
    baseline=None,
    preload=True,
    reject=None,
    reject_by_annotation=False)

evoked_PDD = epochs_PDD.average("misc")
evoked_PDD.plot()

#%% plot coherence around button press

win = (0.2, 0.3)
coh = np.mean(evoked_PDD.crop(win[0], win[1]).get_data(),1)

# plot
subjects_dir = r'R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS'
parc = "HCPMMP1_combined"
labels = mne.read_labels_from_annot("fsaverage", parc=parc, subjects_dir=subjects_dir)
labels = labels[2:]
coh_stc = mne.labels_to_stc(labels, coh)

clim = dict(kind="percent", lims=[30, 50, 100])
coh_stc.plot(subjects_dir=subjects_dir, subject="fsaverage", hemi="both", clim=clim)