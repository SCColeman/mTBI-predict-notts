# -*- coding: utf-8 -*-
"""
Create LCMV beamformer and localise button presses in a CRT task.
Uses outputs from 1_forward_model and 2_preprocessing.
I found that LCMV works better for localising sources of oscillatory
modulation, while min norm works better for parcellating sources/connectivity.

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os.path as op

import numpy as np
import mne
from mne_bids import BIDSPath
from matplotlib import pyplot as plt

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'

# scanning session info
subject = '2014'
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

#%% epoch based on trigger

event_id = [1, 32]   # trigger of interest, [1 32] -> btn press, [101, 102] -> stim
tmin, tmax = -0.7, 1
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

#%% filter epochs for pseudo T

fband = [13, 30]
epochs_filt = epochs.copy().filter(fband[0], fband[1])

#%% compute active and control covariance of filtered data

act_min, act_max = 0.2, 0.6
con_min, con_max = -0.4, 0

active_left = mne.compute_covariance(epochs_filt["32"], tmin=act_min, tmax=act_max, method="shrunk")
control_left= mne.compute_covariance(epochs_filt["32"], tmin=con_min, tmax=con_max, method="shrunk")
#active_left.plot(epochs_filt.info)

active_right = mne.compute_covariance(epochs_filt["1"], tmin=act_min, tmax=act_max, method="shrunk")
control_right= mne.compute_covariance(epochs_filt["1"], tmin=con_min, tmax=con_max, method="shrunk")
#active_right.plot(epochs_filt.info)

#%% produce static sourcemap (pseudo T)

# left
stc_active = mne.beamformer.apply_lcmv_cov(active_left, filters)
stc_base = mne.beamformer.apply_lcmv_cov(control_left, filters)
change_left = (stc_active - stc_base) / (stc_active + stc_base)

# right
stc_active = mne.beamformer.apply_lcmv_cov(active_right, filters)
stc_base = mne.beamformer.apply_lcmv_cov(control_right, filters)
change_right = (stc_active - stc_base) / (stc_active + stc_base)

# for plotting
subjects_dir = r'R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS'
fs_subject = 'sub-' + subject

kwargs = dict(src=src, subject=fs_subject,
                subjects_dir=subjects_dir,
                surface="inflated",
                views=["lat", "med"],
                size=700,
                hemi="split",
                smoothing_steps=10,
                time_viewer=False,
                show_traces=False,
                colorbar=True)
 
change_left.plot(**kwargs, time_label="Left Presses")
change_right.plot(**kwargs, time_label="Right Presses")

#%% extract peak response in label(s) LEFT HAND

# create generator
stc_epochs = mne.beamformer.apply_lcmv_epochs(epochs["32"], filters,
                                              return_generator=True)

# get label names
parc = "aparc"
labels = mne.read_labels_from_annot(fs_subject, parc=parc, subjects_dir=subjects_dir)

# get induced peak within label
label_list = [33, 45, 49]
hemi = "rh"
label_name = hemi + " motor"   # CHANGE THIS TO MATCH LABEL_LIST!!!!!!

# combine vertices and pos from labels
vertices = []
pos = []
for l in label_list:
    vertices.append(labels[l].vertices)
    pos.append(labels[l].pos)
vertices = np.concatenate(vertices, axis=0)   
pos = np.concatenate(pos, axis=0)

# sort vertices and pos
vert_order = np.argsort(vertices)
vertices_ordered = vertices[vert_order]
pos_ordered = pos[vert_order,:]

new_label = mne.Label(vertices_ordered, pos_ordered, hemi=hemi, 
                      name=label_list, subject="sub-" + subject)

stc_inlabel = change_left.in_label(new_label)
label_peak = stc_inlabel.get_peak(mode="abs", vert_as_index=True)[0]

# extract timecourse of peak
n_epochs = len(epochs["32"])
epoch_len = np.shape(epochs[0])[2]
epoch_peak_data = np.zeros((n_epochs,1,epoch_len))
for s,stc_epoch in enumerate(stc_epochs):
    stc_epoch_label = mne.extract_label_time_course(stc_epoch, new_label, 
                                                    src, mode=None)
    epoch_peak_data[s,0,:] = stc_epoch_label[0][label_peak,:]
    
# make source epoch object
ch_names = ["peak"]
ch_types = ["misc"]
source_info = mne.create_info(ch_names=ch_names, sfreq=epochs.info["sfreq"],
                              ch_types=ch_types)
source_epochs = mne.EpochsArray(epoch_peak_data, source_info,
                                tmin=epochs.tmin)

# TFR
baseline = (-0.5, -0.2)
freqs = np.arange(1,35)
n_cycles = freqs/2
power = mne.time_frequency.tfr_morlet(source_epochs, freqs=freqs, n_cycles=n_cycles,
                                           use_fft=True, picks="all"
                                           )
power[0].plot(picks="all", baseline=baseline)

# timecourse
### trial averaged timecourse
source_epochs_filt = source_epochs.copy().filter(fband[0], fband[1], picks="all")
source_epochs_hilb = source_epochs_filt.copy().apply_hilbert(envelope=True, picks="all")
peak_timecourse = source_epochs_hilb.average(picks="all").apply_baseline(baseline)

# calculate MRBD (beta desync) or other during-stimulus response
stimulus_win = (-0.1, 0.1)
stimulus_ind = np.logical_and(peak_timecourse.times > stimulus_win[0], 
                          peak_timecourse.times < stimulus_win[1])
stimulus_response = np.mean(peak_timecourse.get_data()[0][stimulus_ind])

# calculate PMBR (beta rebound) or other poststimulus response
poststim_win = (0.2, 0.4)
poststim_ind = np.logical_and(peak_timecourse.times > poststim_win[0], 
                          peak_timecourse.times < poststim_win[1])
poststim_response = np.mean(peak_timecourse.get_data()[0][poststim_ind])

plt.figure()
plt.plot(peak_timecourse.times, peak_timecourse.get_data()[0], color="black")
plt.ylabel("Oscillatory Power (A.U)")
plt.xlabel("Time (s)")
plt.axhline(stimulus_response, alpha=0.5, color="blue")
plt.axhline(poststim_response, alpha=0.5, color="red")
plt.legend(["Data", "Stim Response", "Poststim Response"])
plt.title(new_label.name)

lh_response_fname = deriv_path.basename + "-lh_resp.npy"
np.save(op.join(deriv_path.directory, lh_response_fname), peak_timecourse.get_data()[0])


#%% extract peak response in label(s) RIGHT HAND

# create generator
stc_epochs = mne.beamformer.apply_lcmv_epochs(epochs["1"], filters,
                                              return_generator=True)

# get label names
parc = "aparc"
labels = mne.read_labels_from_annot(fs_subject, parc=parc, subjects_dir=subjects_dir)

# get induced peak within label
label_list = [32, 44, 48]
hemi = "lh"
label_name = hemi + " motor"   # CHANGE THIS TO MATCH LABEL_LIST!!!!!!

# combine vertices and pos from labels
vertices = []
pos = []
for l in label_list:
    vertices.append(labels[l].vertices)
    pos.append(labels[l].pos)
vertices = np.concatenate(vertices, axis=0)   
pos = np.concatenate(pos, axis=0)

# sort vertices and pos
vert_order = np.argsort(vertices)
vertices_ordered = vertices[vert_order]
pos_ordered = pos[vert_order,:]

new_label = mne.Label(vertices_ordered, pos_ordered, hemi=hemi, 
                      name=label_list, subject="sub-" + subject)

stc_inlabel = change_left.in_label(new_label)
label_peak = stc_inlabel.get_peak(mode="abs", vert_as_index=True)[0]

# extract timecourse of peak
n_epochs = len(epochs["1"])
epoch_len = np.shape(epochs[0])[2]
epoch_peak_data = np.zeros((n_epochs,1,epoch_len))
for s,stc_epoch in enumerate(stc_epochs):
    stc_epoch_label = mne.extract_label_time_course(stc_epoch, new_label, 
                                                    src, mode=None)
    epoch_peak_data[s,0,:] = stc_epoch_label[0][label_peak,:]
    
# make source epoch object
ch_names = ["peak"]
ch_types = ["misc"]
source_info = mne.create_info(ch_names=ch_names, sfreq=epochs.info["sfreq"],
                              ch_types=ch_types)
source_epochs = mne.EpochsArray(epoch_peak_data, source_info,
                                tmin=epochs.tmin)

# TFR
baseline = (-0.5, -0.2)
freqs = np.arange(1,35)
n_cycles = freqs/2
power = mne.time_frequency.tfr_morlet(source_epochs, freqs=freqs, n_cycles=n_cycles,
                                           use_fft=True, picks="all"
                                           )
power[0].plot(picks="all", baseline=baseline)

# timecourse
### trial averaged timecourse
source_epochs_filt = source_epochs.copy().filter(fband[0], fband[1], picks="all")
source_epochs_hilb = source_epochs_filt.copy().apply_hilbert(envelope=True, picks="all")
peak_timecourse = source_epochs_hilb.average(picks="all").apply_baseline(baseline)

# calculate MRBD (beta desync) or other during-stimulus response
stimulus_win = (-0.1, 0.1)
stimulus_ind = np.logical_and(peak_timecourse.times > stimulus_win[0], 
                          peak_timecourse.times < stimulus_win[1])
stimulus_response = np.mean(peak_timecourse.get_data()[0][stimulus_ind])

# calculate PMBR (beta rebound) or other poststimulus response
poststim_win = (0.2, 0.4)
poststim_ind = np.logical_and(peak_timecourse.times > poststim_win[0], 
                          peak_timecourse.times < poststim_win[1])
poststim_response = np.mean(peak_timecourse.get_data()[0][poststim_ind])

plt.figure()
plt.plot(peak_timecourse.times, peak_timecourse.get_data()[0], color="black")
plt.ylabel("Oscillatory Power (A.U)")
plt.xlabel("Time (s)")
plt.axhline(stimulus_response, alpha=0.5, color="blue")
plt.axhline(poststim_response, alpha=0.5, color="red")
plt.legend(["Data", "Stim Response", "Poststim Response"])
plt.title(new_label.name)

rh_response_fname = deriv_path.basename + "-rh_resp.npy"
np.save(op.join(deriv_path.directory, rh_response_fname), peak_timecourse.get_data()[0])
