# -*- coding: utf-8 -*-
"""
A script to perform forward modelling and beamforming on Nottingham 
mTBI-predict data, or any CTF data. The script assumes that the data has 
already been converted to BIDS format. The script also assumes that
you have a FreeSurfer anatomical reconstruction for the subject.

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
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'

# scanning session info
subject = '2001'
session = '01N'
task = 'CRT'  # name of the task
run = '01'
suffix = 'meg'

bids_path = BIDSPath(subject=subject, session=session,
task=task, run=run, suffix=suffix, root=bids_root)

deriv_path = BIDSPath(subject=subject, session=session,
task=task, run=run, suffix=suffix, root=deriv_root)

#%% load in preprocessed data 

data = mne.io.Raw(op.join(deriv_path.directory, deriv_path.basename + "_preproc.fif"))
info = data.info
data.pick("mag")

#%% load in events

events = mne.read_events(op.join(deriv_path.directory, 
                                 deriv_path.basename + "_preproc_events.fif"))

#%% epoch based on trigger

event_id = 101     # trigger of interest, [1 31] -> btn press, [101, 102] -> stim
tmin, tmax = -0.5, 1
epochs = mne.Epochs(
    data,
    events,
    event_id,
    tmin,
    tmax,
    baseline=None,
    preload=True,
    reject_by_annotation=True)

#%% prefilter epochs before calculating covariance

fband = [8, 13]
epochs.filter(fband[0], fband[1])
epochs_filt = epochs.copy()

#%% compute covariance

act_min, act_max = 0.2, 0.6
con_min, con_max = -0.4, 0

active_cov = mne.compute_covariance(epochs_filt, tmin=act_min, tmax=act_max, method="shrunk")
control_cov= mne.compute_covariance(epochs_filt, tmin=con_min, tmax=con_max, method="shrunk")
all_cov = mne.compute_covariance(epochs_filt, method="shrunk")
all_cov.plot(epochs_filt.info)

#%% Get FS reconstruction for subject or use fsaverage for quick testing

fs_dir = mne.datasets.fetch_fsaverage(verbose=True)  # for fsaverage
#subjects_dir = op.dirname(fs_dir)    # for fsaverage
subjects_dir = r'R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS'

# Name of the subject directory in FS subjects_dir
fs_subject = 'sub-' + subject

plot_bem_kwargs = dict(
    subject=fs_subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="coronal",
    slices=[50, 100, 150, 200])

mne.viz.plot_bem(**plot_bem_kwargs)

#%% coregistration

plot_kwargs = dict(
    subject=fs_subject,
    subjects_dir=subjects_dir,
    surfaces="head-dense",
    dig=True,
    meg="sensors",
    show_axes=True,
    coord_frame="meg",
)

coreg = mne.coreg.Coregistration(data.info, fs_subject, 
                            subjects_dir=subjects_dir)
mne.viz.plot_alignment(data.info, trans=coreg.trans, **plot_kwargs)
coreg.fit_fiducials()
coreg.fit_icp(20)
mne.viz.plot_alignment(data.info, trans=coreg.trans, **plot_kwargs)

### UNCOMMENT THE FOLLOWING IF USING GUI RATHER THAN AUTOMATIC COREG

#coreg_gui = mne.gui.coregistration(subjects_dir=subjects_dir, subject=fs_subject, scale_by_distance=False)
#coreg = coreg_gui.coreg
#mne.viz.plot_alignment(data.info, trans=coreg.trans, **plot_kwargs)

#%% compute source space

surf_file = op.join(subjects_dir, fs_subject, "bem", "inner_skull.surf")

# can change oct5 to other surface source space
src = mne.setup_source_space(
    fs_subject, spacing="oct5", add_dist=False, subjects_dir=subjects_dir)
src.plot(subjects_dir=subjects_dir)

#%% forward solution

conductivity = (0.3,)
model = mne.make_bem_model(
    subject=fs_subject, ico=4,
    conductivity=conductivity,
    subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

fwd = mne.make_forward_solution(
    data.info,
    trans=coreg.trans,
    src=src,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5,
    )
print(fwd)

#%% spatial filter

filters = mne.beamformer.make_lcmv(
    data.info,
    fwd,
    all_cov,
    reg=0.05,
    noise_cov=None,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None)

#%% apply filter to covariance for static sourcemap
 
stc_active = mne.beamformer.apply_lcmv_cov(active_cov, filters)
stc_base = mne.beamformer.apply_lcmv_cov(control_cov, filters)
 
stc_change = (stc_active - stc_base) / (stc_active + stc_base)
 
#%% visualise static sourcemap
 
stc_change.plot(src=src, subject=fs_subject,
                subjects_dir=subjects_dir, surface="pial", hemi="both")

#%% apply beamformer to preprocessed raw data for timecourse extraction
 
stc_raw = mne.beamformer.apply_lcmv_raw(data, filters)
 
#%% extract absolute max voxel TFS/timecourse
 
peak = stc_change.get_peak(mode="abs", vert_as_index=True)[0]
 
stc_peak = stc_raw.data[peak]

# make fake raw object from source time course
 
ch_names = ["peak"]
ch_types = ["misc"]
source_info = mne.create_info(ch_names=ch_names, sfreq=info["sfreq"],
                              ch_types=ch_types)
source_raw = mne.io.RawArray([stc_peak], source_info)

baseline = (-0.5, -0.2)
source_epochs = mne.Epochs(
    source_raw,
    events=events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True)
 
 
# TFR
freqs = np.logspace(*np.log10([6, 35]), num=20)
n_cycles = freqs/2
power = mne.time_frequency.tfr_morlet(source_epochs, freqs=freqs, n_cycles=n_cycles,
                                           use_fft=True, picks="all"
                                           )
power[0].plot(picks="all", baseline=baseline)

#%% apply beamformer to filtered raw data for timecourse extraction
 
stc_filt = mne.beamformer.apply_lcmv_raw(data.load_data().filter(fband[0], fband[1]), filters)

#%% extract peak timecourse from within a parcel

# get label names
parc = "aparc"
labels = mne.read_labels_from_annot(fs_subject, parc=parc, subjects_dir=subjects_dir)

# get induced peak within label
label = 6
stc_inlabel = stc_change.in_label(labels[label])
label_peak = stc_inlabel.get_peak(mode="abs", vert_as_index=True)[0]

# extract timecourse of peak
stc_label_all = mne.extract_label_time_course(stc_filt, labels[label], src, mode=None)
stc_label_peak = stc_label_all[0][label_peak,:]

# make fake raw object from source time course
ch_names = ["peak"]
ch_types = ["misc"]
source_info = mne.create_info(ch_names=ch_names, sfreq=info["sfreq"],
                              ch_types=ch_types)
source_label_raw = mne.io.RawArray([stc_label_peak], source_info)

baseline = (-0.5, -0.2)
parcel_epochs = mne.Epochs(
    source_label_raw,
    events=events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True)
 

### trial averaged timecourse
parcel_epochs.apply_hilbert(envelope=True, picks="all")
peak_timecourse = parcel_epochs.average(picks="all").apply_baseline(baseline)
peak_timecourse.plot()

# calculate MRBD (beta desync) or other during-stimulus response
stimulus_win = (0, 0.2)
stimulus_ind = np.logical_and(peak_timecourse.times > stimulus_win[0], 
                          peak_timecourse.times < stimulus_win[1])
stimulus_response = np.mean(peak_timecourse.get_data()[0][stimulus_ind])

# calculate PMBR (beta rebound) or other poststimulus response
poststim_win = (0.4, 0.6)
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
plt.title(labels[label].name)
