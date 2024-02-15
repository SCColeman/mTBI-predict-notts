# -*- coding: utf-8 -*-
"""
A script to perform pre-processing on Nottingham mTBI-predict data, 
or any CTF data. The script assumes that the data has already been 
converted to BIDS format.

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

#%% get movement parameters

chpi_locs = mne.chpi.extract_chpi_locs_ctf(data, verbose=False)
head_pos = mne.chpi.compute_head_pos(data.info, chpi_locs, verbose=False)
original_head_dev_t = mne.transforms.invert_transform(data.info["dev_head_t"])
#average_head_dev_t = mne.transforms.invert_transform(compute_average_dev_head_t(thrd_data, head_pos))
head_movement_fig = mne.viz.plot_head_positions(head_pos)

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

data.filter(l_freq=1, h_freq=100)
data_bband = data.copy()

#%% remove bad channels (LEFT CLICK ON PLOT)

data.plot()

#%% find and annotate EOG events

EOG_channels = ["UADC007-4123", "UADC010-4123"]
ECG_channels = ["UADC009-4123"]
eog_events = mne.preprocessing.find_eog_events(data, ch_name=EOG_channels[0])
eog_onset = eog_events[:,0]/data.info['sfreq'] - 0.2
eog_duration = np.repeat(0.5, len(eog_events))
eog_description = ['blink'] * len(eog_events)
eog_annot = mne.Annotations(eog_onset, eog_duration, eog_description, 
                        orig_time=data.info['meas_date'])
data.set_annotations(eog_annot)
data.copy().pick(EOG_channels[0]).plot()
data_eog = data.copy()

#%% annotate muscle artefacts

threshold_muscle = 10  # z-score
annot_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
    data,
    threshold=threshold_muscle,   # zscore
    ch_type = "mag",
    min_length_good=0.2,
    filter_freq=(80, 100)
)

fig, ax = plt.subplots()
ax.plot(data.times, scores_muscle)
ax.axhline(y=threshold_muscle, color="r")
ax.set(xlabel="time, (s)", ylabel="zscore", title="Muscle activity")
data.set_annotations(annot_muscle)
data_muscle = data.copy()

#%% ICA (LEFT CLICK ON CARDIAC AND BLINKING TIMECOURSES)

ica = mne.preprocessing.ICA(n_components=20)
ica.fit(data.copy().pick_types(meg=True))
ica.plot_components()
ica.plot_sources(data)

#%% remove bad components (THE ONES YOU CLICKED ON IN PREVIOUS PLOT)

ica.apply(data)
data_ica = data.copy()

#%% save data

preproc_fname = deriv_path.basename + "_preproc.fif"
events_fname = deriv_path.basename + "_preproc_events.fif"
#head_pos_fname = deriv_path.basename + "_preproc_headpos.fif"
data.save(op.join(deriv_path.directory, preproc_fname), overwrite=True)
#head_pos.save(op.join(deriv_path.directory, head_pos_fname), overwrite=True)
mne.write_events(op.join(deriv_path.directory, events_fname), events, overwrite=True)

#%% report

# plot psd after preprocessing to put in report
timecourse_before = data_raw.copy().pick("mag").plot()
timecourse_after = data.copy().pick("mag").plot()
psd_before = data_raw.plot_psd(fmax=100, picks='mag').show()
psd_after = data.plot_psd(fmax=100, picks='mag').show()

report_fname = deriv_path.basename + "_preproc_report.html"
report = mne.Report(title=deriv_path.basename)
report.add_raw(raw=data_raw.copy().pick("mag"), title='Raw', psd=True, butterfly=True)
report.add_figure(timecourse_before, "Raw Channel Data")
report.add_ica(ica, title="ICA", inst=data)
report.add_raw(raw=data.copy().pick("mag"), title='Preprocessed', psd=True, butterfly=True)
report.add_figure(timecourse_after, "Preprocessed Channel Data")
report.save(op.join(deriv_path.directory, report_fname), overwrite=True, open_browser=True)



