
# -*- coding: utf-8 -*-
"""
Source-space connectivity of button presses in CRT task. CTF MEG data.
Uses minimum-norm inverse solution.

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os.path as op

import numpy as np
import mne
from mne_bids import BIDSPath
from matplotlib import pyplot as plt
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
import mne_connectivity

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

#%% orthogonalised envelope connectivity

bp_lims = [8, 13]

def bp_gen(label_ts):
    """Make a generator that band-passes on the fly."""
    for ts in label_ts:
        yield mne.filter.filter_data(ts, sfreq, bp_lims[0], bp_lims[1])

corr_obj = mne_connectivity.envelope_correlation(bp_gen(label_ts), 
                                                 orthogonalize="pairwise")
corr = corr_obj.combine()
corr = corr.get_data(output="dense")[:, :, 0]

def plot_corr(corr, title):
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.imshow(corr, cmap="viridis", clim=np.percentile(corr, [5, 95]))
    fig.suptitle(title)

plot_corr(corr, "Pairwise AEC")
plt.yticks(ticks=np.arange(0, len(labels), 4), 
           labels=[labels[i].name for i in np.arange(0, len(labels), 4)])
plt.xticks(ticks=np.arange(0, len(labels), 4), 
           labels=[labels[i].name for i in np.arange(0, len(labels), 4)],
           rotation=90)

def plot_degree(corr, title):
    threshold_prop = 0.15  # percentage of strongest edges to keep in the graph
    degree = mne_connectivity.degree(corr, threshold_prop=threshold_prop)
    stc = mne.labels_to_stc(labels, degree)
    stc = stc.in_label(
        mne.Label(inv["src"][0]["vertno"], hemi="lh")
        + mne.Label(inv["src"][1]["vertno"], hemi="rh")
    )
    return stc.plot(
        clim=dict(kind="percent", lims=[75, 85, 95]),
        colormap="gnuplot",
        subjects_dir=subjects_dir,
        views="dorsal",
        hemi="both",
        smoothing_steps=10,
        time_label=title,
    )

brain = plot_degree(corr, "pairwise connectivity, DK atlas")