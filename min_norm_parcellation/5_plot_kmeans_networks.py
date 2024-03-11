# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:27:22 2024

@author: ppysc6
"""

import os.path as op
import os

import numpy as np
import mne
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid, inset_locator, make_axes_locatable

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-PSR\Seb\mTBI_testing\derivatives'

# scanning session info
subject = '2001'
task = 'CRT'  # name of the task
run = '01'
suffix = 'meg'

#%% load kmeans outputs

k = 6
centroids_fname = op.join(deriv_root, "kmeans_networks_all_sessions", 
                          subject + "_" + str(k) + "k_centroids.npy")
idx_fname = op.join(deriv_root, "kmeans_networks_all_sessions", 
                          subject + "_" + str(k) + "k_idx.npy")
centroids = np.load(centroids_fname)
idx = np.load(idx_fname)

#%% create epochs object out of microstate index timecourses

sfreq = 250

cluster_timecourses = np.zeros((k, len(idx)))
for state in range(k):
    cluster_timecourses[state,:] = idx==state
    
cluster_names=["cluster_" + str(i) for i in range(k)]
cluster_info = mne.create_info(ch_names=cluster_names, sfreq=sfreq, ch_types='misc')
cluster_raw = mne.io.RawArray(cluster_timecourses, cluster_info)

duration = 2 + 1/sfreq
events = mne.make_fixed_length_events(cluster_raw, duration=duration)

event_id = 1
tmin, tmax = 0, 2
cluster_epochs = mne.Epochs(
    cluster_raw,
    events,
    event_id,
    tmin,
    tmax,
    baseline=None,
    preload=True)

#%% plot timecourse of each cluster

cluster_evoked = cluster_epochs.average("all")

for state in range(k):
    cluster_evoked.plot("cluster_" + str(state))

#%% take screenshot of brain plot

fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
parc = "aparc"
labels = mne.read_labels_from_annot("fsaverage", parc=parc, subjects_dir=subjects_dir)
labels = labels[:-1]

colormap = "bwr"
clim=dict(kind="value", pos_lims=[0.4, 0.6, 1])

screenshot = []

for state in range(k):
    stc = mne.labels_to_stc(labels, centroids[state,:].transpose())
    
    brain = stc.plot(
        clim=clim,
        colormap=colormap,
        background="w",
        subjects_dir=subjects_dir,
        views=["lat", "med"],
        surface="inflated",
        size=600,
        hemi="split",
        smoothing_steps=10,
        time_viewer=False,
        show_traces=False,
        colorbar=False,
        alpha=1
    )
    screenshot.append(brain.screenshot())
    brain.close()
    
#%% crop white space

cropped_screenshot = []
for shot in screenshot:
    nonwhite_pix = (shot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot.append(shot[nonwhite_row][:, nonwhite_col])
    
# plot before and after example
fig = plt.figure(figsize=(4, 4))
axes = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0.5)
for ax, image, title in zip(
    axes, [screenshot[0], cropped_screenshot[0]], ["Before", "After"]
):
    ax.imshow(image)
    ax.set_title("{} cropping".format(title))
    
#%% set rc params

# Tweak the figure style
plt.rcParams.update(
    {
        "ytick.labelsize": "small",
        "xtick.labelsize": "small",
        "axes.labelsize": "medium",
        "axes.titlesize": "medium",
        "grid.color": "0.75",
        "grid.linestyle": ":",
    }
)

#%% create figure

for state in range(k):

    # figsize unit is inches
    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(6, 6), gridspec_kw=dict(height_ratios=[5, 3])
    )
    
    
    # we'll put the timecourse plot in the lower axes, and the brain above
    timecourse_idx = 1
    brain_idx = 0
    
    divider = make_axes_locatable(axes[brain_idx])
    cax = divider.append_axes("right", size="5%", pad=0.5)
    
    time = np.linspace(-0.8, 1.2, len(cluster_evoked.get_data()[state]))
    axes[timecourse_idx].plot(time, cluster_evoked.get_data()[state], color="black")
    axes[timecourse_idx].axvline(0, alpha=0.5, color="gray")
    axes[timecourse_idx].set_ylabel("Fractional Occupancy")
    axes[timecourse_idx].set_xlabel("Time (s)")
    axes[timecourse_idx].grid(True)
    
    # brain plot
    axes[brain_idx].imshow(cropped_screenshot[state])
    axes[brain_idx].axis("off")
    # add a vertical colorbar with the same properties as the 3D one
    cbar = mne.viz.plot_brain_colorbar(cax, clim, colormap, label="Alpha Envelope (SD)")
    plt.tight_layout()