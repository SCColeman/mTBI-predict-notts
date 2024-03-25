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

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'

# scanning session info
subject = '2011'
for session in ['03N']:
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
    
    #%% make events "raw" that we can add to the source epochs object later
    
    events_data = np.zeros((4, len(data)))
    for e, event in enumerate([102, 101, 32, 1]):
        event_i = events[:,0][events[:,2]==event]
        events_data[e, event_i] = 1
        
    names = ["stim_left", "stim_right", "resp_left", "resp_right"]
    events_info = mne.create_info(ch_names=names, 
                               sfreq=data.info["sfreq"], 
                               ch_types='stim')
    events_raw = mne.io.RawArray(events_data, events_info)
    data.load_data().add_channels([events_raw], force_update_info=True)
    
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
    
    events_epochs = epochs.copy().pick('stim') # to add to source epochs
    
    #%% make forward model and inverse from files
    
    src = mne.read_source_spaces(src)
    fwd = mne.make_forward_solution(data.info, trans, src, bem, verbose=True)
    inv = make_inverse_operator(data.info, fwd, noise_cov)
    
    #%% test beamformer by plotting pseudo T
    
    fband = [4, 8]
    epochs_filt = epochs.copy().filter(fband[0], fband[1])
    lambda2 = 1  # this should be 1/SNR^2, but we assume SNR=1 for non-evoked data
    
    act_min, act_max = 0.3, 0.6
    con_min, con_max = -0.5, -0.3
    
    active_cov = mne.compute_covariance(epochs_filt, tmin=act_min, tmax=act_max, method="shrunk")
    control_cov= mne.compute_covariance(epochs_filt, tmin=con_min, tmax=con_max, method="shrunk")
    
    stc_active = apply_inverse_cov(
        active_cov, epochs.info, inv, lambda2=lambda2, pick_ori="normal",
        method="eLORETA"
        )
    stc_base = apply_inverse_cov(
        control_cov, epochs.info, inv, lambda2=lambda2, pick_ori="normal",
        method="eLORETA"
        )
     
    stc_change = (stc_active - stc_base) / (stc_active + stc_base)
    
    # for plotting
    subjects_dir = r'R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS'
    fs_subject = 'sub-' + subject
     
    stc_change.plot(src=src, subject=fs_subject,
                    subjects_dir=subjects_dir,
                    surface="inflated",
                    views=["lat", "med"],
                    size=600,
                    hemi="split",
                    smoothing_steps=10,
                    time_viewer=False,
                    show_traces=False,
                    colorbar=True,)
    
    #%% parcellation beamformer
    
    # get labels from parcellation
    subjects_dir = r'R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS'
    fs_subject = 'sub-' + subject
    parc = "aparc"
    labels = mne.read_labels_from_annot(fs_subject, parc=parc, subjects_dir=subjects_dir)
    
    stcs = apply_inverse_epochs(
        epochs, inv, lambda2=lambda2, pick_ori="normal", return_generator=True,
        method="eLORETA"
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
    
    # add events 
    source_epochs.add_channels([events_epochs], force_update_info=True)
    
    source_epochs.plot("all")
    
    #%% save source epochs
    
    source_epochs_fname = deriv_path.basename + "-source_epochs.fif"
    source_epochs.save(op.join(deriv_path.directory, source_epochs_fname), overwrite=True)
