# -*- coding: utf-8 -*-
"""
Extract peak theta response in motor cortex in a CRT task. 
Uses minimum norm inverse solution.

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os.path as op
import os
import numpy as np
import mne
from mne_bids import BIDSPath
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, apply_inverse_cov
from matplotlib import pyplot as plt

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'
out_folder = op.join(deriv_root, 'min_norm_induced')

# options
plot_brains = False
plot_timecourse = False
plot_TFS = False

# scanning session info
for subject in ['2001', '2003', '2008', '2009', '2014']:
    #subject = '2001'    
    # each session
    sessions = os.listdir(op.join(deriv_root, "sub-" + subject))
    sessions = [sessions[i][4:] for i in range(len(sessions))]    
    for session in sessions:
        session = '01N'
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
        
        src = mne.read_source_spaces(src)
        fwd = mne.make_forward_solution(data.info, trans, src, bem, verbose=True)
        inv = make_inverse_operator(data.info, fwd, noise_cov)
        
        #%% get pseudo T
        
        # filter epochs
        fband = [13, 30]
        epochs_filt = epochs.copy().filter(fband[0], fband[1])
        lambda2 = 1  # this should be 1/SNR^2, but we assume SNR=1 for non-evoked data
        
        # active and control covariance of filtered data
        act_min, act_max = 0.3, 0.6
        con_min, con_max = -0.3, 0
        
        active_left = mne.compute_covariance(epochs_filt["102"], tmin=act_min, tmax=act_max, method="shrunk")
        control_left= mne.compute_covariance(epochs_filt["102"], tmin=con_min, tmax=con_max, method="shrunk")
        
        active_right = mne.compute_covariance(epochs_filt["101"], tmin=act_min, tmax=act_max, method="shrunk")
        control_right= mne.compute_covariance(epochs_filt["101"], tmin=con_min, tmax=con_max, method="shrunk")
        
        # left pseudo T
        stc_active = apply_inverse_cov(
            active_left, epochs.info, inv, lambda2=lambda2, pick_ori="normal",
            method="eLORETA"
            )
        stc_base = apply_inverse_cov(
            control_left, epochs.info, inv, lambda2=lambda2, pick_ori="normal",
            method="eLORETA"
            ) 
        pseudoT_left = (stc_active - stc_base) / (stc_active + stc_base)
        
        # right pseudo T
        stc_active = apply_inverse_cov(
            active_right, epochs.info, inv, lambda2=lambda2, pick_ori="normal",
            method="eLORETA"
            )
        stc_base = apply_inverse_cov(
            control_right, epochs.info, inv, lambda2=lambda2, pick_ori="normal",
            method="eLORETA"
            ) 
        pseudoT_right = (stc_active - stc_base) / (stc_active + stc_base)
        
        # for plotting
        subjects_dir = r'R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS'
        fs_subject = 'sub-' + subject
        
        if plot_brains:
            for inst in [pseudoT_left, pseudoT_right]:
                inst.plot(src=src, subject=fs_subject,
                            subjects_dir=subjects_dir,
                            surface="inflated",
                            views=["lat", "med"],
                            size=600,
                            hemi="split",
                            smoothing_steps=10,
                            time_viewer=False,
                            show_traces=False,
                            colorbar=True)
                
        # save pseudo T
        pseudoT_left_fname = deriv_path.basename + "-left_stim_beta"
        pseudoT_left.save(op.join(out_folder, pseudoT_left_fname), overwrite=True)
        pseudoT_right_fname = deriv_path.basename + "-right_stim_beta"
        pseudoT_right.save(op.join(out_folder, pseudoT_right_fname), overwrite=True)
        
        #%% extract peak timecourse LEFT HAND (right motor)
        
        # create generator
        stc_epochs = apply_inverse_epochs(
            epochs["102"], inv, lambda2=lambda2, pick_ori="normal", return_generator=True,
            method="eLORETA"
        )
        
        # get label names
        parc = "aparc"
        labels = mne.read_labels_from_annot(fs_subject, parc=parc, subjects_dir=subjects_dir)
        
        # get induced peak within label
        label_list = [33, 45, 49, 63]
        hemi = "rh"
        label_name = hemi + " somatosensory"   # CHANGE THIS TO MATCH LABEL_LIST!!!!!!
        
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
                              name=label_name, subject="sub-" + subject)
        
        stc_inlabel = pseudoT_left.in_label(new_label)
        label_peak = stc_inlabel.get_peak(mode="abs", vert_as_index=True)[0]
        
        # extract timecourse of peak
        n_epochs = len(epochs["102"])
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
        if plot_TFS:
            baseline = (-0.5, -0.2)
            freqs = np.arange(1,35)
            n_cycles = freqs/2
            power = mne.time_frequency.tfr_morlet(source_epochs, freqs=freqs, n_cycles=n_cycles,
                                                       use_fft=True, picks="all"
                                                       )
            power[0].plot(picks="all", baseline=baseline)
        
        # timecourse
        if plot_timecourse:
            source_epochs_filt = source_epochs.copy().filter(fband[0], fband[1], picks="all")
            source_epochs_hilb = source_epochs_filt.copy().apply_hilbert(envelope=True, picks="all")
            peak_timecourse = source_epochs_hilb.average(picks="all").apply_baseline(baseline)
            
            plt.figure()
            plt.plot(peak_timecourse.times, peak_timecourse.get_data()[0], color="black")
            plt.ylabel("Oscillatory Power (A.U)")
            plt.xlabel("Time (s)")
            plt.title(new_label.name)
        
        # save peak source epoch
        lh_source_fname = deriv_path.basename + "-lh_beta_source.fif"
        source_epochs.save(op.join(out_folder, lh_source_fname), overwrite=True)
        
        #%% extract peak timecourse RIGHT HAND (left motor)
        
        # create generator
        stc_epochs = apply_inverse_epochs(
            epochs["101"], inv, lambda2=lambda2, pick_ori="normal", return_generator=True,
            method="eLORETA"
        )
        
        # get label names
        parc = "aparc"
        labels = mne.read_labels_from_annot(fs_subject, parc=parc, subjects_dir=subjects_dir)
        
        # get induced peak within label
        label_list = [32, 44, 48, 62]
        hemi = "lh"
        label_name = hemi + " somatosensory"   # CHANGE THIS TO MATCH LABEL_LIST!!!!!!
        
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
                              name=label_name, subject="sub-" + subject)
        
        stc_inlabel = pseudoT_left.in_label(new_label)
        label_peak = stc_inlabel.get_peak(mode="abs", vert_as_index=True)[0]
        
        # extract timecourse of peak
        n_epochs = len(epochs["101"])
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
        if plot_TFS:
            baseline = (-0.5, -0.2)
            freqs = np.arange(1,35)
            n_cycles = freqs/2
            power = mne.time_frequency.tfr_morlet(source_epochs, freqs=freqs, n_cycles=n_cycles,
                                                       use_fft=True, picks="all"
                                                       )
            power[0].plot(picks="all", baseline=baseline)
        
        # timecourse
        if plot_timecourse:
            source_epochs_filt = source_epochs.copy().filter(fband[0], fband[1], picks="all")
            source_epochs_hilb = source_epochs_filt.copy().apply_hilbert(envelope=True, picks="all")
            peak_timecourse = source_epochs_hilb.average(picks="all").apply_baseline(baseline)
            
            plt.figure()
            plt.plot(peak_timecourse.times, peak_timecourse.get_data()[0], color="black")
            plt.ylabel("Oscillatory Power (A.U)")
            plt.xlabel("Time (s)")
            plt.title(new_label.name)
        
        # save peak source epoch
        rh_source_fname = deriv_path.basename + "-rh_beta_source.fif"
        source_epochs.save(op.join(out_folder, rh_source_fname), overwrite=True)