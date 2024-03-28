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
from mne.minimum_norm import make_inverse_operator, apply_inverse
from matplotlib import pyplot as plt

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'
out_folder = op.join(deriv_root, 'min_norm_evoked')

# options
plot_brains = False
plot_timecourse = True

# scanning session info
for subject in ['2001', '2003', '2008', '2009', '2014']:
    #subject = '2001'    
    # each session
    sessions = os.listdir(op.join(deriv_root, "sub-" + subject))
    sessions = [sessions[i][4:] for i in range(len(sessions))]    
    for session in sessions[1:]:
        #session = '01N'
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
        
        event_id = [1, 32] #[101, 102]  # trigger of interest
        tmin, tmax = -0.6, 1
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
        
        #%% get activation map
        
        # average epochs
        evoked_left = epochs["32"].average()
        evoked_right = epochs["1"].average()
        lambda2 = 1/9  # this should be 1/SNR^2, but we assume SNR=1 for non-evoked data
        
        # apply inverse
        stc_left = apply_inverse(
            evoked_left, inv, lambda2=lambda2, pick_ori="normal",
            method="eLORETA"
        )
        
        stc_right = apply_inverse(
            evoked_right, inv, lambda2=lambda2, pick_ori="normal",
            method="eLORETA"
        )
        
        # for plotting
        subjects_dir = r'R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS'
        fs_subject = 'sub-' + subject
        
        if plot_brains:
            brain = stc_right.plot(
                surface="inflated",
                hemi="both",
                src=src,
                views="coronal",
                subjects_dir=subjects_dir,
                brain_kwargs=dict(silhouette=True),
                smoothing_steps=7,
            )
            
        # get label names
        parc = "HCPMMP1_combined"
        labels = mne.read_labels_from_annot("fsaverage", parc=parc, subjects_dir=subjects_dir)
        labels = labels[2:]
        
        # morph labels to subject
        fs_subject = 'sub-' + subject
        labels = mne.morph_labels(labels, fs_subject, "fsaverage", subjects_dir)
        
        #### LEFT HAND BUTTON PRESSES PEAK
        # get induced peak within label
        label_list = [27, 33, 37]
        hemi = "rh"
        label_name = hemi + " sensorimotor"   # CHANGE THIS TO MATCH LABEL_LIST!!!!!!
        
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
        
        stc_inlabel = stc_left.in_label(new_label)
        peak_left = stc_inlabel.data[stc_inlabel.get_peak(mode='abs',vert_as_index=True)[0],:]
        
        #### RIGHT HAND BUTTON PRESSES PEAK
        # get induced peak within label
        label_list = [26, 32, 36]
        hemi = "lh"
        label_name = hemi + " sensorimotor"   # CHANGE THIS TO MATCH LABEL_LIST!!!!!!
        
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
        
        stc_inlabel = stc_right.in_label(new_label)
        peak_right = stc_inlabel.data[stc_inlabel.get_peak(mode='abs',vert_as_index=True)[0],:]
        
        if plot_timecourse:
            plt.figure()
            plt.plot(stc_right.times, peak_right)
            plt.plot(stc_left.times, peak_left)
            plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
            plt.legend(["right btn", "left btn"])
        
        peak_right_fname = deriv_path.basename + "-rh_btn_ERP.npy"
        peak_left_fname = deriv_path.basename + "-lh_btn_ERP.npy"
        np.save(op.join(out_folder, peak_right_fname), peak_right)
        np.save(op.join(out_folder, peak_left_fname), peak_left)

        