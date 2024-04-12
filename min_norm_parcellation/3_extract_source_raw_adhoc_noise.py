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
from scipy.stats import zscore
from mne_connectivity import symmetric_orth

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'
out_path = op.join(deriv_root, 'HMM', 'data')
subjects = ['2001', '2003', '2008', '2009', '2014']
plot_brains = False
plot_epochs = False

# scanning session info
for s, subject in enumerate(subjects):

    # load data for each session
    sessions = os.listdir(op.join(deriv_root, "sub-" + subject))
    sessions = [sessions[i][4:] for i in range(len(sessions))]
    
    for s, session in enumerate(sessions):
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
        stim_raw = data.copy().pick('stim')
        
        #%% calculate data and noise covariance
        
        cov = mne.compute_raw_covariance(data)
        noise_cov = mne.compute_raw_covariance(noise)
        
        #%% calculate ad-hoc noise cov from data cov
        
        n_channels = cov.data.shape[0]
        noise_avg = np.mean(np.diag(cov.data))
        noise_cov_diag = np.array([noise_avg] * n_channels)
        # take mean of diag of data cov
        noise_cov = mne.Covariance(noise_cov_diag, cov.ch_names, 
                                   data.info['bads'], data.info["projs"], 
                                   nfree=1e10)
        
        #%% make forward model and inverse from files
        
        src = mne.read_source_spaces(src)
        fwd = mne.make_forward_solution(data.info, trans, src, bem, verbose=True)
        inv = make_inverse_operator(data.info, fwd, noise_cov)
        
        #%% pseudo-epoch to create generator (saves RAM)
        
        duration = 2 + 1/sfreq
        events = mne.make_fixed_length_events(data, duration=duration)
        
        event_id = 1
        tmin, tmax = 0, 2
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
        
        stim_epochs = epochs.copy().pick('stim')
            
        #%% parcellation beamformer
        
        # get labels from parcellation
        subjects_dir = r'R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS'
        parc = "HCPMMP1_combined"
        labels = mne.read_labels_from_annot("fsaverage", parc=parc, subjects_dir=subjects_dir)
        labels = labels[2:]
        
        # morph labels to subject
        fs_subject = 'sub-' + subject
        labels = mne.morph_labels(labels, fs_subject, "fsaverage", subjects_dir)
        
        lambda2 = 1  # this should be 1/SNR^2, but we assume SNR=1 for non-evoked data
        
        stcs = apply_inverse_epochs(
            epochs, inv, lambda2=lambda2, pick_ori="normal", return_generator=True,
            method="eLORETA"
        )

        label_ts = mne.extract_label_time_course(
            stcs, labels, src, return_generator=False
        )
        del stcs
        
        #%% create source epochs object
         
        n_epochs = (len(epochs))
        epoch_len = np.shape(epochs[0])[2]
        source_epochs_data = np.zeros((len(epochs), len(labels), np.shape(epochs[0])[2]))
        for s, stc_epoch in enumerate(label_ts):
            source_epochs_data[s,:,:] = stc_epoch
            
        # fake epoch object
        ch_names=[labels[i].name for i in range(len(labels))]
        epochs_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='misc')
        source_epochs = mne.EpochsArray(source_epochs_data, epochs_info)
        source_epochs.add_channels([stim_epochs], force_update_info=True)
        
        # now collapse into source raw
        df = source_epochs.to_data_frame()
        source_data = df.iloc[:,3:].to_numpy().transpose()
        del df
        source_raw = mne.io.RawArray(source_data, source_epochs.info)
        del source_epochs
        
        # extract data and reject annotations
        #stim_data = source_raw.copy().pick('stim').get_data()
        #source_data = source_raw.copy().pick('misc').get_data()
        #source_ortho = symmetric_orth(source_data)
        #source_ortho_raw = mne.io.RawArray(np.concatenate((source_ortho, stim_data), 0), source_raw.info)
        
        source_raw_fname = deriv_path.basename + "-source_raw.fif"
        source_raw.save(op.join(deriv_path.directory, source_raw_fname), overwrite=True)
        
