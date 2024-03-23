# -*- coding: utf-8 -*-
"""
Create forward model based on FreeSurfer reconstruction, CTF MEG 

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os.path as op
import mne
from mne_bids import BIDSPath, read_raw_bids

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'

# scanning session info
subject = '2014'
session = '05N'
task = 'CRT'  # name of the task
run = '01'
suffix = 'meg'

bids_path = BIDSPath(subject=subject, session=session,
task=task, run=run, suffix=suffix, root=bids_root)

deriv_path = BIDSPath(subject=subject, session=session,
task=task, run=run, suffix=suffix, root=deriv_root)

# write derivatives path if it doesn't exist
if not op.exists(deriv_path.directory):
    deriv_path.mkdir()

#%% load data

data = read_raw_bids(bids_path=bids_path, verbose=False)
data.pick("mag")
info = data.info

#%% Get FS reconstruction for subject or use fsaverage for quick testing

#subjects_dir = op.dirname(mne.datasets.fetch_fsaverage(verbose=True))   # for fsaverage
subjects_dir = r'R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS'

# Name of the subject directory in FS subjects_dir
fs_subject = 'sub-' + subject
#fs_subject = "fsaverage"

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
coreg.set_grow_hair(0)
coreg.fit_icp(20)
coreg.omit_head_shape_points(5 / 1000)
coreg.fit_icp(20)
coreg.omit_head_shape_points(5 / 1000)
coreg.fit_icp(20)
coreg.omit_head_shape_points(5 / 1000)
coreg.fit_icp(20)
mne.viz.plot_alignment(data.info, trans=coreg.trans, **plot_kwargs)

trans_fname = deriv_path.basename + "-trans.fif"
coreg.trans.save(op.join(deriv_path.directory, trans_fname), overwrite=True)

#%% compute source space

# can change oct5 to other surface source space
src = mne.setup_source_space(
    fs_subject, spacing="oct6", add_dist=False, subjects_dir=subjects_dir)
src.plot(subjects_dir=subjects_dir)

src_fname = deriv_path.basename + "-src.fif"
src.save(op.join(deriv_path.directory, src_fname), overwrite=True)

#%% single shell conduction model

conductivity = (0.3,)
model = mne.make_bem_model(
    subject=fs_subject, ico=4,
    conductivity=conductivity,
    subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

bem_fname = deriv_path.basename + "-bem.fif"
mne.write_bem_solution(op.join(deriv_path.directory, bem_fname), 
                       bem, overwrite=True)

#%% forward solution

fwd = mne.make_forward_solution(
    info,
    trans=coreg.trans,
    src=src,
    bem=bem,
    meg=True,
    eeg=False
    )
print(fwd)

fwd_fname = deriv_path.basename + "-fwd.fif"
mne.write_forward_solution(op.join(deriv_path.directory, fwd_fname), 
                       fwd, overwrite=True)
