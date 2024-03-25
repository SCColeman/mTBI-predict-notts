# Source Reconstruction of Induced MEG Responses using an LCMV Beamformer
This pipeline is used to analyse CTF MEG data taken in Nottingham (SPMIC). Pre-processing, forward modelling, and source reconstruction are separated into three scripts which have each been written to allow for easy automation, e.g. running through bash. Data are already converted into BIDS here - this is easy to implement using MNE-BIDS functionality (https://mne.tools/mne-bids/stable/generated/mne_bids.write_raw_bids.html), but also easy to remove if not using BIDS format. The below describes each script in a lot of detail to allow easy usage by non-experienced programmers/scientists. For each script, each section corresponds to a section heading in the script. 

## Forward Modelling
The components required for a MEG forward model are:
1) A segmented MRI, produced using FreeSurfer (see FreeSurfer section)
2) A transformation matrix (`trans`) that is used to map the segmented MRI to the space defined by the MEG sensors (often called co-registration)
3) A source-space (`src`), which is a set of coordinates within the segmented MRI that will be used to model magnetic dipoles
4) A conduction model (`bem`), which uses the segmented MRI to estimate field conduction through the head volume. 

A forward solution is then calculated using the above components. The outcome is a set of *lead fields* (`fwd`), which estimate how fields produced by dipoles at each of the source-space coordinates project onto the array of sensors.

The following sections will describe how each of these components are produced using the `1_forward_model.py` script.

Before beginning, note the line in the first section of the script:
`mne.viz.set_3d_options(depth_peeling=False, antialias=False)`
This seems to enable all the 3D plotting within MNE - without this line some of the features don't work properly on some computers.

##### Setting up paths
In this pipeline we use BIDS paths (directory trees containing all *subjects* and *sessions* in your dataset, with specific naming format) containing the data, as well as a separate BIDS path for *derivatives*, i.e., data that has been processed in some way. These folder structures are set using the `mne_bids` function `BIDSPath`, which requires the user to set values for `subject` (e.g. '05'), `session` (e.g. '01'), `task` (e.g. 'CRT', which stands for choice-reaction-time), `run` (e.g. '01') and `suffix` (often meaning the modality, e.g. 'meg'). A folder structure using the above examples would go as **"...BIDS_root\sub-05\ses-01\meg"** and an example file name in the folder would go as **"sub-2002_ses-03N_task-EmoFace_run-01_meg.fif"**.

Whenever we save objects out, they must be saved into the derivatives BIDS path, e.g. **"...derivatives_root\sub-05\ses-01\meg"**, **not** the data BIDS path. This can easily be replaced with your own paths if not using BIDS format by simply setting `data_path = r"path\to\your\data"`. 

##### Loading data
Data is loaded from the BIDS paths using `read_raw_bids` from the mne_bids package. For forward modelling, We are only using this data for the `info` object, containing meta-data stored in the MEG dataset. The info object can be simply obtained by typing `data.info`. Specifically, this info object is required because it contains a *montage*, i.e., a set of 3D coordinates corresponding to sensor locations, HPI coil locations, and digitised headshape points (taken using a polhemus or other digitisation methods). If you run `data.info`, you should see a line among the outputs that looks something like *"dig: 385 items (3 Cardinal, 482 Extra)"*, meaning the montage contains a digitisation of three HPI coils (used as fiducial markers) and 482 headshape points.

![The Info Object](URL "The Info Object")

##### Load FreeSurfer Files
Next we load the outputs from FreeSurfer. These are contained in the `subjects_dir` directory. For more information on this, see the README file in **freesurfer_recon**. If you have a FreeSurfer reconstruction for the subject, set `subjects_dir = r"path\to\your\freesurfer\subjects_dir"` and then set `fs_subject` to a string corresponding to the name of the subject folder in `subjects_dir`, e.g. "sub-01". If you do not have a FreeSurfer reconstruction for the subject MRI, set `subjects_dir = op.dirname(mne.datasets.fetch_fsaverage(verbose=True))`, and set `fs_subject = "fsaverage"`, which allows you to use a template freesurfer reconstruction from the MNI-152 brain. However, note that **fsaverage should only be used for testing, not proper study results**.


![fsaverage](URL)

##### Co-Registration
Next we need to obtain a transformation matrix that maps the segmented MRI, loaded in the last section, to the position of the head in the MEG helmet. Here we take an automated approach, which works pretty well for most cases, but does not allow manual tweaking. 

The coregistration object, `coreg`, is created using `mne.coreg.Coregistration`, taking the `info` object (containing the digitisation), as well as the FreeSurfer folder and subject (containing the segmented MRI). Once this object is created, we can apply a series of *methods* (class-specific functions) to perform the coregistration.

Firstly, a crude transformation is performed by simply matching the *fiducial* positions in both the MRI and the digitisation. This is performed by running `coreg.fit_fiducials()`. The fiducial points in the digitisation have a position which is determined by the HPI coils in the MEG, i.e., these remain fixed as this was where the head was positioned during the scan. The fiducial positions on the MRI are automatically estimated using the corresponding positions on fsaverage, and are therefore very approximate, but given that we will refine the transformation later, this is not too important. 

Next, an iterative closest point (ICP) algorithm is applied to match the surface of the MRI to the headshape points in the digitisation. To do this, we run `coreg.fit_icp()` with a single argument which is the number of iterations to perform - technically the more the better, but the algorithm will automatically stop at convergence (20 is usually enough).

Here, we may want to remove outlier points in the digitisation that may have thrown off the ICP. To do this, we run `coreg.omit_head_shape_points()` with a single argument which is the distance threshold for point omission in m, e.g. 5/1000 for 5mm threshold. Following this, further ICP can be performed to refine the coregistration. 

An image of the MRI and MEG sensor array before and after coregistration is shown below.

![coreg](URL)

##### Computing a Source Space
Computing a source space is simple, but requires several choices that can impact your results. In general there are two options: surface source spaces and volume source spaces. Surface source spaces only define source space coordinates on the cortical surface, whereas volume source spaces define source space coordinates in a regular grid across the entire brain. Surface source spaces make it easier to work with *cortical parcellations*, as these parcellations are defined using the cortical surface. Surface source spaces also allow for more accurate dipole modelling in minimum-norm inverse solutions. However, volume source spaces are better at identifying deep sources. Here, we use surface source spaces to allow for easy integration with parcellations.

The main parameter we need to choose for a surface source space is `spacing`, which we set as `"oct6"`. This can be switched to `"oct5"` for a sparser (and therefore faster) source space.

![sourcespace](URL)

##### Computing the Conduction Model
Next, a conduction model is calculated using the boundary element method (BEM). This takes the segmented MRI from FreeSurfer, along with approximate conductivity values for each of the layers (brain, skull, scalp) to create a conduction model. MEG signals pass largely undistorted through the boundaries between the different tissue layers, so we can use a *single-shell* model which treats the entire head as uniform conductor. Therefore, we only supply a single value to the `conductivity` parameter, which is a standard value and should not be changed.

##### Computing the Forward Solution
Now all components have been loaded or calculated, a forward model can be computed. Ensure that the modality is set correctly (either `meg=True` or `eeg=True`). The forward model object `fwd` contains the lead fields which are used later to calculate an inverse solution.

## Pre-Processing
The goals of pre-processing MEG data (in this case) are as follows:
1) Remove high and low frequency noise using synthetic third-order gradiometry and bandpass filtering
2) Annotate periods of data that are corrupted by noise, caused by SQUID sensor "resets", head movement, and muscle artefacts
3) Perform ICA on the data to remove blink and cardiac artefacts

As opposed to other pre-processing procedures, this one is designed to be as general as possible, i.e., avoiding narrowband filtering and epoching, to allow for a lot of flexibility in later analysis steps. In addition, this pre-processing procedure (along with forward modelling and beamforming) is completely automated, requiring no user input beyond the scanning details (which can be inputted quite simply using bash etc).

#####



