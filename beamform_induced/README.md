# Source Reconstruction of Induced MEG Responses using an LCMV Beamformer
This pipeline is used to analyse CTF MEG data taken in Nottingham (SPMIC). Pre-processing, forward modelling, and source reconstruction are separated into three scripts which have each been written to allow for easy automation, e.g. running through bash. Data are already converted into BIDS here - this is easy to implement using MNE-BIDS functionality (https://mne.tools/mne-bids/stable/generated/mne_bids.write_raw_bids.html), but also easy to remove if not using BIDS format. The below describes each script in a lot of detail to allow easy usage by non-experienced programmers/scientists. For each script, each section corresponds to a section heading in the script. 

## Forward Modelling
Before beginning, note the line in the first section of the script:
`mne.viz.set_3d_options(depth_peeling=False, antialias=False)`
This seems to enable all the 3D plotting within MNE - without this line some of the features don't work properly on some computers.

##### Set up BIDS path
This section sets up the BIDS paths (directory trees containing all *subjects* and *sessions* in your dataset, with specific naming format) containing the data, as well as a separate BIDS path for *derivatives*, i.e., data that has been processed in some way. Whenever we save objects out, they must be saved into the derivatives BIDS path, **not** the data BIDS path. This can easily be replaced with your own paths if not using BIDS format. 

##### Load data
This section loads the MEG data specified in the previous section, using `read_raw_bids` from the mne_bids package. We are only using this data for the *info* object, containing meta-data stored in the MEG dataset. The info object can be simply obtained by typing `data.info`. Specifically, this info object is required because it contains a *montage*, i.e., a set of 3D coordinates corresponding to sensor locations, HPI coil locations, and digitised headshape points (taken using a polhemus or other digitisation methods). If you run `data.info`, you should see a line among the outputs that looks something like *"dig: 385 items (3 Cardinal, 382 Extra)"*, meaning the montage contains a digitisation of three HPI coils (used as fiducial markers) and 382 headshape points.

##### Get FS reconstruction for subject or use fsaverage for quick testing
