# MEG Analysis Pipelines

This repository contains two pipelines for MEG source reconstruction and analysis. The data used here were from a choice-reaction-time (CRT) paradigm, taken as part of the mTBI-predict project between Birmingham, Nottingham and Aston.

Author: Sebastian C. Coleman, Email: ppysc6@nottingham.ac.uk.

### Source-Space Induced Responses
The first pipeline, *LCMV_button_presses*, uses a linearly-constrained minimum-variance (LCMV) beamformer to localise induced responses related to button presses. Scripts should be ran in number order, starting with *1_forward_model.py*, followed by *2_preprocessing.py* and *3_beamform_button_presses.py*. Further information can be found in the README file in the pipeline folder.

### Parcellation Beamformer and Connectivity
The second pipeline, *min_norm_parcellation*, uses a minimum-norm estimate (MNE) inverse solution to reconstruct activity in each parcel of an anatomical atlas. The reconstructed activity is then used to calculate classical amplitude envelope connectivity, as well as a more novel approach which uses k-means clustering to calculate source-space MEG *microstates*, acting as a measure of dynamic functional connectivity. The first two scripts, *1_forward_model.py* and *2_preprocessing.py* are functionally identical as the first pipeline. The outputs from these two scripts can then be used in any of the other scripts to produce various connectivity results, either on the single subject or group level.

