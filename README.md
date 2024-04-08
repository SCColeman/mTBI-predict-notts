# MEG Analysis Pipelines

This repository contains several pipelines for MEG source reconstruction and analysis. The data used here were from a choice-reaction-time (CRT) paradigm, taken as part of the mTBI-predict project between Birmingham, Nottingham and Aston.

Author: Sebastian C. Coleman, Email: ppysc6@nottingham.ac.uk.

### Source-Space Induced Responses
The pipelines *beamform_induced* and *min_norm_induced* can be used to localise induced responses related to a choice-reaction time (CRT) task. Scripts should be ran in number order, starting with *1_forward_model.py*, followed by *2_preprocessing.py* and *3_peak_theta_response.py* (or any other script starting with 3). Each pipeline uses a different inverse method, namely LCMV beamforming and eLORETA minimum-norm inverse modelling.

### Parcellation Beamformer and Connectivity
The pipeline, *min_norm_parcellation*, uses an exact low resolution brain electromagnetic tomography (eLORETA) minimum-norm inverse solution to reconstruct activity in each parcel of an anatomical atlas. The reconstructed activity is then used to calculate classical amplitude envelope connectivity. The first two scripts, *1_forward_model.py* and *2_preprocessing.py* are functionally identical to the *min_norm_induced* pipeline.

