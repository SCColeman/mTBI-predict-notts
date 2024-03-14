# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:18:31 2024

@author: ppysc6
"""

import forward_model_auto
import preprocessing_auto
import extract_source_epochs_auto

subject = "2001"
session = "03N"
date = "20230620"

forward_model_auto.forward_model(subject, session)
preprocessing_auto.preprocessing(subject, session, date)
extract_source_epochs_auto.source_recon(subject, session)