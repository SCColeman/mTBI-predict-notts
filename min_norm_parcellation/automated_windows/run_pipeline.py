# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:18:31 2024

@author: ppysc6
"""

import forward_model_auto
import preprocessing_auto
import extract_source_epochs_auto

subjects = ["2001"] * 4 + ["2003"] * 4 + ["2008"] * 4 + ["2009"] * 4
sessions = ["01N", "03N", "05N", "06N", "01N", "03N", "04N", "05N", "01N", 
            "03N", "04N", "05N", "02N", "03N", "04N", "05N"]
dates = ['20230616', '20230620', '20230703', '20230704', '20230718', '20230724', '20230725',
         '20230731', '20231026', '20231106', '20231107', '20231109', '20231113', '20231114',
         '20231120', '20231121']

for s, subject in enumerate(subjects):
    forward_model_auto.forward_model(subject, sessions[s])
    preprocessing_auto.preprocessing(subject, sessions[s], dates[s])
    extract_source_epochs_auto.source_recon(subject, sessions[s])
