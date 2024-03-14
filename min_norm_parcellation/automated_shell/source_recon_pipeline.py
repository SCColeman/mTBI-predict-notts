# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:58:19 2024

@author: ppysc6
"""

from subprocess import call

subjects = [2001] * 4 + [2003] * 4 + [2008] * 4 + [2009] * 4
sessions = ["01N", "03N", "05N", "06N", "01N", "03N", "04N", "05N", "01N", 
            "03N", "04N", "05N", "02N", "03N", "04N", "05N"]
dates = [20230616, 20230620, 20230703, 20230704, 20230718, 20230724, 20230725,
         20230731, 20231026, 20231106, 20231107, 20231109, 20231113, 20231114,
         20231120, 20231121]

for s, subject in enumerate(subjects):
    session = sessions[s]
    date = dates[s]
    call(["python", "1_preprocessing_auto.py", subject, session, date])
    call(["python", "2_forward_model_auto.py", subject, session])
    call(["python", "3_extract_source_epochs_auto.py", subject, session])