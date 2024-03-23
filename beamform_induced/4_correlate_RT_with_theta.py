# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:29:35 2024

@author: ppysc6
"""

import os.path as op
import os
import numpy as np
import mne
from matplotlib import pyplot as plt
from mne_bids import BIDSPath, read_raw_bids

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% set up BIDS path

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'
deriv_root = r'R:\DRS-mTBI\Seb\mTBI_predict\derivatives'

#%% load reaction times

RTs_lh = []
RTs_rh = []
resp_lh = []
resp_rh = []
# scanning session info
for subject in ['2001', '2003', '2008', '2009', '2014']:
    
    # each session
    sessions = os.listdir(op.join(deriv_root, "sub-" + subject))
    sessions = [sessions[i][4:] for i in range(len(sessions))]
    
    if subject=='2014':
        sessions = sessions[1:]
    
    for session in sessions:
        
        task = 'CRT'  # name of the task
        run = '01'
        suffix = 'meg'
        
        bids_path = BIDSPath(subject=subject, session=session,
        task=task, run=run, suffix=suffix, root=bids_root)
        
        deriv_path = BIDSPath(subject=subject, session=session,
        task=task, run=run, suffix=suffix, root=deriv_root)
        
        RT_fname = deriv_path.basename + "-avg_RT_lh.npy"
        RTs_lh.append(np.load(op.join(deriv_path.directory, RT_fname)))
        RT_fname = deriv_path.basename + "-avg_RT_rh.npy"
        RTs_rh.append(np.load(op.join(deriv_path.directory, RT_fname)))
        
        resp_fname = deriv_path.basename + "-lh_stim_theta.npy"
        resp_lh.append(np.load(op.join(deriv_path.directory, resp_fname)))
        resp_fname = deriv_path.basename + "-rh_stim_theta.npy"
        resp_rh.append(np.load(op.join(deriv_path.directory, resp_fname)))
        
RTs_lh = np.array(RTs_lh)
RTs_rh = np.array(RTs_rh)

#%% take mean timecourses and get MRBD

### mean timecourse
mean_lh = np.mean(resp_lh, axis=0)
mean_rh = np.mean(resp_rh, axis=0)

time = np.linspace(-0.8, 1.2, 1201)

plt.figure()
plt.plot(time, mean_rh, color="black")
plt.xlim([-0.75, 1.15])
plt.ylabel("Oscillatory Power (A.U)")
plt.xlabel("Time (s)")

plt.figure()
plt.plot(time, mean_lh, color="black")
plt.xlim([-0.75, 1.15])
plt.ylabel("Oscillatory Power (A.U)")
plt.xlabel("Time (s)")

win_lh = np.array([RTs_lh-0.1, RTs_lh+0.1]).transpose()
win_lh_i = np.array([0.8*600 + win_lh[:,0]*600, 0.8*600 + win_lh[:,1]*600]).astype(int).transpose()
win_rh = np.array([RTs_rh-0.1, RTs_rh+0.1]).transpose()
win_rh_i = np.array([0.8*600 + win_rh[:,0]*600, 0.8*600 + win_rh[:,1]*600]).astype(int).transpose()
win_lh = [resp_lh[i][win_lh_i[i,0]:win_lh_i[i,1]] for i in range(len(resp_lh))]
win_rh = [resp_rh[i][win_rh_i[i,0]:win_rh_i[i,1]] for i in range(len(resp_rh))]

MRBD_lh = np.mean(win_lh, axis=1)
MRBD_rh = np.mean(win_lh, axis=1)

#%% correlate with RT

import statsmodels.api as sm
import pandas as pd
import seaborn as sns

def slr_plot(X, Y, plot=True):
    results = sm.OLS(Y, sm.add_constant(X)).fit()
    c, m = results.params
    y_model = (m * X) + c
    p = results.f_pvalue
    rsq = results.rsquared
    
    if plot:
        fig, ax = plt.subplots(figsize = ( 4 , 3.5 ))
        df = pd.DataFrame()
        df['X'] = X
        df['Y'] = Y
        df['y_model'] = y_model
        sns.scatterplot(df, x='X', y='Y')
        sns.lineplot(df, x='X', y='y_model', color='r')
        ax.set(title='R^2 = ' + str("%.2f" % rsq) + ' , p = ' + str("%.2f" % p))
        plt.show()
    else:
        fig = []
        ax = []
    return rsq, p, fig, ax

# left hand
x = MRBD_lh
y = RTs_lh.copy()
sns.set()
rsq, p, fig, ax = slr_plot(x, y, True)
ax.set(ylabel='Reaction Times (s)', xlabel='LH Theta (A.U)')
plt.tight_layout()

# left hand
x = MRBD_rh
y = RTs_rh.copy()
sns.set()
rsq, p, fig, ax = slr_plot(x, y, True)
ax.set(ylabel='Reaction Times (s)', xlabel='RH Theta (A.U)')
plt.tight_layout()
