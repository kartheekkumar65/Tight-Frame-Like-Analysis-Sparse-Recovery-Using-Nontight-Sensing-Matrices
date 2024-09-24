#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:08:41 2022

@author: kartheek
"""

import numpy as np
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from plotting_tools import *



plt.style.use(['science','ieee'])
plt.rcParams.update({"font.family": "sans","font.serif": ["cm"],"mathtext.fontset": "cm","font.size": 26}) 


plt.figure(figsize=(8,6))
ax = plt.gca()

xmax = 6000 # max(len(PTF_RSNR), len(RSNR), len(UNTF_RSNR)) + 1000

anotate = False

# sensing_mats = ['Normal', 'Bernoulli', 'Laplacian', 'Uniform']
sensing_mats = [ 'N', 'B', 'L', 'U']

SNR_ISTA_mean = [30.06, 30.00, 30.00, 30.37];
SNR_ISTA_std  = [5.42, 5.56, 5.52, 5.32];
time_ISTA     = [12.44, 14.14, 9.76, 8.8]

SNR_TF_ISTA_mean = [33.55, 33.47, 33.47, 33.66];
SNR_TF_ISTA_std  = [3.80, 4.04, 4.05, 3.91];
time_TF_ISTA     = [3.74, 4.46, 3.37, 2.81]

SNR_CTF_ISTA_mean = [36.52, 36.36, 36.43, 36.60];
SNR_CTF_ISTA_std  = [3.94, 4.47, 4.33, 4.20];
time_CTF_ISTA     = [5.52, 6.28, 4.92, 4.75]

time_FISTA = [4.92, 6.59, 4.94, 4.46]
time_TF_FISTA = [1.66, 2.23, 1.57, 1.73]
time_CTF_FISTA = [1.85, 3.59, 2.57, 2.17]

# x1 = np.arange(len(RSNR))
# plt.errorbar(estimate, tk, xerr=devs, fmt=marker_style, color=plot_colour,
#         ms=marker_size, capsize=10, alpha=0.8, label=legend_label)

# plt.errorbar( x, SNR_ISTA_mean, SNR_ISTA_std, linestyle = ' ' , capsize=10, alpha=0.8, label='ISTA')
# plt.errorbar( x, SNR_TF_ISTA_mean, yerr=SNR_TF_ISTA_std,  linestyle = ' ' , capsize=10, alpha=0.8, label='TF-ISTA')
# plt.errorbar( x, SNR_CTF_ISTA_mean, yerr=SNR_CTF_ISTA_std , linestyle = ' ' , capsize=10, alpha=0.8, label='CTF-ISTA')
# plt.ylim([20, 45])
# plt.legend()

plt.figure(figsize=(8,6))
ax = plt.gca()
s = 25 * 3

# plot_points(xpoints, ypoints, ax=None, line_width=None, point_colour='black',
#     alpha=1, legend_label=None, legend_show=True, marker_style='o',
#     legend_loc='lower left', title_text=None, show=False,
#     xlimits=[0,1], ylimits=[-1,1], save=None):

x = np.arange(4)
plot_points(x, time_ISTA, ax=ax,  point_colour= (0, 0, 1),
    legend_label=r'ISTA', marker_style='.', size = s*np.ones(4, ) )
plot_points(x, time_FISTA, ax=ax,  point_colour= (0, 1, 0.5 ),
    legend_label=r'FISTA', marker_style='o' , size = s*np.ones(4, ) )
plot_points(x, time_TF_ISTA, ax=ax,  point_colour='g',
    legend_label=r'TF-ISTA', marker_style='^' , size = s*np.ones(4, ) )
plot_points(x, time_TF_FISTA, ax=ax,  point_colour=(0.5, 0.2, 0.5),
    legend_label=r'TF-FISTA', marker_style='v' , size = s*np.ones(4, ) )
plot_points(x, time_CTF_ISTA, ax=ax,  point_colour= 'r',
    legend_label=r'RTF-ISTA', marker_style='+' , size = s*np.ones(4, ) )


plot_points(x, time_CTF_FISTA, ax=ax,  point_colour= (1, 0.5, 0.5) , size = s*np.ones(4, ) 
    , marker_style='x',  xaxis_label=r'SENSING MATRIX', yaxis_label=r'TIME (S)',
    legend_label=r'RTF-FISTA', ylimits=[0, 25], xlimits = [-1, 4],
    xticks = sensing_mats,legend_loc='upper center' , save = 'figures/time_sensing')