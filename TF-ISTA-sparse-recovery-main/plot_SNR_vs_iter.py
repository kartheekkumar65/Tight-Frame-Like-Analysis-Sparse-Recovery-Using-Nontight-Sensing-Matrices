    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:24:07 2022

@author: kartheek

To open up the PSNR vs iteration dictionaries and plot nice graphs
"""

import numpy as np
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from plotting_tools import *

#%%

parser = ArgumentParser(description='Plot-SNR-vs-Iterations')
parser.add_argument('--sparsity', type=float, default=0.3, help='% of non-zeros in the sparse vector')
parser.add_argument('--noise_level', type=float, default=0.001, help='Noise Level')
parser.add_argument('--device', type=str, default='cuda:0', help='The GPU id')
parser.add_argument('--sensing', type=str, default='normal', help='Sensing matrix type')
parser.add_argument('--maxIter', type=int, default=0, help='Number of iterations for the plotting')
args = parser.parse_args()

# step = 0.2
sparsity = args.sparsity 
noise_level = args.noise_level
sensing = args.sensing
xmax = args.maxIter

thr_dict = {0.001: 0.01, 0.005: 0.01, 0.003: 0.01, 0.01: 0.01, 0.03: 0.03, 0.05: 0.05, 0.1:0.1, 0:0.001}
iter_dict = {0.001: 10000, 0.005: 5000, 0.003: 5000, 0.01: 5000, 0.03: 3000, 0.05: 1000, 0.1:1000, 0:100000}

thr_ = thr_dict[noise_level]
numIter = iter_dict[noise_level]
# RSNR_dict['RSNR'] = SNR_mean_list

import scipy.io as sio
TF_RSNR_dict = sio.loadmat(f'stored_results/TF_ISTA_iter_{numIter}_noise_{noise_level:.3f}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
RTF_RSNR_dict = sio.loadmat(f'stored_results/RTF_ISTA_iter_{numIter}_noise_{noise_level:.3f}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
RSNR_dict = sio.loadmat(f'stored_results/ISTA_iter_{numIter}_noise_{noise_level:.3f}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')

F_TF_RSNR_dict = sio.loadmat(f'stored_results/TF_FISTA_iter_{numIter}_noise_{noise_level:.3f}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
F_RTF_RSNR_dict = sio.loadmat(f'stored_results/RTF_FISTA_iter_{numIter}_noise_{noise_level:.3f}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
F_RSNR_dict = sio.loadmat(f'stored_results/FISTA_iter_{numIter}_noise_{noise_level:.3f}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')

TF_RSNR = TF_RSNR_dict['RSNR'].reshape(-1,)
RTF_RSNR = RTF_RSNR_dict['RSNR'].reshape(-1,)
RSNR = RSNR_dict['RSNR'].reshape(-1,)

F_TF_RSNR = F_TF_RSNR_dict['RSNR'].reshape(-1,)
F_RTF_RSNR = F_RTF_RSNR_dict['RSNR'].reshape(-1,)
F_RSNR = F_RSNR_dict['RSNR'].reshape(-1,)

#%% 

plt.style.use(['science','ieee'])
plt.rcParams.update({"font.family": "sans","font.serif": ["cm"],"mathtext.fontset": "cm","font.size": 20}) 

plt.figure(figsize=(8,6))
ax = plt.gca()

if xmax == 0 :
    xmax =  max(len(TF_RSNR), len(RSNR), len(RTF_RSNR)) + 1000

legend = True

anotate = False

x1 = np.arange(len(RSNR))
plot_signal(x1, -RSNR, ax=ax,
    legend_label=r'ISTA', legend_show= legend,
    line_width=2, plot_colour='green', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -RSNR[-1] -2])

x2 = np.arange(len(TF_RSNR))
plot_signal(x2, -TF_RSNR, ax=ax,
    legend_label=r'TF-ISTA', legend_show= legend,
    line_width=2, plot_colour='blue', line_style='-', annotates = anotate, 
    annotation = r'$6.48$ s ', pos = [x2[-1], -TF_RSNR[-1]])

x3 = np.arange(len(RTF_RSNR))
plot_signal(x3, -RTF_RSNR, ax=ax,
    legend_label=r'RTF-ISTA', legend_show= legend,
    line_width=2, plot_colour='red', line_style='-', annotates = anotate, 
    annotation = r'$10.11$ s ', pos = [x2[-1], -RTF_RSNR[-1]])

x1 = np.arange(len(F_RSNR))
plot_signal(x1, -F_RSNR, ax=ax,
    legend_label=r'FISTA', legend_show= legend,
    line_width=2, plot_colour='green', line_style='--', annotates = anotate, 
    annotation = r'$15.87$ s ', pos = [x1[-1], -F_RSNR[-1] + 2])


x2 = np.arange(len(F_TF_RSNR))
plot_signal(x2, -F_TF_RSNR, ax=ax,
    legend_label=r'TF-FISTA', legend_show= legend,
    line_width=2, plot_colour='blue', line_style='--', annotates = anotate, 
    annotation = r'$3.41$ s ', pos = [x2[-1], -F_TF_RSNR[-1]])

x3 = np.arange(len(F_RTF_RSNR))
plot_signal(x3, -F_RTF_RSNR, ax=ax,
    xaxis_label=r'ITERATIONS', yaxis_label= r'NMSE [dB]',
    legend_label=r'RTF-FISTA', legend_show= legend,
    n_col=2, legend_loc='upper center',
    line_width=2, plot_colour='red', line_style='--',
    xlimits=[0,xmax], ylimits=[-40, 0] , annotates = anotate, 
    annotation = r'$3.51$ s ', pos = [x3[-1], -F_RTF_RSNR[-1]], save = f'figures/ISTA_noise_{noise_level}_sparsity_{sparsity}_sensing_{sensing}_{xmax}')


print('-'*20)



























































