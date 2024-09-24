#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 20:23:43 2021

@author: kartheekreddy

This file implements ISTA
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import uuid
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#%%

parser = ArgumentParser(description='ISTA')
parser.add_argument('--sparsity', type=float, default=0.3, help='% of non-zeros in the sparse vector')
parser.add_argument('--noise_level', type=float, default=0.001, help='Noise Level')
parser.add_argument('--device', type=str, default='cuda:0', help='The GPU id')
parser.add_argument('--sensing', type=str, default='normal', help='Sensing matrix type')
args = parser.parse_args()

# step = 0.2
sparsity = args.sparsity 
noise_level = args.noise_level
sensing = args.sensing

thr_dict = {0.001: 0.01, 0.005: 0.01, 0.003: 0.01, 0.01: 0.01, 0.03: 0.03, 0.05: 0.05, 0.1:0.1, 0:0.001}
iter_dict = {0.001: 10000, 0.005: 5000, 0.003: 5000, 0.01: 5000, 0.03: 3000, 0.05: 1000, 0.1:1000, 0:100000}

thr_ = thr_dict[noise_level]
numIter = iter_dict[noise_level]

def soft_thr(input_, theta_):
    return F.relu(input_-theta_)-F.relu(-input_-theta_)


class ISTA(nn.Module):
    def __init__(self, m, n, Dict, numIter, alpha, device):
        super(ISTA, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.numIter = numIter
        self.A = Dict
        self.alpha = alpha
        self.device = device
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        alpha = self.alpha
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(A.T, A))
        
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*A.T)
        B = B.float().to(self.device)
        
        thr = torch.ones(1, 1) * thr_ / alpha
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.S = S
        self.W = B
        self.thr.data = nn.Parameter(thr.to(self.device))


    def forward(self, y):
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        
        import time
        start = time.time()
        time_list = []
        # time_list.append(time.time() - start)
        x.append(d)
        
        for iter in range(self.numIter):
            d_old = d
            # d = soft_thr(self._W(y) + self._S(d), self.thr)
            d = soft_thr(torch.mm(self.S, d.T).T + torch.mm(self.W, y.T).T, self.thr)
            x.append(d)
            
            time_list.append(time.time() - start)
            if torch.norm(d - d_old) < 1e-4:
                break
        return x, time_list
    
#%%

def test_(net, Y, D, device):
    
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(D.T)
    D_t = D_t.float().to(device)

    ratio = 1
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_list, time_list = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_list = X_list.view(-1)
        X_final = X_list[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_list, time_list
#%%

def pes(x,x_est):
  d = []
  for i in range(x.shape[1]):
    M = max(np.sum(x[:,i] != 0),np.sum(x_est[:,i] != 0))
    pes_ = (M - np.sum((x[:,i]!=0) * (x_est[:,i]!=0)))/M
    if not np.isnan(pes_):
        d.append(pes_)
    else:
        print(M)
        print('nan is found here')
  return np.mean(d),np.std(d)

def data_gen( D, noise_Std_Dev, k, p, rng):
  m, n = D.shape
  x = rng.normal(0,1,(n,p))*rng.binomial(1,k,(n,p))
  
  y = D@x 
  noise = rng.normal(0,noise_Std_Dev,y.shape)
  y = y + noise

  return x, y

#%%
seed = 80
print('seed: ', seed)
rng = np.random.RandomState(seed)

m = 70; n = 100;
# create the random matrix

if sensing == 'normal':
    D = rng.normal(0, 1/np.sqrt(m), [m, n])
    D /= np.linalg.norm(D,2,axis=0)
    
elif sensing == 'bernoulli':
    D = rng.normal(0, 1/np.sqrt(m), [m, n])
    D[D <= 0] = -1; D[D > 0] = 1
    D /= np.linalg.norm(D,2,axis=0)

elif sensing == 'laplacian':
    D = rng.laplace(0, 1/np.sqrt(m), [m, n])
    D /= np.linalg.norm(D,2,axis=0)

elif sensing == 'uniform':
    D = rng.uniform(-1, 1, [m, n])
    D /= np.linalg.norm(D,2,axis=0)

elif sensing == 'exponential':
    D = rng.exponential(1/np.sqrt(m), [m, n])
    D /= np.linalg.norm(D,2,axis=0)

numTrain = 100000; numTest = 1000
X_train, Y_train = data_gen( D, noise_level, sparsity, numTrain, rng)
X_test, Y_test = data_gen( D, noise_level, sparsity, numTest, rng)


#%%
# retrieve the Signal from Y and D using LISTA using true ground truth

numLayers = numIter

device = 'cpu'

import time
start = time.time()

import datetime
t = datetime.datetime.now().timetuple();

alpha = (np.linalg.norm(D, 2) ** 2 )*1.001
net = ISTA(m, n, D, numLayers, alpha = alpha, device = device)
net.weights_init()


#%%    
# Sparse Estimation using LISTA (testing phase)

PES_list = []; SNR_mean_list = []
SNR_list = [];

import time
start = time.time()

X_out, X, time_list = test_(net, Y_test, D, device)
end = time.time()
print(f'time elapsed {round(end - start, 2)} seconds' )

N_test = Y_test.shape[1]
for i in range(N_test):
    err = np.linalg.norm(X_out[:, i] - X_test[:, i])
    RSNR = 20*np.log10(np.linalg.norm(X_test[:, i])/err)
    if np.isnan(RSNR):
        break
    SNR_list.append(RSNR)

SNR_list = np.array(SNR_list)
PES_mean, PES_std = pes(X_test, X_out)

print('Testing: my ISTA avg SNR is ', np.mean(SNR_list))
print('Testing: my ISTA std SNR is ', np.std(SNR_list))
print('Testing: my ISTA avg PES is ', PES_mean)
print('Testing: my ISTA std PES is ', PES_std)

#%% save results

SNR_mean_list = []; RSNR_dict = {}

with torch.no_grad():
    loss= []
    y = torch.tensor(Y_test, dtype=(torch.float32))
    D_t = torch.tensor(D, dtype=(torch.float32))
    for x_b in X:
        x_b = x_b.cpu().T
        err = np.linalg.norm(x_b - X_test, 'fro')
        RSNR = 20*np.log10(np.linalg.norm(X_test, 'fro')/err)
        SNR_mean_list.append(RSNR)

RSNR_dict['RSNR'] = SNR_mean_list

import scipy.io as sio
sio.savemat(f'stored_results/ISTA_iter_{numIter}_noise_{noise_level:.3f}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations', RSNR_dict)

output_file_name = f"./log/ISTA_iter_{numIter}_noise_{noise_level}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations.txt"

output_data = f"thr: {thr_} RSNR: {np.mean(SNR_list): .4f}, {np.std(SNR_list): .4f}, PES: {PES_mean: .4f}, {PES_std: .4f}\n"

output_file = open(output_file_name, 'a')
output_file.write(output_data)
output_file.write(f'time elapsed {round(end - start, 2)} seconds' )
output_file.close()

print('-'*20)

















