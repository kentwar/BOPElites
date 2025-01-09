import numpy as np
import torch, sys, os, inspect, shutil, argparse
import matplotlib.pyplot as plt
# Neccesary to load the main BOP-Elites code
script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
parent_folder_path = os.path.abspath(os.path.join(script_path, "../.."))
sys.path.append(parent_folder_path)
np.allow_pickle = True
hp = '/home/rawsys/matdrm/PhD_code/BOP-Elites/experiment_data/plotting/final_results/'

A = np.load(f'BOPParsec_pred_25.npy', allow_pickle=True).item()
B = np.load(f'BOPParsec_true_25.npy', allow_pickle=True)

A = np.load(f'{hp}BOPParsec_pred_25.npy', allow_pickle=True).item()
for key in A.keys():
    if key != 0:
        A[key] = A[key] + 5
B = np.load(f'{hp}BOPParsec_true_25.npy', allow_pickle=True)

import scienceplots
plt.style.use(['science','grid'])

maxval = 2215.7217
fig = plt.figure(figsize=(8,3), dpi = 300)
Amean = torch.stack(list(A.values() )).mean(dim = 1) 
Aster = torch.stack(list(A.values() )).std(dim = 1)/np.sqrt(len(A))
Ax = np.array(list(A.keys())) + 1e-6
plt.scatter(Ax, maxval - Amean, label = 'BOP-Elites PM', marker = 'o', s = 10, color = 'purple')
plt.plot(Ax, maxval - Amean, alpha = 0.2, color = 'purple')
# #plt.xscale('log')
plt.fill_between(A.keys(), maxval - Amean - Aster, maxval - Amean + Aster, alpha = 0.2, color = 'purple')
b = maxval - np.mean(B, axis = 0)
bsem = np.std(B, axis = 0)/np.sqrt(B.shape[0])
plt.plot(range(len(b)), b, color = 'blue',  label = 'BOP-Elites')
plt.fill_between(range(len(b)), b - bsem, b + bsem, color = 'blue', alpha = 0.2)

#plt.xscale('log')

# plt.xlim([40,1250])
# plt.vlines(40,0,550, color = 'black', linestyle = '--')


# A = np.load(f'{hp}SPHENmishra_bird_function_pred_25.npy', allow_pickle=True).item()
# B = np.load(f'{hp}SPHENmishra_bird_function_true_25.npy', allow_pickle=True)

# Amean = torch.stack(list(A.values() )).mean(dim = 1) 
# Aster = torch.stack(list(A.values() )).std(dim = 1)/np.sqrt(len(A))
# Ax = np.array(list(A.keys())) + 1e-6
# plt.scatter(Ax, maxval - Amean, label = 'SPHEN PM', marker = 'x', s = 10, color = 'red')
# plt.plot(Ax, maxval - Amean, alpha = 0.2)
# #plt.xscale('log')
# plt.fill_between(A.keys(), maxval - Amean - Aster, maxval - Amean + Aster, alpha = 0.2, color = 'red')
# b = maxval - np.mean(B, axis = 0)
# bsem = np.std(B, axis = 0)/np.sqrt(B.shape[0])
# plt.plot(range(len(b)), b, color = 'red',  label = 'SPHEN')
# plt.fill_between(range(len(b)), b - bsem, b + bsem, color = 'red', alpha = 0.2)
# #plt.xscale('log')
# plt.yscale('log')
# #plt.title('Mishra Bird Function [25 x 25] performance, distance from optima')
# plt.xlabel('Number of evaluations')
# plt.ylabel('Log distance to optima')



import csv

old_dir = '/home/rawsys/matdrm/pySAIL_0.5/Analysis_Scripts/PlottingData/SOBOL/parsec/25/'
A = []
for dir in list(os.walk(old_dir))[0][2]:
    np.loadtxt(old_dir + dir, delimiter=" ", dtype=str)
    arr = np.genfromtxt(old_dir + dir,
                    delimiter=" ", dtype=np.double)
    A.append(arr)
A = np.array(A)
B = A.mean(axis = 0)[:1250]
b = maxval - B
bsem = np.std(B, axis = 0)/np.sqrt(B.shape[0])
plt.plot(range(len(b)), b, color = 'orange',  label = 'SOBOL')
plt.fill_between(range(len(b)), b - bsem, b + bsem, color = 'orange', alpha = 0.2)
#plt.xscale('log')
plt.yscale('log')
#plt.title('Robot Arm [25 x 25] performance, distance from optima')
plt.xlabel('Number of evaluations')
plt.ylabel('Log distance to optima')
# Make legend slightly transparent





old_dir = '/home/rawsys/matdrm/pySAIL_0.5/Analysis_Scripts/PlottingData/ME/parsec/25/'

A = []
for dir in list(os.walk(old_dir))[0][2]:
    np.loadtxt(old_dir + dir, delimiter=" ", dtype=str)
    arr = np.genfromtxt(old_dir + dir,
                    delimiter=" ", dtype=np.double)
    A.append(arr)
A = np.array(A)
B = A.mean(axis = 0)[:1250]
b = maxval - B
bsem = np.std(B, axis = 0)/np.sqrt(B.shape[0])
plt.plot(range(len(b)), b, color = 'green',  label = 'Map-Elites')
plt.fill_between(range(len(b)), b - bsem, b + bsem, color = 'green', alpha = 0.2)
#plt.xscale('log')
plt.yscale('log')
plt.title('Parsec 10d [25 x 25] performance, distance from optima')
plt.xlabel('Number of evaluations')
plt.ylabel('Log distance to optima')
plt.legend( ncols = 2)

