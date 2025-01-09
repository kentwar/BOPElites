# Required to access parent modules
import sys, os
import inspect
import shutil
import argparse
import torch
script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
parent_folder_path = os.path.abspath(os.path.join(script_path, "../.."))
sys.path.append(parent_folder_path)

# %%
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mr', type=int, help='The maximum resolution',default=25)
parser.add_argument('--d', type=str, help='The problem Domain', default='robotarm')
parser.add_argument('--seed', type=int, help='Seed')
parser.add_argument('--n', type=int, help='number of points to restart from')
args = parser.parse_args()
try:
    max_res = args.mr
    if max_res == 10:
        max_n = 1000
    else:
        max_n = 1250
    domain = args.d
    seed = args.seed
    restart_n = args.n
    if seed == None:
        seed = torch.randint(100000, (1,)).item()

except:
    print('yes')
    max_res = 10
    max_n = 1000
    restart_n = -1
    domain = 'robotarm'
    seed = torch.random.seed()

from surrogates import GP
if domain == 'robotarm':
    from benchmarks.robotarm import RobotArm as _domain
if domain == 'rosenbrock6d':
    from benchmarks.rosenbrock6d import Rosenbrock6d as _domain
if domain == 'syntheticGP10d2f':
    from benchmarks.SyntheticGP10d2f import SyntheticGP10d2f as _domain
if domain == 'mishra':
    from benchmarks.mishra_bird_function import Mishra_bird_function as _domain
from archives.archives import structured_archive
from acq_functions.BOP_UKD_beta import BOP_UKD_beta_beta
from acq_functions.UKD_mean import UKD_mean
from algorithm.BOP_Elites_UKD_beta import algorithm
from optimizers import patternsearch_beta,patternsearch, mapelites 
import matplotlib.pyplot as plt
import numpy as np
import torch

#plt.switch_backend('agg')

## We configure the experiment by setting the domain, archive and optimizer
## The algorithm is imported from the algorithm folder, and we feed it the
## other modules.
resolutions = [[5,5],[max_res]*2]
domain    = _domain(feature_resolution = resolutions[-1], seed = seed)
QDarchive = structured_archive(domain)
optimizer = patternsearch_beta.PatternSearchOptimizer


# %%

## 'Known_features' defines whether we use a surrogate to model the descriptor
## space. 
## 'test_mode' tells the algorithm whether to store the data in the test
## folder or directly in the experiment folder.
## 'init_beta' is the initial value of beta in the BOP-Elites_beta algorithm (not needed in others)

BOP = algorithm(domain, QDarchive , BOP_UKD_beta_beta, optimizer , resolutions,seed = seed,  **{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : False,
                                                'init_beta' : 0,
                                                'stable_beta' : False,})
## Save version of this script
shutil.copy(__file__, f'{BOP.save_path}/experiment_script.py' ) 

# %%
## Plot initial data
#BOP.plot_archive2d(text = 'Initial_data')
#BOP.plot_convergence()


cwd = os.getcwd()
domain = BOP.domain.name
alg = BOP.acq_fun_eval.name
fdims = BOP.resolutions[-1]
BOP.load_data(f"{cwd}/experiment_data/{domain}/{alg}/{fdims}/{seed}", restart_n)
#BOP.load_data(load_path, n = 40)

## Run one iteration of BOP-Elites for 2 points (n_initial_points is 10d)
BOP.run(n_restarts = 10, max_iter = max_n)

## Plot data after two iterations
#BOP.plot_archive2d(text = str(max_n))
#BOP.plot_convergence()


# ## Generate prediction map
# from tools.prediction_archive import prediction_archive
# from acq_functions.mean import GPmean

# PMoptimizer = mapelites.MAPelites
# pred_archive = prediction_archive(BOP, PMoptimizer, GPmean,  **{'known_features' : False , 'return_pred' : True} )
# true_plot_archive = pred_archive.true_pred_archive
# plot_archive = pred_archive.pred_archive
# BOP.plot_archive2d(archive = true_plot_archive, text = 'True Predicted archive')
# BOP.plot_archive2d(archive = plot_archive, text = 'Predicted archive')
# %%
