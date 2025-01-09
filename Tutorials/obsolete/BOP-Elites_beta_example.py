# Required to access parent modules
import sys, os
import inspect
script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
parent_folder_path = os.path.abspath(os.path.join(script_path, "../.."))
sys.path.append(parent_folder_path)
#%%
from surrogates import GP
from benchmarks.robotarm import RobotArm
from archives.archives import structured_archive
from acq_functions.BOP_EI_beta import BOP_EI_beta
from algorithm.obselete.BOP_Elites_beta import algorithm
from acq_functions.mean import GPmean
from tools.prediction_archive import prediction_archive
from optimizers import patternsearch_beta, mapelites
import numpy as np
import torch

resolutions = [[5,5], [25,25]]
domain    = RobotArm(feature_resolution = resolutions[-1])
QDarchive = structured_archive(domain)
optimizer = patternsearch_beta.PatternSearchOptimizer


BOP = algorithm(domain, QDarchive , BOP_EI_beta, optimizer, resolutions = resolutions,
                                                 **{'sampler' : 5, 
                                                'known_features' : True,
                                                'test_mode' : False,
                                                'init_beta' : 50,})

x = torch.tensor(np.random.random((1,4)), dtype=torch.double)
## Check fitness
BOP.fitGP(x)
## Check acq_fun
BOP.acq_fun_eval.evaluate(x)
## Check it correctly finds a current elite
BOP.acq_fun_eval.findfstar(x)

## Load previous data 
#load_dir = 'experiment_data/robotarm/25/55413'
#load_dir = 'experiment_data/robotarm/25/95468'
#BOP.load_data(load_dir, 100)

# Plot initial data
#BOP.plot_archive2d()
#BOP.plot_convergence()

#Run one iteration of BOP-Elites
BOP.run(n_restarts = 25, max_iter = 41)

## Generate prediction map
from tools.prediction_archive import prediction_archive
from acq_functions.mean import GPmean
PMoptimizer = mapelites.MAPelites

pred_archive = prediction_archive(BOP, PMoptimizer, GPmean)
BOP.plot_archive2d(archive = pred_archive.pred_archive)