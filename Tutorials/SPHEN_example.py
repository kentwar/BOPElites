# Required to access parent modules
import sys, os
import inspect
script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
parent_folder_path = os.path.abspath(os.path.join(script_path, "../.."))
sys.path.append(parent_folder_path)
#%%
from surrogates import GP
#from benchmarks.SyntheticGP10d2f import SyntheticGP10d2f
from benchmarks.robotarm import RobotArm
from archives.archives import structured_archive
from acq_functions.UCB import GPUCB
from acq_functions.mean import GPmean
from algorithm.SPHEN import algorithm
from optimizers import mapelites

import logging
import numpy as np
import torch

# Define problem instance, archive type, and optimizer
seed = 11
domain    = RobotArm(feature_resolution = [10,10])
QDarchive = structured_archive(domain)
optimizer = mapelites.MAPelites


# Initialize SAIL
SPHEN = algorithm(domain, QDarchive, GPUCB, optimizer, beta = 3,seed = seed, **{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,})

# Run SAIL
SPHEN.run(batchsize = 10, max_iter = 1250)

# Plot outputs
SPHEN.plot_archive2d()
SPHEN.plot_convergence()

## Generate prediction map
from tools.prediction_archive import prediction_archive
from acq_functions.mean import GPmean

PMoptimizer = mapelites.MAPelites
pred_archive = prediction_archive(SPHEN, PMoptimizer, GPmean,  **{'known_features' : False,})
# pred_archive.true_pred_archive is the true value
# pred_archive.pred_archive is the predicted value
SPHEN.plot_archive2d(archive = pred_archive.true_pred_archive)