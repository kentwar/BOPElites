# Required to access parent modules
import sys, os
import inspect
script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
parent_folder_path = os.path.abspath(os.path.join(script_path, "../.."))
sys.path.append(parent_folder_path)

#%%

from surrogates import GP
from benchmarks.parsec import Parsec
from archives.archives import structured_archive
from acq_functions.UCB_Parsec import GPUCB_Parsec
from acq_functions.mean import GPmean
from algorithm.SAIL_Parsec import algorithm
from optimizers import mapelites
import numpy as np
import torch

seed = np.random.randint(1000)
# Define problem instance, archive type, and optimizer
domain    = Parsec(feature_resolution = [25,25], seed = seed)
QDarchive = structured_archive(domain)
optimizer = mapelites.MAPelites


# Initialize SAIL
SAIL = algorithm(domain, QDarchive, GPUCB_Parsec, optimizer, beta = 10, seed = seed, **{'sampler' : 5, 
                                                'known_features' : True,
                                                'test_mode' : False,})

# Run SAIL
SAIL.run(batchsize = 10, max_iter = 1250)

# Plot outputs
SAIL.plot_archive2d()
SAIL.plot_convergence()

## Generate prediction map
from tools.prediction_archive import prediction_archive
from acq_functions.mean import GPmean

#PMoptimizer = mapelites.MAPelites
#pred_archive = prediction_archive(SAIL, PMoptimizer, GPmean,  **{'known_features' : True,})
#plot_archive = pred_archive.true_pred_archive
# plot_archive = pred_archive.pred_archive
#SAIL.plot_archive2d(archive = plot_archive)