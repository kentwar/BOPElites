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
from acq_functions.BOP_EI_Parsec import BOP_EI_Parsec
from algorithm.BOP_Elites_Parsec import algorithm
from optimizers import patternsearch , mapelites

import numpy as np
import torch

## Progressive upscaling requires providing a list of resolutions
resolutions = [[5,5], [25,25]]
domain    = Parsec(feature_resolution = resolutions[0])
QDarchive = structured_archive(domain)
optimizer = patternsearch.PatternSearchOptimizer

BOP = algorithm(domain, 
                QDarchive , 
                BOP_EI_Parsec, 
                optimizer, 
                **{'known_features' : True,
                'test_mode' : True,
                'resolutions' : resolutions,
                          }  
                )


#Plot initial data
BOP.plot_archive2d()
BOP.plot_convergence()

#Run one iteration of BOP-Elites
BOP.run(n_restarts = 10, max_iter = 1250)

# ## Generate prediction map
# from tools.prediction_archive import prediction_archive
# from acq_functions.mean import GPmean

# PMoptimizer = mapelites.MAPelites
# pred_archive = prediction_archive(BOP, PMoptimizer, GPmean,  **{'known_features' : True ,
#                                                             'return_pred' : True}, )
# true_plot_archive = pred_archive.true_pred_archive
# plot_archive = pred_archive.pred_archive
# BOP.plot_archive2d(archive = true_plot_archive, text = 'True Predicted archive')
# BOP.plot_archive2d(archive = plot_archive, text = 'Predicted archive')


# %%
cur_dir = os.getcwd()
# %%
