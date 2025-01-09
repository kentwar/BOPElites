# Required to access parent modules
import sys, os
script_path = os.path.abspath(__file__)
parent_folder_path = os.path.abspath(os.path.join(script_path, "../.."))
sys.path.append(parent_folder_path)

# %%
from surrogates import GP
from benchmarks.robotarm import RobotArm
from archives.archives import structured_archive
from acq_functions.BOP_UKD_vec import BOP_UKD_vec
#from acq_functions.BOP_UKD_beta_single import BOP_UKD_beta_single
from acq_functions.UKD_mean import UKD_mean
from algorithm.obselete.BOP_Elites_UKD_vec import algorithm
from optimizers import  mapelites , LBFGSB
import matplotlib.pyplot as plt
import numpy as np
import torch

#plt.switch_backend('agg')

## We configure the experiment by setting the domain, archive and optimizer
## The algorithm is imported from the algorithm folder, and we feed it the
## other modules.

domain    = RobotArm(feature_resolution = [5,5])
QDarchive = structured_archive(domain)
optimizer = LBFGSB.LBFGSB_optimizer



## 'Known_features' defines whether we use a surrogate to model the descriptor
## space. 
## 'test_mode' tells the algorithm whether to store the data in the test
## folder or directly in the experiment folder.
## 'init_beta' is the initial value of beta in the BOP-Elites_beta algorithm (not needed in others)

BOP = algorithm(domain, QDarchive , BOP_UKD_vec, optimizer , **{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                'init_beta' : 10,})
# %%
BOP.run(n_restarts = 4, max_iter = 1250)
# %%
## Plot initial data
#BOP.plot_archive2d(text = 'Initial_data')
#BOP.plot_convergence()

## Run one iteration of BOP-Elites for 2 points (n_initial_points is 10d)
#BOP.run(n_restarts = 10, max_iter = 50)

## Plot data after two iterations
BOP.plot_archive2d(text = '1250')
BOP.plot_convergence()

# %%
## We can perform 'upscaling' by running the algorithm on a higher resolution
## We adjust the algorithm and run some more

# upscaled_resolution = [10,10]
# BOP.upscale(structured_archive, upscaled_resolution)
# BOP.run(n_restarts = 10, max_iter = 44)

# ## Plot data after four iterations
# BOP.plot_archive2d()
# BOP.plot_convergence()

# %%

# upscaled_resolution = [25,25]

# BOP.upscale(structured_archive, upscaled_resolution)
#BOP.run(n_restarts = 10, max_iter = 46)

# ## Plot data after six iterations
# BOP.plot_archive2d()
# BOP.plot_convergence()

# %%

## We access modules of the algorithm from within the algorithm itself. Lets
## check the fitness prediction of the algorithm.
#  
# x = torch.tensor(np.random.random((1,domain.xdims)), dtype=torch.double) # Random input

# ## Check fitness predictions
# true_fitness = BOP.domain.fitness_fun(x.numpy().reshape(-1,domain.xdims))
# fitness_prediction = BOP.predict_fit(x).item()
# print('True fitness: ', true_fitness)
# print('Fitness prediction: ', fitness_prediction)

# %%

## We can load previous data for analysis, reading in n observations from the 
## saved run. 
# load_dir = 'experiment_data/robotarm/25/55413'
# BOP.load_data(load_dir, n = 100)

# %%

## We can also load the data from a previous run and continue the run from there
## or we could choose to generate a prediction map, which means running MAP-Elites
## on the surrogate models in order to assess how good the models are.

## Generate prediction map
from tools.prediction_archive import prediction_archive
from acq_functions.mean import GPmean
PMoptimizer = mapelites.MAPelites

# Prediction map if features are known
# pred_archive = prediction_archive(BOP, PMoptimizer, GPmean, **{'known_features' : True,})
# BOP.plot_archive2d(archive = pred_archive.pred_archive)

# Prediction map if features are unknown
pred_archive = prediction_archive(BOP, PMoptimizer, UKD_mean, **{'known_features' : False,})
BOP.plot_archive2d(archive = pred_archive.pred_archive)
# %%
