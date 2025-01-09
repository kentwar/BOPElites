# Required to access parent modules
import sys, os
script_path = os.path.abspath(__file__)
parent_folder_path = os.path.abspath(os.path.join(script_path, "../.."))
sys.path.append(parent_folder_path)
# %%
from surrogates import GP
from benchmarks.robotarm import RobotArm
from benchmarks.SyntheticGP10d2f import SyntheticGP10d2f 
from archives.archives import structured_archive
from acq_functions.obsolete.BOP_UKD_beta import BOP_UKD_beta
#from acq_functions.BOP_UKD_vec import BOP_UKD_vec
from acq_functions.UKD_mean import UKD_mean

from algorithm.BOP_Elites_UKD import algorithm
from optimizers import patternsearch, mapelites, Gradient_Descent, LBFGSB
import numpy as np
import torch

## We configure the experiment by setting the domain, archive and optimizer
## The algorithm is imported from the algorithm folder, and we feed it the
## other modules.

#domain    = RobotArm(feature_resolution = [25,25], seed = 195)
domain    = SyntheticGP10d2f(feature_resolution = [40,40], seed = 195)
QDarchive = structured_archive(domain)
#optimizer = Gradient_Descent.Scipy_optimizer
#optimizer = LBFGSB.LBFGSB_optimizer

optimizer = patternsearch.PatternSearchOptimizer

## 'Known_features' defines whether we use a surrogate to model the descriptor
## space. 
## 'test_mode' tells the algorithm whether to store the data in the test
## folder or directly in the experiment folder.
## 'init_beta' is the initial value of beta in the BOP-Elites_beta algorithm (not needed in others)

BOP = algorithm(domain, QDarchive , BOP_UKD_beta, optimizer ,resolutions = [[25,25]], seed = 9999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                'init_beta' : 0,})


# Plot initial data
BOP.plot_archive2d()
BOP.plot_convergence()

## Run one iteration of BOP-Elites for 2 points (n_initial_points is 10d)
BOP.run(n_restarts = 10, max_iter = 101)

## Plot data after two iterations
BOP.plot_archive2d()
BOP.plot_convergence()


# %%

# We access modules of the algorithm from within the algorithm itself. Lets
# check the fitness prediction of the algorithm.
 
x = torch.tensor(np.random.random((1,domain.xdims)), dtype=torch.double) # Random input

## Check fitness predictions
true_fitness = BOP.domain.fitness_fun(x.numpy().reshape(-1,domain.xdims))
fitness_prediction = BOP.predict_fit(x).item()
print('True fitness: ', true_fitness)
print('Fitness prediction: ', fitness_prediction)

# %%
## Generate prediction map
from tools.prediction_archive import prediction_archive
from acq_functions.mean import GPmean

PMoptimizer = mapelites.MAPelites
pred_archive = prediction_archive(BOP, PMoptimizer, GPmean,  **{'known_features' : True ,
                                                            'return_pred' : True}, )
true_plot_archive = pred_archive.true_pred_archive
plot_archive = pred_archive.pred_archive
BOP.plot_archive2d(archive = true_plot_archive, text = 'True Predicted archive')
BOP.plot_archive2d(archive = plot_archive, text = 'Predicted archive')
#%%


