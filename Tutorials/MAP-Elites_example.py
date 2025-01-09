# Required to access parent modules
import sys, os
import inspect
script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
parent_folder_path = os.path.abspath(os.path.join(script_path, "../.."))
sys.path.append(parent_folder_path)

# %% 
from surrogates import GP
from benchmarks.robotarm import RobotArm
from benchmarks.SyntheticGP10d2f import SyntheticGP10d2f
from archives.archives import structured_archive
from algorithm.MAP_Elites import algorithm
from optimizers import mapelites
import torch
import numpy as np

## We configure the experiment by setting the domain, archive and optimizer
## The algorithm is imported from the algorithm folder, and we feed it the
## other modules.
# 122 was ok
seed = 192   

domain    = SyntheticGP10d2f(feature_resolution = [40,40], seed = seed)
QDarchive = structured_archive(domain)
optimizer = mapelites.MAPelites

# %%

## 'Known_features' defines whether we use a surrogate to model the descriptor
## space. 'test_mode' tells the algorithm whether to store the data in the test
## folder or directly in the experiment folder.

ME = algorithm(domain, QDarchive, optimizer, seed = 192,**{'known_features' : True,
                                                'test_mode' : True,})


# %%

## Elements of the algorithm can be accessed through the 'algorithm' object.

x = torch.tensor(np.random.random((1,domain.xdims)), dtype=torch.double) # Random input

true_fitness = domain.fitness_fun(x.numpy().reshape(-1,domain.xdims))
print('True fitness: ', true_fitness)

# %%

#Run one iteration of MAP-Elites for 1000 points

ME.run(n_children = 64, max_iter = 450)
# ME.fitness= ME.fitness[:450]
# ME.x = ME.x[:450]
# ME.descriptors = ME.descriptors[:450]
ME.plot_archive2d()
ME.plot_convergence()
ME.QDarchive.get_num_niches() 
# %%

## Generate prediction map
## Generate prediction map
from tools.prediction_archive import prediction_archive
from acq_functions.mean import GPmean

PMoptimizer = mapelites.MAPelites
pred_archive = prediction_archive(ME, PMoptimizer, GPmean,  **{'known_features' : True ,
                                                            'return_pred' : True}, )
true_plot_archive = pred_archive.true_pred_archive
plot_archive = pred_archive.pred_archive
ME.plot_archive2d(archive = true_plot_archive, text = 'True Predicted archive')
ME.plot_archive2d(archive = plot_archive, text = 'Predicted archive')



### 
# 
from tools.interactive_tools import *