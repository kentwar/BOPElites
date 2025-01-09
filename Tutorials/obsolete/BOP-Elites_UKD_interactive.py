# Required to access parent modules
import sys, os
script_path = os.path.abspath(__file__)
parent_folder_path = os.path.abspath(os.path.join(script_path, "../.."))
sys.path.append(parent_folder_path)

#%%

# %%
from surrogates import GP
from benchmarks.robotarm import RobotArm
from benchmarks.SyntheticGP10d2f import SyntheticGP10d2f 
from archives.archives import structured_archive
from acq_functions.BOP_UKD_MB import BOP_UKD_MB
#from acq_functions.BOP_UKD_vec import BOP_UKD_vec
from acq_functions.UKD_mean import UKD_mean
import copy

#from algorithm.BOP_Elites_UKD import algorithm
from algorithm.BOP_Elites_UKD_MB import algorithm
from optimizers import patternsearch_beta, mapelites, Gradient_Descent, LBFGSB
import numpy as np
import torch

## We configure the experiment by setting the domain, archive and optimizer
## The algorithm is imported from the algorithm folder, and we feed it the
## other modules.

domain    = SyntheticGP10d2f(feature_resolution = [40,40], seed = 195)
QDarchive = structured_archive(domain)
#optimizer = Gradient_Descent.Scipy_optimizer
#optimizer = LBFGSB.LBFGSB_optimizer

optimizer = patternsearch_beta.PatternSearchOptimizer

## 'Known_features' defines whether we use a surrogate to model the descriptor
## space. 
## 'test_mode' tells the algorithm whether to store the data in the test
## folder or directly in the experiment folder.
## 'init_beta' is the initial value of beta in the BOP-Elites_beta algorithm (not needed in others)

BOP = algorithm(domain, QDarchive , BOP_UKD_MB, optimizer , seed = 9999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                'init_beta' : 10,})


## Plot initial data
# BOP.plot_archive2d()
# BOP.plot_convergence()

# ## Run one iteration of BOP-Elites for 2 points (n_initial_points is 10d)
#BOP.run(n_restarts = 10, max_iter = 42)

# ## Plot data after two iterations
# BOP.plot_archive2d()
# BOP.plot_convergence()

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
# BOP.run(n_restarts = 10, max_iter = 46)

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
## 400: 321,      200: 190          - 200 at 5/5 with 100, 200 at 10/10 with 100
#load_dir = 'experiment_data/robotarm/BOP_UKD_beta/5/93345'
## 400 : 360,     200: 247.28       - 200 at 5/5 20, 200 at 10/10 at 10
#load_dir = 'experiment_data/robotarm/BOP_UKD_beta/5/18101'
## 400 : 324.88,  200: 231.27    - 200 at 5/5 with 10, 200 at 25/25 with 10init
#load_dir = 'experiment_data/robotarm/BOP_UKD_beta/5/64299'
## 400 : 362.69,  200: 252.23   - 200 at 5/5 with 20, 200 at 25/25 with 0
#load_dir = 'experiment_data/robotarm/BOP_UKD_beta/5/53231'
## 400 : 369.38,  200: 224.43   - 200 at 25/25 with 4, 200 at 25/25 with 4init
#load_dir = 'experiment_data/robotarm/BOP_UKD_beta/25/67053'
## 400 : 362.51,  200: 175.59   - 200 at 25/25 with 10, 200 at 25/25 with 10init
#load_dir = 'experiment_data/robotarm/BOP_UKD_beta/25/54923'

#
load_dir = 'experiment_data/robotarm/BOP_UKD_beta_beta/5/4963'

## 
load_dir = 'experiment_data/robotarm/BOP_UKD_beta_beta/5/64580'
## 
load_dir = 'experiment_data/robotarm/BOP_UKD_beta_beta/2/16216'
##
load_dir = 'experiment_data/robotarm/BOP_UKD_beta_beta/5/91159'
load_dir = 'experiment_data/robotarm/BOP_UKD_beta_beta/5/5969' # 5 - 10 -25 short switching, not full
#load_dir = 'experiment_data/robotarm/SAIL/25/Test/81797'
#load_dir = 'experiment_data/robotarm/BOP_UKD_beta_beta/25/47141'
load_dir = 'experiment_data/robotarm/BOP_UKD_entropy/5/77560'
load_dir = 'experiment_data/robotarm/BOP_UKD_entropy/5/66984'


#%%
# SPHEN 
#load_dir = 'experiment_data/SyntheticGP10d2f/SAIL/20/Test/195'
# SAIL  - 
#load_dir = 'experiment_data/SyntheticGP10d2f/SAIL/20/195'
# BOP   - 
load_dir = 'experiment_data/SyntheticGP10d2f/BOP_UKD_entropy/5/195'
BOP.load_data(load_dir, n = 309)
from tools.prediction_archive import prediction_archive
from acq_functions.mean import GPmean
PMoptimizer = mapelites.MAPelites
pred_archive  = prediction_archive(BOP, PMoptimizer, GPmean, **{'known_features' : False,
                                                                'return_pred': True})
predfig, predax = BOP.plot_archive2d(archive = pred_archive.pred_archive)
#%%
## Find rectangular regions of interest
from tools.interactive_tools import select_zoom_region, rescale_rectangle , first_multiple, zoom_edges, gen_distance_func
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
rects = [select_zoom_region(BOP) for i in range(2)]  # lower left point, width and height
resized_rects = [rescale_rectangle(r, [40,40]) for r  in  rects]

predfig, predax = BOP.plot_archive2d(archive = pred_archive.pred_archive)
[predax.add_patch(Rectangle(rect[0],rect[1],rect[2], ec = 'black', fc = 'none')) for rect in resized_rects]
plt.show()
#%%
## Create a new BOP instance for each rectangle
valid_ranges = torch.tensor(domain.Xconstraints)
edges = []
# Define edges in each descriptor dimension
for i in range( BOP.QDarchive.fdims ):
    edge_boundaries = np.linspace( BOP.QDarchive.fmins[ i ] , 
                                    BOP.QDarchive.fmaxs[ i ],
                                    [20,20][ i ] + 1 )
    edges.insert( i, edge_boundaries ) 

# Convert to array
edges = np.array( edges )

domain_zoom1  = copy.copy(domain)
zoom1_edges = zoom_edges(edges, rects[0])
feature_resolution = [int(len(zoom1_edges[i]) -1) for i in range(len(zoom1_edges))]
domain_zoom1.feature_resolution = feature_resolution
QDarchive_zoom1  = structured_archive(domain_zoom1)
QDarchive_zoom1.edges = zoom1_edges
#%%
BOP_zoom1 = algorithm(domain_zoom1, QDarchive_zoom1 , BOP_UKD_MB, optimizer , seed = 9999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                'init_beta' : 1,})

BOP_zoom1.zoom_inherit(BOP)
BOP_zoom1.plot_archive2d()
predfig, predax = BOP.plot_archive2d()
[predax.add_patch(Rectangle(rect[0],rect[1],rect[2], ec = 'black', fc = 'none')) for rect in resized_rects]

plt.show()

# %%
## Simulate the next 60 steps
load_dir = 'experiment_data/SyntheticGP10d2f/BOP_UKD_entropy/40/Zoom1'
BOP_zoom1.load_data(load_dir, n = 366)
#BOP_zoom1.run(10, BOP_zoom1.fitness.shape[0] + np.product(BOP_zoom1.QDarchive.feature_resolution))
# %%
domain2    = SyntheticGP10d2f(feature_resolution = [40,40], seed = 195)
QDarchive2 = structured_archive(domain)
BOP2 = algorithm(domain2, QDarchive2 , BOP_UKD_MB, optimizer , seed=99999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                'init_beta' : 1,})
BOP2.zoom_inherit(BOP_zoom1)
BOP2.plot_archive2d()
pred_archive2  = prediction_archive(BOP2, PMoptimizer, GPmean, **{'known_features' : False,
                                                                'return_pred': True})
BOP2.plot_archive2d(archive = pred_archive2.pred_archive)
# %%
pred_archive2  = prediction_archive(BOP2, PMoptimizer, GPmean, **{'known_features' : False,
                                                                'return_pred': False})
_,_ = BOP2.plot_archive2d(archive = pred_archive2.pred_archive)


### Zoom2

domain_zoom2 = copy.copy(domain)
zoom2_edges = zoom_edges(edges, rects[1])
feature_resolution = [int(len(zoom2_edges[i]) -1) for i in range(len(zoom2_edges))]
domain_zoom2.feature_resolution = feature_resolution
QDarchive_zoom2  = structured_archive(domain_zoom2)
QDarchive_zoom2.edges = zoom2_edges
BOP_zoom2 = algorithm(domain_zoom2, QDarchive_zoom2 , BOP_UKD_MB, optimizer , seed = 9999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                'init_beta' : 1,})

#%%
load_dir = 'experiment_data/SyntheticGP10d2f/BOP_UKD_MB/40/Zoom2'
BOP_zoom2.load_data(load_dir, n = 379)
#BOP_zoom2.zoom_inherit(BOP2)
#BOP_zoom2.run(10, BOP2.fitness.shape[0] + np.product(BOP2.QDarchive.feature_resolution))
#%%
domain2   = SyntheticGP10d2f(feature_resolution = [40,40], seed = 195)
QDarchive2 = structured_archive(domain)
BOP3 = algorithm(domain2, QDarchive2 , BOP_UKD_MB, optimizer , seed=99999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                 'init_beta' : 1,})
          # %%
from surrogates import GP
domain_zoom2  = copy.copy(domain)
zoom2_edges = zoom_edges(edges, rects[1])
feature_resolution = [int(len(zoom2_edges[i]) -1) for i in range(len(zoom2_edges))]
domain_zoom2.feature_resolution = feature_resolution
QDarchive_zoom2  = structured_archive(domain_zoom2)
QDarchive_zoom2.edges = zoom2_edges
BOP_zoom2 = algorithm(domain_zoom2, QDarchive_zoom2 , BOP_UKD_MB, optimizer , seed = 9999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                'init_beta' : 1,})

#%%
load_dir = 'experiment_data/SyntheticGP10d2f/BOP_UKD_MB/40/Zoom2'
BOP_zoom2.load_data(load_dir, n = 379)
#BOP_zoom2.zoom_inherit(BOP2)
#BOP_zoom2.run(10, BOP2.fitness.shape[0] + np.product(BOP2.QDarchive.feature_resolution))
#%%
domain2   = SyntheticGP10d2f(feature_resolution = [40,40], seed = 195)
QDarchive2 = structured_archive(domain)
BOP3 = algorithm(domain2, QDarchive2 , BOP_UKD_MB, optimizer , seed=99999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                 'init_beta' : 1,})
          # %%
from surrogates import GP
from benchmarks.robotarm import RobotArm
from benchmarks.SyntheticGP10d2f import SyntheticGP10d2f 
from archives.archives import structured_archive
from acq_functions.BOP_UKD_MB import BOP_UKD_MB
#from acq_functions.BOP_UKD_vec import BOP_UKD_vec
from acq_functions.UKD_mean import UKD_mean
import copy

#from algorithm.BOP_Elites_UKD import algorithm
from algorithm.BOP_Elites_UKD_MB import algorithm
from optimizers import patternsearch_beta, mapelites, Gradient_Descent, LBFGSB
import numpy as np
import torch

## We configure the experiment by setting the domain, archive and optimizer
## The algorithm is imported from the algorithm folder, and we feed it the
## other modules.

domain    = SyntheticGP10d2f(feature_resolution = [40,40], seed = 195)
QDarchive = structured_archive(domain)
#optimizer = Gradient_Descent.Scipy_optimizer
#optimizer = LBFGSB.LBFGSB_optimizer

optimizer = patternsearch_beta.PatternSearchOptimizer

## 'Known_features' defines whether we use a surrogate to model the descriptor
## space. 
## 'test_mode' tells the algorithm whether to store the data in the test
## folder or directly in the experiment folder.
## 'init_beta' is the initial value of beta in the BOP-Elites_beta algorithm (not needed in others)

BOP3 = algorithm(domain, QDarchive , BOP_UKD_MB, optimizer , seed = 9999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                'init_beta' : 10,})

BOP3.zoom_inherit(BOP_zoom2)
BOP3.plot_archive2d()

# %%
pred_archive3  = prediction_archive(BOP3, PMoptimizer, GPmean, **{'known_features' : False,
                                                                'return_pred': True})

predfig, predax = BOP3.plot_archive2d(archive = pred_archive3.pred_archive)
[predax.add_patch(Rectangle(rect[0],rect[1],rect[2], ec = 'black', fc = 'none')) for rect in resized_rects[0:]]

plt.show()

from benchmarks.robotarm import RobotArm
from benchmarks.SyntheticGP10d2f import SyntheticGP10d2f 
from archives.archives import structured_archive
from acq_functions.BOP_UKD_MB import BOP_UKD_MB
#from acq_functions.BOP_UKD_vec import BOP_UKD_vec
from acq_functions.UKD_mean import UKD_mean
import copy

#from algorithm.BOP_Elites_UKD import algorithm
from algorithm.BOP_Elites_UKD_MB import algorithm
from optimizers import patternsearch_beta, mapelites, Gradient_Descent, LBFGSB
import numpy as np
import torch

## We configure the experiment by setting the domain, archive and optimizer
## The algorithm is imported from the algorithm folder, and we feed it the
## other modules.

domain    = SyntheticGP10d2f(feature_resolution = [40,40], seed = 195)
QDarchive = structured_archive(domain)
#optimizer = Gradient_Descent.Scipy_optimizer
#optimizer = LBFGSB.LBFGSB_optimizer

optimizer = patternsearch_beta.PatternSearchOptimizer

## 'Known_features' defines whether we use a surrogate to model the descriptor
## space. 
## 'test_mode' tells the algorithm whether to store the data in the test
## folder or directly in the experiment folder.
## 'init_beta' is the initial value of beta in the BOP-Elites_beta algorithm (not needed in others)

BOP3 = algorithm(domain, QDarchive , BOP_UKD_MB, optimizer , seed = 9999,**{'sampler' : 5, 
                           
domain_zoom2  = copy.copy(domain)
zoom2_edges = zoom_edges(edges, rects[1])
feature_resolution = [int(len(zoom2_edges[i]) -1) for i in range(len(zoom2_edges))]
domain_zoom2.feature_resolution = feature_resolution
QDarchive_zoom2  = structured_archive(domain_zoom2)
QDarchive_zoom2.edges = zoom2_edges
BOP_zoom2 = algorithm(domain_zoom2, QDarchive_zoom2 , BOP_UKD_MB, optimizer , seed = 9999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                'init_beta' : 1,})

#%%
load_dir = 'experiment_data/SyntheticGP10d2f/BOP_UKD_MB/40/Zoom2'
BOP_zoom2.load_data(load_dir, n = 379)
#BOP_zoom2.zoom_inherit(BOP2)
#BOP_zoom2.run(10, BOP2.fitness.shape[0] + np.product(BOP2.QDarchive.feature_resolution))
#%%
domain2   = SyntheticGP10d2f(feature_resolution = [40,40], seed = 195)
QDarchive2 = structured_archive(domain)
BOP3 = algorithm(domain2, QDarchive2 , BOP_UKD_MB, optimizer , seed=99999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                 'init_beta' : 1,})
          # %%
from surrogates import GP
from benchmarks.robotarm import RobotArm
from benchmarks.SyntheticGP10d2f import SyntheticGP10d2f 
from archives.archives import structured_archive
from acq_functions.BOP_UKD_MB import BOP_UKD_MB
#from acq_functions.BOP_UKD_vec import BOP_UKD_vec
from acq_functions.UKD_mean import UKD_mean
import copy

#from algorithm.BOP_Elites_UKD import algorithm
from algorithm.BOP_Elites_UKD_MB import algorithm
from optimizers import patternsearch_beta, mapelites, Gradient_Descent, LBFGSB
import numpy as np
import torch

## We configure the experiment by setting the domain, archive and optimizer
## The algorithm is imported from the algorithm folder, and we feed it the
## other modules.

domain    = SyntheticGP10d2f(feature_resolution = [40,40], seed = 195)
QDarchive = structured_archive(domain)
#optimizer = Gradient_Descent.Scipy_optimizer
#optimizer = LBFGSB.LBFGSB_optimizer

optimizer = patternsearch_beta.PatternSearchOptimizer

## 'Known_features' defines whether we use a surrogate to model the descriptor
## space. 
## 'test_mode' tells the algorithm whether to store the data in the test
## folder or directly in the experiment folder.
## 'init_beta' is the initial value of beta in the BOP-Elites_beta algorithm (not needed in others)

BOP3 = algorithm(domain, QDarchive , BOP_UKD_MB, optimizer , seed = 9999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                'init_beta' : 10,})

BOP3.zoom_inherit(BOP_zoom2)
BOP3.plot_archive2d()

# %%
pred_archive3  = prediction_archive(BOP3, PMoptimizer, GPmean, **{'known_features' : False,
                                                                'return_pred': True})

predfig, predax = BOP3.plot_archive2d(archive = pred_archive3.pred_archive)
[predax.add_patch(Rectangle(rect[0],rect[1],rect[2], ec = 'black', fc = 'none')) for rect in resized_rects[0:]]

plt.show()

pred_archive3  = prediction_archive(BOP3, PMoptimizer, GPmean, **{'known_features' : False,
                                                                'return_pred': True})

predfig, predax = BOP3.plot_archive2d(archive = pred_archive3.pred_archive)
[predax.add_patch(Rectangle(rect[0],rect[1],rect[2], ec = 'black', fc = 'none')) for rect in resized_rects[0:]]

plt.show()

#%%

import itertools as it
from tools.interactive_tools import *

# pick a random point inside rectangle

def pick_random_target_in_rect(rects):
    rect = rects[np.random.choice(len(rects))]
    randomtargetval = torch.rand(2, dtype = torch.double)
    randomtarget = rect[0] + randomtargetval * torch.stack([rect[1], rect[2]])
    return randomtarget
randomtarget = pick_random_target_in_rect([rects[0],rects[1]])

predfig, predax = BOP3.plot_archive2d(archive = pred_archive3.pred_archive)
[predax.add_patch(Rectangle(rect[0],rect[1],rect[2], ec = 'black', fc = 'none')) for rect in resized_rects[0:]]


plt.scatter(randomtarget[1]*40, randomtarget[0]*40, c = 'red')
plt.scatter([0.5796, 0.5313][0]*40, [0.5796, 0.5313][1]*40, c = 'white')
plt.scatter([0.0505, 0.6015][0]*40, [0.0505, 0.6015][1]*40, c = 'black')
plt.show()
#randomtarget = torch.tensor([0.0505, 0.6015], dtype=torch.float64)
# # random z values from the univariate gaussian

# fitz = torch.randn(nd**BOP.domain.fdims, dtype = torch.double)
# featz = torch.randn(nd*BOP.domain.fdims, dtype = torch.double)

# # calculate the acquisition function

# dist = euclidean_distance_vectorized(randomx[0,:2].unsqueeze(0), target.unsqueeze(0))
# acq_val = MC_acq_func(randomx, fitz, featz, BOP.fitGP, BOP.DGPs, target, 1, behdist)
# %%
# scipy optimisation loop
nd = 30   # this will be 30^fdims monte carlo samples

def acqfunc_maker(afmtarget, algorithm, alpha, bestval):
    # Generate z's for sampling
    fitz = torch.randn(nd**algorithm.domain.fdims, dtype = torch.double)
    featz = torch.randn(nd*algorithm.domain.fdims, dtype = torch.double)
    # generate distance function from target
    behdist = gen_distance_func(afmtarget)
    if behdist( afmtarget.unsqueeze(0)) != 0:
        raise ValueError('distance function not working')
    # Now save the acquisition function as a function of x
    def af(x):
        x = torch.tensor(x, dtype = torch.double).reshape(-1, BOP3.domain.xdims)
        return(-MC_acq_func(x, fitz, featz, BOP3.fitGP, BOP3.DGPs,  alpha, behdist, bestval, BOP3))
    return(af)

import numpy as np

from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem 
from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from tqdm import tqdm

xl = BOP3.optimizer.xl
xu = BOP3.optimizer.xu
xdims = BOP3.optimizer.xdims
termination = SingleObjectiveDefaultTermination(
                    x_tol=1e-8,
                    cv_tol=1e-6,
                    f_tol=1e-4,
                    nth_gen=5,
                    n_last=20,
                    n_max_gen=100,
                    n_max_evals=1500
                    ) 
def pymooset(x0):
    optalgorithm = PatternSearch(x0 = x0)
    return(optalgorithm)

def runopt(x0, problem, termination):
    print(f'x0 = {x0}')
    x0 = np.array(x0)
    algorithm = pymooset(x0)
    res = minimize(problem ,
                       algorithm,
                       termination=termination,
                       seed=1,
                       verbose=False)
    print(f'final poistion {res.X}')
    print(f'final value {res.F}')
    return(res)



def MCoptimise(MCOtarget, alpha  , bestval):
    # Initial guess
    print('making function and finding initial guesses')
    acqf = acqfunc_maker(MCOtarget, BOP3, alpha, bestval)
    X0s = []
    Xobserved = BOP3.x
    try:
        Xobserved = torch.cat((Xobserved,torch.tensor(BOP3.saved_points2)))
    except:
        pass
    F0 = [acqf(x0) for x0 in Xobserved]
    for c, x in enumerate(Xobserved):
        if F0[c] < 0:
            X0s.append(x)
    while len(X0s) < 10:
        X0 = BOP.domain.get_Xsample(1000)
        F0 = [acqf(x0) for x0 in X0]
        for c, x in enumerate(X0):
            if F0[c] < 0:
                X0s.append(x)
        mutatedx0 = X0s[np.random.randint(len(X0s))] + np.random.normal(0, 0.1, size = (100, BOP3.domain.xdims))
        mutatedx0 = torch.clip(mutatedx0, 0, 1)
        F0 = [acqf(x0) for x0 in mutatedx0]
        bestmutant = mutatedx0[np.argmin(F0)]
        X0s.append(bestmutant)
    x0 = X0s[:10]
    # Xobserved = BOP3.x
    # X0 = torch.cat((X0, Xobserved))
    # F0 = [acqf(x0) for x0 in X0] 
    # # pick the 10 best values from F0
    # x0 = X0[np.argsort(F0)[:10]]
    print('optimising')
    # Minimize the function using the Nelder-Mead algorithm
    results = []
    for x0 in tqdm(x0):
        acqf = acqfunc_maker(MCOtarget, BOP3, alpha, bestval)
        problem = FunctionalProblem(BOP3.domain.xdims,
                                        objs = acqf,
                                        xl=xl,
                                        xu=xu
                                        )
        result = runopt(x0, problem, termination)
        results.append((result.X, result.F))
    # Print the optimized value
    # pick the lowest value
    acqf = acqfunc_maker(MCOtarget, BOP3, alpha, bestval)
    bestvals = [acqf(r[0]) for r in results]
    best = results[np.argmin(bestvals)]
    BOP3.saved_points2 = [r[0] for r in results]
    return(best[0])
    
target_region_index = tuple(BOP3.QDarchive.nichefinder(randomtarget).numpy()[0])
elitepoint = BOP3.QDarchive.genomes[target_region_index]
if torch.isnan(elitepoint).any():
    print('no elite point in this region')
else:
    efit = BOP3.domain.fitness_fun(elitepoint.numpy().reshape(-1,10))
    edesc = BOP3.domain.feature_fun(elitepoint.numpy().reshape(-1,10))
    print('assumed best = ', efit - euclidean_distance(torch.tensor(edesc), randomtarget))

#target2 = torch.tensor([0.8921, 0.3878], dtype=torch.float64)
#behdist= gen_distance_func(target2)
alpha = 1
values = BOP3.fitness - alpha * behdist(BOP3.descriptors)
bestindex = torch.argmax(values)
bestx = BOP3.x[bestindex]
bestpos = BOP3.descriptors[bestindex]

# Produce a list of BOP.fitness  by v in values
orderedvalues = values[torch.argsort(values)]


std = BOP3.fitness.std().item()
mean = BOP3.fitness.mean().item()
predval = BOP3.fitGP(bestx.reshape(-1,10)).mean*std +mean - alpha * behdist(bestpos) 
bestval = predval
BOP3.saved_points2 = []
for i in range(20):
    new_point = MCoptimise(randomtarget, alpha, bestval)
    new_obs = BOP3.evaluate_new_point(new_point)
    BOP3.update_storage([new_obs])
    BOP3.createmodels()
    ymean = BOP3.fitness.mean(); ystd = BOP.fitness.std()
    new_obs[2] = torch.tensor(new_obs[2], dtype = torch.double)
    BOP3.QDarchive.updatearchive([new_obs], ymean, ystd)


    bestfit = BOP3.fitness[-1]
    true_value = bestfit - alpha * behdist(BOP3.descriptors[-1])
    if true_value > bestval:
        bestval = true_value
        bestx = new_point[0]
        bestpos = BOP3.descriptors[-1]
    print('best point so far: ', bestval)


# %%

