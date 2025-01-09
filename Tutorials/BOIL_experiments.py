# Required to access parent modules
import sys, os
import inspect
script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
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
import copy

#from algorithm.BOP_Elites_UKD import algorithm
from algorithm.BOP_Elites_UKD_MB import algorithm
from optimizers import patternsearch_beta, mapelites
from tools.prediction_archive import prediction_archive
from acq_functions.mean import GPmean
import numpy as np
from tools.interactive_tools import *
from matplotlib.patches import Rectangle
import torch
import matplotlib.pyplot as plt
import math
## We configure the experiment by setting the domain, archive and optimizer
## The algorithm is imported from the algorithm folder, and we feed it the
## other modules.

#%%  This cell initiates the first run of the algorithm which will terminate 
# Once the acquisition function drops below 0.5


seed = 192#np.random.randint(10000)
n_restarts = 10
domain    = SyntheticGP10d2f(feature_resolution = [5,5], seed = seed)
QDarchive = structured_archive(domain)
optimizer = patternsearch_beta.PatternSearchOptimizer

## 'Known_features' defines whether we use a surrogate to model the descriptor
## space. 
## 'test_mode' tells the algorithm whether to store the data in the test
## folder or directly in the experiment folder.
## 'init_beta' is the initial value of beta in the BOP-Elites_beta algorithm (not needed in others)

BOP = algorithm(domain, QDarchive , BOP_UKD_MB, optimizer , seed = seed,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : False,
                                                'init_beta' : 1,})
experiment_save_path = BOP.save_path

# %%
## We perform a run of BOP_mb (BOP with model building using the entropy 
# acquisition function using 10 restart points per run)

BOP.run(n_restarts = n_restarts)

# %%

# At this point we can now visualise the plot, both from a true fitness perspective
# and a predicted map perspective

def plot_hires(alg, resolution):
    domain_hires = SyntheticGP10d2f(feature_resolution = [40,40], seed = seed)
    QDarchive_hires = structured_archive(domain_hires)
    optimizer = patternsearch_beta.PatternSearchOptimizer
    BOP_hires = algorithm(domain_hires, QDarchive_hires , BOP_UKD_MB, optimizer , seed = 9999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : False,
                                                'init_beta' : 1,})
    BOP_hires.inherit(alg)
    PMoptimizer = mapelites.MAPelites
    hires_pred = prediction_archive(BOP_hires, PMoptimizer, GPmean, **{'known_features' : False,
                                                                'return_pred': True})
    pred_archive = hires_pred.pred_archive
    true_archive = hires_pred.true_pred_archive
    BOP_hires.plot_archive2d(text = f'observed_{resolution})')
    BOP_hires.plot_archive2d(archive = pred_archive, text = f'predmap_{resolution}')
    BOP_hires.plot_archive2d(archive = true_archive, text = f'true_map_{resolution}')
    return(BOP_hires, true_archive, pred_archive)

BOP_5, true_archive_5, pred_archive_5 = plot_hires(BOP, [5,5])
# %%
# Upscale the archive 
BOP.upscale(resolution = [10,10])
BOP.run(n_restarts = n_restarts)
# %%
# Plot in full resolution with prediction archive
BOP_hires_10, true_archive_10, pred_archive_10 = plot_hires(BOP, [10,10])
## Find rectangular regions of interest

#%%
# Now we simulate a decision maker selecting 2 rectangular regions of interest

rects = [select_zoom_region(BOP) for i in range(2)]  # lower left point, width and height
resized_rects = [rescale_rectangle(r, [40,40]) for r  in  rects]

predax, predfig = BOP_hires_10.plot_archive2d(archive = pred_archive_10, text = 'predmap_10')
[predax.add_patch(Rectangle(rect[0],rect[1],rect[2], ec = 'black', fc = 'none')) for rect in rects]
plt.plot()
#%%
## Create a new BOP instance for each rectangle
def gen_zoomed_archive(rect, resolution, archive):
    '''
    Generates a new archive with a new domain and new edges based on the rectangle

    '''
    zoomed_edges = []
    for i in range(archive.fdims):
        edge_boundaries = np.linspace(archive.fmins[i], archive.fmaxs[i], resolution[i] + 1)
        zoomed_edges.insert(i, edge_boundaries)
    edges = archive.edges
    zoomed_edges = zoom_edges(zoomed_edges, rect)
    feature_resolution = [int(len(zoomed_edges[i]) -1) for i in range(len(zoomed_edges))]
    zoomed_domain  = copy.copy(domain)
    zoomed_domain.feature_resolution = feature_resolution
    zoomed_QDarchive  = structured_archive(zoomed_domain)
    zoomed_QDarchive.edges = zoomed_edges
    return(zoomed_domain, zoomed_QDarchive)


# valid_ranges = torch.tensor(domain.Xconstraints)
# edges = []
# # Define edges in each descriptor dimension
# for i in range( BOP.QDarchive.fdims ):
#     edge_boundaries = np.linspace( BOP.QDarchive.fmins[ i ] , 
#                                     BOP.QDarchive.fmaxs[ i ],
#                                     [20,20][ i ] + 1 )
#     edges.insert( i, edge_boundaries ) 

# # Convert to array
# edges = np.array( edges )

# domain_zoom1  = copy.copy(domain)
# zoom1_edges = zoom_edges(edges, rects[0])
# feature_resolution = [int(len(zoom1_edges[i]) -1) for i in range(len(zoom1_edges))]
# domain_zoom1.feature_resolution = feature_resolution
# QDarchive_zoom1  = structured_archive(domain_zoom1)
# QDarchive_zoom1.edges = zoom1_edges
#%%
zoom1_domain, zoom1_archive = gen_zoomed_archive(rects[0], [20,20], BOP.QDarchive)

BOP_zoom1 = algorithm(zoom1_domain, zoom1_archive , BOP_UKD_MB, optimizer , seed = 9999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                'init_beta' : 1,})

BOP_zoom1.zoom_inherit(BOP)
BOP_zoom1.run(n_restarts = n_restarts)
BOP.inherit(BOP_zoom1)  # We give the points back to the main algorithm


# %%
zoom2_domain, zoom2_archive = gen_zoomed_archive(rects[1], [20,20], BOP.QDarchive)
BOP_zoom2 = algorithm(zoom2_domain, zoom2_archive , BOP_UKD_MB, optimizer , seed = 9999,**{'sampler' : 5, 
                                                'known_features' : False,
                                                'test_mode' : True,
                                                'init_beta' : 1,})
BOP_zoom2.zoom_inherit(BOP)
BOP_zoom2.run(n_restarts = n_restarts)
BOP.inherit(BOP_zoom2)  
# %%
BOP_hires_z2, true_archive_z2, pred_archive_z2 =  plot_hires(BOP_zoom2, [20,20])

# %%

#%%
# Now we simulate a decision maker picking a single point in descriptor space
# and a single alpha value
# file = '/home/rawsys/matdrm/PhD_code/BOP-Elites/Tutorials/experiment_data/SyntheticGP10d2f/BOP_UKD_MB/5/195/'
# BOP.load_data(file)

def pick_random_target_in_rect(rects):
    rect = rects[np.random.choice(len(rects))]
    randomtargetval = torch.rand(2, dtype = torch.double)
    randomtarget = rect[0] + randomtargetval * torch.stack([rect[1], rect[2]])
    return randomtarget

randomtarget = pick_random_target_in_rect([rects[0],rects[1]])

predax, predfig = BOP.plot_archive2d(archive = pred_archive_z2)
[predax.add_patch(Rectangle(rect[0],rect[1],rect[2], ec = 'black', fc = 'none')) for rect in rects[0:]]

plt.scatter(randomtarget[1]*40, randomtarget[0]*40, c = 'red')
plt.show()
# %%
# patternsearch optimisation loop
file = '/home/rawsys/matdrm/PhD_code/BOP-Elites/Tutorials/experiment_data/SyntheticGP10d2f/BOP_UKD_MB/5/195/'
BOP.load_data(file)
BOP.upscale(resolution = [40,40])
for c, X in enumerate(BOP.x):
    BOP.fitness[c] = torch.tensor(BOP.domain.fitness_fun(X.numpy().reshape(-1,10)), dtype = torch.double)
    BOP.descriptors[c] = torch.tensor(BOP.domain.feature_fun(X.numpy().reshape(-1,10)), dtype = torch.double)


from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem 
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch

from tqdm import tqdm

nd = 30   # this will be 30^fdims monte carlo samples

#alpha = torch.rand(1, dtype = torch.double) * 3
target = randomtarget
alpha = torch.tensor(1, dtype = torch.double)

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
        x = torch.tensor(x, dtype = torch.double).reshape(-1, BOP.domain.xdims)
        return(-MC_acq_func(x, fitz, featz, BOP.fitGP, BOP.DGPs,  alpha, behdist, bestval, BOP))
    return(af)

xl = BOP.optimizer.xl
xu = BOP.optimizer.xu
xdims = BOP.optimizer.xdims
termination = DefaultSingleObjectiveTermination(
                    xtol=1e-8,
                    cvtol=1e-6,
                    ftol=1e-4,
                    n_max_gen=100,
                    n_max_evals=1000
                    ) 

def pymooset(x0):
    optalgorithm = PatternSearch(x0 = x0)
    return(optalgorithm)

def runopt(x0, problem, termination):
    #print(f'x0 = {x0}')
    x0 = np.array(x0)
    algorithm = pymooset(x0)
    res = minimize(problem ,
                       algorithm,
                       termination=termination,
                       seed=1,
                       verbose=False)
    #print(f'final poistion {res.X}')
    print(f'final value {res.F}')
    return(res)

def MCoptimise(MCOtarget, alpha  , bestval):
    # Initial guess
    print('making function and finding initial guesses')
    acqf = acqfunc_maker(MCOtarget, BOP, alpha, bestval)
    X0s = []
    Xobserved = BOP.x
    try:
        Xobserved = torch.cat((Xobserved,torch.tensor(BOP.saved_points2)))
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
        mutatedx0 = X0s[np.random.randint(len(X0s))] + np.random.normal(0, 0.1, size = (100, BOP.domain.xdims))
        mutatedx0 = torch.clamp(mutatedx0, 0, 1)
        F0 = [acqf(x0) for x0 in mutatedx0]
        bestmutant = mutatedx0[np.argmin(F0)]
        X0s.append(bestmutant)
    x0 = X0s[:10]

    print('optimising')

    results = []
    for x0 in tqdm(x0):
        acqf = acqfunc_maker(MCOtarget, BOP, alpha, bestval)
        problem = FunctionalProblem(BOP.domain.xdims,
                                        objs = acqf,
                                        xl=xl,
                                        xu=xu
                                        )
        result = runopt(x0, problem, termination)
        results.append((result.X, result.F))
    # Print the optimized value
    # pick the lowest value
    acqf = acqfunc_maker(MCOtarget, BOP, alpha, bestval)
    bestvals = [acqf(r[0]) for r in results]
    best = results[np.argmin(bestvals)]
    BOP.saved_points2 = [r[0] for r in results]
    return(best[0])

def find_best(alg, target):
    '''
    looks through the observation list and returns the best value given the 
    declared preference for the target and returns the value of the 
    acquisition function at that point
    '''
    behdist = gen_distance_func(target)
    values = alg.fitness - alpha * behdist(alg.descriptors)
    bestindex = torch.argmax(values)
    bestx = alg.x[bestindex]
    bestpos = alg.descriptors[bestindex]
    std = alg.fitness.std().item()
    mean = alg.fitness.mean().item()
    predfit = alg.fitGP(bestx.reshape(-1,10)).mean*std +mean
    preddist = alpha * behdist(bestpos) 
    predval = predfit  - preddist
    return(predval, bestx)



bestval, bestx = find_best(BOP, randomtarget)
bestbeh = BOP.domain.feature_fun(bestx.numpy().reshape(-1,10))
besttrueval = BOP.domain.fitness_fun(bestx.numpy().reshape(-1,10)) - alpha * behdist(torch.tensor(bestbeh))

BOP.saved_points2 = []

behdist = gen_distance_func(randomtarget)

for i in range(20):
    print('iteration ', i)
    new_point = MCoptimise(randomtarget, alpha, bestval)
    new_obs = BOP.evaluate_new_point(new_point)
    BOP.update_storage([new_obs])
    BOP.createmodels()
    ymean = BOP.fitness.mean(); ystd = BOP.fitness.std()
    new_obs[2] = torch.tensor(new_obs[2], dtype = torch.double)
    BOP.QDarchive.updatearchive([new_obs], ymean, ystd)


    bestfit = BOP.fitness[-1]
    true_value = bestfit - alpha * behdist(BOP.descriptors[-1])
    print('true value: ', true_value)
    print('bestfit: ', bestfit)
    print('behavioural distance: ', behdist(BOP.descriptors[-1]))
    if true_value > besttrueval:
        besttrueval = true_value
        bestval = besttrueval
        bestx = new_point
        bestpos = BOP.descriptors[-1]
    print('best point so far: ', besttrueval)
BOP.save_path = experiment_save_path 
BOP.save_data()


## Save the final value in a text file
with open(f'{BOP.save_path}/bestval.txt', 'w') as f:
    bestfit = BOP.domain.fitness_fun(bestx.reshape(-1, BOP.domain.xdims))
    bestpos = BOP.domain.feature_fun(bestx.reshape(-1, BOP.domain.xdims))
    bestdist = behdist(torch.tensor(bestpos))
    f.write(f'best value {bestval} \n')
    f.write(f'best fitness {bestfit} \n')
    f.write(f'best position {bestdist} \n')
    f.write(f'best x {bestx} \n')
    f.write(f'target x {randomtarget} \n')
    f.write(f'rectangle1 {rects[0]} \n')
    f.write(f'rectangle2 {rects[1]}')
import pdb; pdb.set_trace()
# %%

