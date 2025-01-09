import numpy as np
import torch, sys, os, inspect, shutil, argparse
# Neccesary to load the main BOP-Elites code
script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
parent_folder_path = os.path.abspath(os.path.join(script_path, "../.."))
sys.path.append(parent_folder_path)

from archives.archives import structured_archive
from acq_functions.BOP_UKD_beta import BOP_UKD_beta_beta
from acq_functions.BOP_EI_Parsec import BOP_EI_Parsec
from acq_functions.UKD_mean import UKD_mean
from algorithm.BOP_Elites_UKD_beta import algorithm
from algorithm.BOP_Elites_Parsec import algorithm as algorithm2
from optimizers import patternsearch_beta,patternsearch, mapelites 
seed = 4
from benchmarks.robotarm import RobotArm 
from benchmarks.rosenbrock6d import Rosenbrock6d
from benchmarks.mishra_bird_function import Mishra_bird_function
from benchmarks.SyntheticGP10d2f import SyntheticGP10d2f
from benchmarks.parsec import Parsec



def make_BOP(max_res, domain, seed = 4, Known_Features = True):
    if domain == 'robotarm':
        _domain = RobotArm
    elif domain == 'rosenbrock6d':
        _domain = Rosenbrock6d
    elif domain == 'mishra_bird_function':
        _domain = Mishra_bird_function
    elif domain == 'SyntheticGP10d2f':
        _domain = SyntheticGP10d2f
    elif domain == 'Parsec':
        _domain = Parsec
    resolutions = [[5,5],[max_res]*2]
    domain    = _domain(feature_resolution = resolutions[-1], seed = seed)
    QDarchive = structured_archive(domain)
    optimizer = patternsearch_beta.PatternSearchOptimizer
    if not _domain == Parsec:
        BOP = algorithm(domain, QDarchive , BOP_UKD_beta_beta, optimizer , resolutions,seed = seed,  **{'sampler' : 5, 
                                                    'known_features' : Known_Features,
                                                    'test_mode' : False,
                                                    'init_beta' : 0,
                                                    'stable_beta' : False,})
    else:
        BOP = algorithm2(domain, QDarchive , BOP_EI_Parsec, optimizer , resolutions,seed = seed,  **{'sampler' : 5, 
                                                    'known_features' : True,
                                                    'test_mode' : False,
                                                    'init_beta' : 0,
                                                    'stable_beta' : False,
                                                    'n_init' : 5,})
    return(BOP)