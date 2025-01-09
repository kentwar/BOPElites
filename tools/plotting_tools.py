from tools.prediction_archive import prediction_archive
from acq_functions.mean import GPmean
from acq_functions.UKD_mean import UKD_mean
from acq_functions.UKD_50 import UKD_50
from optimizers import  mapelites 
import os, torch
import numpy as np
import matplotlib.pyplot as plt

def calc_true_values(exp_dir, BOP, max_n):
    '''
    We assume the BOP algorithm is correctly configured for 
    the experiment.

    exp_dir: directory containing the experiments
    BOP: BOP algorithm
    max_n: maximum iteration of experiments to load
    
    returns: list of true values
   '''
    BOP.load_data(exp_dir, max_n, models = False)
    return(BOP.progress)

def true_from_exp_dir(home_dir, BOP, max_n):
    '''
    home_dir: directory containing the experiments
    BOP: BOP algorithm
    max_n: maximum iteration of experiments to load
    
    returns: list of true values
   '''
    true_vals = []
    for exp_dir in os.listdir(home_dir):
        true_vals.append(calc_true_values(home_dir + '/'+exp_dir, BOP, max_n)[:max_n])
    return(torch.stack(true_vals))

def calc_pred_values(exp_dir, BOP, max_n):
    '''
    produces a dictionary with the pred-map values indexed by iteration
    '''
    PMoptimizer = mapelites.MAPelites
    first_iter =  10 * BOP.domain.xdims
    pred_values = {0: torch.tensor(0)}
    iter = 100
    valrange = [first_iter] + list(range(100,max_n, iter))
    if not [max_n] in valrange:
        valrange.append(max_n)
    for i in valrange:
        print(f'Loading iteration {i}')
        BOP.load_data(exp_dir, i)
        pred_archive = prediction_archive(BOP, PMoptimizer, UKD_mean,
                                            **{'known_features' : BOP.known_features,
                                               'return_pred' : True, 
                                               'PM_params' : [2**7,2**8]} )
        true_plot_archive = pred_archive.true_pred_archive
        plot_archive = pred_archive.pred_archive
        pred_values[i] = torch.nansum(true_plot_archive.fitness)
    return(pred_values)

def pred_from_exp_dir(home_dir, BOP, max_n):
    '''
    home_dir: directory containing the experiments
    BOP: BOP algorithm
    max_n: maximum iteration of experiments to load
    
    returns: list of true values
   '''
    pred_vals = []
    for exp_dir in os.listdir(home_dir):
        pred_vals.append(calc_pred_values(home_dir + '/'+exp_dir, BOP, max_n))
    pred_dic = {}
    for i in pred_vals[0]:
        pred_dic[i] = torch.stack([p[i] for p in pred_vals])
    return(pred_dic)

def plot_from_true(values, label = None, color = 'k'):
    mean = values.mean(dim = 0)
    sem = values.std(dim = 0)/np.sqrt(values.shape[0])
    N = values.shape[1]
    plt.plot(range(N),mean, color = color, label = label)
    plt.fill_between(range(N), mean-sem, mean+sem, color = color, alpha = 0.2)
    plt.xlabel('Iteration - log scale')
    plt.ylabel('True value')
    plt.xscale('log')
    plt.legend()

def plot_from_preds(pred_dic, label = None, color = 'k'):
    mean = np.array([np.mean(pred_dic[p]) for p in pred_dic])
    sem = np.array([np.std(pred_dic[p])/np.sqrt(len(pred_dic[p])) for p in pred_dic])
    plt.plot(list(pred_dic.keys()),mean, color = color, label = label)
    plt.fill_between(list(pred_dic.keys()), mean-sem, mean+sem, color = color, alpha = 0.2)
    plt.xlabel('Iteration - log scale')
    plt.ylabel('True value')
    plt.xscale('log')
    plt.legend()

