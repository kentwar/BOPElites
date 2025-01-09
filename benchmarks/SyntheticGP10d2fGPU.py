from benchmarks.BaseBenchmark import BaseExperiment
import numpy as np
import torch
import os
from benchmarks.GP_maker import GP_makerGPU
synthetic_GP = GP_makerGPU.synthetic_GP

class SyntheticGP10d2f(BaseExperiment):
    def __init__(self, feature_resolution , seed = 100):
        
        kwargs =    {
            'example_x' : [0,0,0,0,0,0,0,0,0,0] ,
            'Xconstraints' : [[0, 1], [0, 1], [0, 1], [0, 1],[0, 1], [0, 1], [0, 1], [0, 1] ,[0, 1], [0, 1] ] ,
            'featmins' : [0,0] ,
            'featmaxs' : [1,1] ,
            'lowestvalue' : 0 ,
            'maxvalue' : 1 , # Max value  is not important for this benchmark
            }
        tkwargs = {
        "dtype": torch.double,
        "device": "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
        self._set_Xconstraints(np.array(kwargs['Xconstraints']))    #input x ranges [min,max]
        self.example_x = kwargs['example_x']
        self.xdims = len(kwargs['example_x']) 
        self.fdims = len(kwargs['featmins']) 
        self.featmins = kwargs['featmins']
        self.featmaxs = kwargs['featmaxs']
        self.feature_resolution = feature_resolution
        self.lowestvalue = kwargs['lowestvalue']
        self.maxvalue = kwargs['maxvalue']
        self.seed = seed
        self.name = 'SyntheticGP10d2f'
        self.desc1name = 'feature 1'
        self.desc2name = 'feature 2'
        torch.manual_seed(seed)
        self.fitls = torch.rand(1, **tkwargs)*0.6+0.2
        self.featls = torch.rand(1, **tkwargs)*0.6+0.2 
        self.dtype = torch.double
        self.fit_GP   = synthetic_GP(seed = seed, dim = self.xdims , hypers_ls = self.fitls, bounds = [-1.787691461612756, 2.735910248722252])
        self.feat_GP1  = synthetic_GP(seed = seed+1, dim = self.xdims , hypers_ls = self.featls, bounds = [-1.8271578800271637, 2.1981304658748804])
        self.feat_GP2  = synthetic_GP(seed = seed+2, dim = self.xdims , hypers_ls = self.featls, bounds = [-2.2376418652748344,2.564892808332293])

    def fitness_fun(self, X):
        '''Function wrapper
        '''
        t = type(X)
        s = np.shape(X)
        ms = np.shape(X[0])
        assert t == np.ndarray, 'Input to the fitness function must be an array'
        assert s == (1,self.xdims) or ms == (self.xdims,), 'genome is the wrong shape'
        ## Single Point    
        ## Single Point    
        if s == (1,self.xdims):
            X = torch.from_numpy(np.double(X[0])).reshape(-1,self.xdims)
            return(self.fit_GP.evaluate_norm(X).detach().numpy()[0][0] )
        else:
            X = torch.from_numpy(X).reshape(-1,self.xdims)
            return([self.fit_GP.evaluate_norm(x).detach().numpy()[0][0] for x in X])

    def feature_fun(self, X):
        '''Function wrapper
        '''
        t = type(X)
        s = np.shape(X)
        ms = np.shape(X[0])
        assert t == np.ndarray, 'Input to the fitness function must be an array'
        assert s == (1,self.xdims) or ms == (self.xdims,), 'genome is the wrong shape'
        assert (X >= self.Xconstraints[:,0]).all() , 'The point is outside the box constraints (lower bound)'
        assert (X <= self.Xconstraints[:,1]).all() , 'The point is outside the box constraints (Upper bound)'
        ## Single Point    
        if s == (1,self.xdims):
            X = torch.from_numpy(np.double(X[0])).reshape(-1,self.xdims)
            feat1 = self.feat_GP1.evaluate_norm(X).detach().numpy()[0][0]
            feat2 = self.feat_GP2.evaluate_norm(X).detach().numpy()[0][0]
            return(np.array([feat1,feat2]))
        else:
            X = torch.from_numpy(X).reshape(-1,self.xdims)
            feat1 = [self.feat_GP1.evaluate_norm(x).detach().numpy()[0][0] for x in X]
            feat2 = [self.feat_GP2.evaluate_norm(x).detach().numpy()[0][0] for x in X]
            return(np.array([feat1,feat2]).T)


    def BOtorch_fitness_fun(self, X):
        '''Function wrapper
        '''
        ## Single Point    
        return(self.fit_GP.evaluate_norm(X).squeeze(-1))

    def BOtorch_feature_fun(self, X):
        '''Function wrapper
        '''
        ## Single Point 
        feat1 = self.feat_GP1.evaluate_norm(X)
        feat2 = self.feat_GP2.evaluate_norm(X) 
        return(torch.stack((feat1,feat2), dim = -1).squeeze(1))

