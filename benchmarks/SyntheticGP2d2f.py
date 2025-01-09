import sys
sys.path.insert(0, '/home/rawsys/matdrm/PhD_code/Juan/BOP-Elites-2022/')


from benchmarks.BaseBenchmark import BaseExperiment
import numpy as np
import torch
import os
from typing import List, Tuple

from benchmarks.GP_maker import GP_maker
synthetic_GP = GP_maker.synthetic_GP

class SyntheticGP2d2f(BaseExperiment):
    def __init__(self, feature_resolution , seed = 133):
        
        kwargs =    {
            'example_x' : [0,0] ,
            'Xconstraints' : [[0, 1], [0, 1] ] ,
            'featmins' : [0,0] ,
            'featmaxs' : [1,1] ,
            'lowestvalue' : 0 ,
            'maxvalue' : 1 , 
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
        self.name = 'SyntheticGP2d2f'
        self.desc1name = 'feature 1'
        self.desc2name = 'feature 2'

        # Save the current states - neccesary for keeping the random seed intact
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()


        torch.manual_seed(seed)
        self.fitls = torch.rand(1)*0.6+0.2
        self.featls = torch.rand(1)*0.6+0.2 
        self.dtype = torch.double
        self.fit_GP   = synthetic_GP(seed = seed, dim = self.xdims , hypers_ls = self.fitls)
        #self.feat_GP1  = synthetic_GP(seed = seed+1, dim = self.xdims , hypers_ls = self.featls)
        #self.feat_GP2  = synthetic_GP(seed = seed+2, dim = self.xdims , hypers_ls = self.featls)

        # Restore the saved states
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)

    def fitness(self, genotype):
        '''Fitness function '''
        fit = self.fit_GP.evaluate_norm(genotype)
        return fit.detach()

    def fitness_with_grads(self, genotype):
        '''Fitness function '''
        fit = self.fit_GP.evaluate_norm(genotype)
        return fit

    def torch_fitness(self, genotype):
        fit = self.fitness_with_grads(genotype)
        b = self.feature_fun(genotype)
        return fit, b

    def fitness_fun(self, X ):
        '''Function wrapper
        '''
        t = type(X)
        s = np.shape(X)
        ms = np.shape(X[0])
        if t == torch.Tensor:
            return(self.fitness(X))
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

    # def feature_fun(self, X ):
    #     '''Function wrapper
    #     '''
    #     t = type(X)
    #     s = np.shape(X)
    #     ms = np.shape(X[0])
    #     assert t == np.ndarray, 'Input to the fitness function must be an array'
    #     assert s == (1,self.xdims) or ms == (self.xdims,), 'genome is the wrong shape'
    #     assert (X >= self.Xconstraints[:,0]).all() , 'The point is outside the box constraints (lower bound)'
    #     assert (X <= self.Xconstraints[:,1]).all() , 'The point is outside the box constraints (Upper bound)'
    #     ## Single Point    
    #     if s == (1,self.xdims):
    #         X = torch.from_numpy(np.double(X[0])).reshape(-1,self.xdims)
    #         feat1 = self.feat_GP1.evaluate_norm(X).detach().numpy()[0][0]
    #         feat2 = self.feat_GP2.evaluate_norm(X).detach().numpy()[0][0]
    #         return(np.array([feat1,feat2]))
    #     else:
    #         X = torch.from_numpy(X).reshape(-1,self.xdims)
    #         feat1 = [self.feat_GP1.evaluate_norm(x).detach().numpy()[0][0] for x in X]
    #         feat2 = [self.feat_GP2.evaluate_norm(x).detach().numpy()[0][0] for x in X]
    #         return(np.array([feat1,feat2]).T)


    # def fitness_fun(self, X):
    #     '''Function wrapper
    #     '''
    #     ## Single Point    
    #     Xconstraints = torch.from_numpy(self.Xconstraints).double()
    #     assert (X >= Xconstraints[:,0]).all() , 'The point is outside the box constraints (lower bound)'
    #     assert (X <= Xconstraints[:,1]).all() , 'The point is outside the box constraints (Upper bound)'
    #     return(self.fit_GP.evaluate_norm(X).squeeze(-1))

    # def feature_fun(self, X):
    #     '''Function wrapper
    #     '''
    #     ## Single Point 
    #     istensor = isinstance(X, torch.Tensor)
    #     if istensor:
    #         Xconstraints = torch.from_numpy(self.Xconstraints).double()
    #         assert (X >= Xconstraints[:,0]).all() , 'The point is outside the box constraints (lower bound)'
    #         assert (X <= Xconstraints[:,1]).all() , 'The point is outside the box constraints (Upper bound)'
    #         feat1 = self.feat_GP1.evaluate_norm(X)
    #         feat2 = self.feat_GP2.evaluate_norm(X) 
    #         return(torch.stack((feat1,feat2)).squeeze(-1).T)
    #     else:
    #         Xconstraints = self.Xconstraints
    #         assert (X >= Xconstraints[:,0]).all() , 'The point is outside the box constraints (lower bound)'
    #         assert (X <= Xconstraints[:,1]).all() , 'The point is outside the box constraints (Upper bound)'
    #         feat1 = self.feat_GP1.evaluate_norm(torch.from_numpy(X)).detach().numpy()[0][0]
    #         feat2 = self.feat_GP2.evaluate_norm(torch.from_numpy(X)).detach().numpy()[0][0]
    #         return(np.array([feat1,feat2]).T)
    def simple_feature1(self, X):
        x = (X[..., 0] - 1)
        y = (X[..., 1] - 1) 
        return -x

    def simple_feature2(self, X):
        x = (X[..., 0] - 1) 
        y = (X[..., 1] - 1) 
        return -y

    def feature_fun(self, genomes):
        if type(genomes) == np.ndarray:
            genomes = torch.from_numpy(genomes)
            changeback = True
        else:
            changeback = False
        feat1 = self.simple_feature1(genomes)
        feat2 = self.simple_feature2(genomes)
        if changeback:
            return torch.stack([feat1, feat2], dim=1).squeeze(0).numpy()
        else:
            return torch.stack([feat1, feat2], dim=1)

    def feat_fun(self, genomes):
        return(self.feature_fun(genomes))

if __name__ == '__main__':
    # plot the fitness and descriptors.
    import matplotlib.pyplot as plt
    from matplotlib import cm
    for i in range(1):
        domain = SyntheticGP10d2f(seed = 133 + i, feature_resolution = 3)
        X = torch.rand(10000,2).detach()
        Y = domain.fitness(X).detach()
        Z = domain.BOtorch_feature_fun(X).detach()
        # change size of scatter plot points
        plt.rcParams['scatter.marker'] = '.'
        scatter = plt.scatter(Z[:,0],Z[:,1], c = Y, cmap = cm.viridis)
        colorbar = plt.colorbar(scatter)
        plt.xlim(0,1)
        plt.ylim(0,1)   
        plt.show()

