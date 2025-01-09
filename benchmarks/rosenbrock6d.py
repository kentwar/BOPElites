from benchmarks.BaseBenchmark import BaseExperiment
import numpy as np
import torch

class Rosenbrock6d(BaseExperiment):
    def __init__(self, feature_resolution , seed = 100):
        
        kwargs =    {
            'example_x' : [0,0,0,0,0,0] ,
            'Xconstraints' : [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1] ] ,
            'featmins' : [0,0] ,
            'featmaxs' : [1,1] ,
            'lowestvalue' : 400 ,
            'maxvalue' : 1600 ,
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
        self.name = 'rosenbrock6d'
        self.desc1name = 'x position of joints'
        self.desc2name = 'y position of joints'

    def rosenbrock6d(self, x):
        """The Rosenbrock function"""
        return (sum(100.0*(2*x[1:]-2*x[:-1]**2)**2.0 + (1-2*x[:-1])**2.0))

    def rosenbrock6d_torch(self, x):
        """The Rosenbrock function"""
        return torch.sum(100.0 * (2 * x[..., 1:] - 2*x[..., :-1] ** 2) ** 2 + (1 - 2 * x[..., :-1]) ** 2, dim=-1)

    def fitness(self, genotype):
        '''Fitness function '''
        if type(genotype) == torch.Tensor:
            return(self.rosenbrock6d_torch(genotype))
        
        if len(genotype.shape) > 1:
            try:
                genotype = genotype.reshape(self.xdims)
            except:
                raise ValueError('The genotype is not the right shape')
        fit = self.rosenbrock6d(genotype)
        return fit 

    def fitness_fun(self, X):
        '''Function wrapper
        '''
        t = type(X)
        s = np.shape(X)
        ms = np.shape(X[0])
        if t == torch.Tensor:
            return(self.fitness(X))

        assert t == np.ndarray, 'Input to the fitness function must be an array'
        assert s == (1,self.xdims) or ms == (self.xdims,), 'genome is the wrong shape'
        assert (X >= self.Xconstraints[:,0]).all() , 'The point is outside the box constraints (lower bound)'
        assert (X <= self.Xconstraints[:,1]).all() , 'The point is outside the box constraints (Upper bound)'
        ## Single Point    
        if s == (1,self.xdims):
            return(self.fitness(X[0]) )
        else:
            return([self.fitness(x) for x in X])

    # def feature_fun(self, X):
    #     '''Rosenbrock - constrained
    #     '''
    #     t = type(X)
    #     s = np.shape(X)
    #     ms = np.shape(X[0])
    #     assert t == np.ndarray, 'Input to the fitness function must be an array'
    #     assert s == (1,self.xdims) or ms == (self.xdims,), 'genome is the wrong shape'
    #     if s == (1,self.xdims):        
    #         feat1 =  0.5*(X[0,0]+X[0,1]) 
    #         feat2 =  (X[0,2]-1)**2
    #         return(np.array([feat1,feat2]))
    #     else:
    #         feat1 = [0.5*(r[0]+r[1]) for r in X]
    #         feat2 = [(r[2]-1)**2 for r in X]
    #         combinedlist = zip(feat1, feat2)
    #         return([list(c) for c in combinedlist])

    def feature_fun(self, X):
        '''Rosenbrock - constrained
        '''
        t = type(X)
        s = X.shape
        ms = X[0].shape if len(X) > 0 else ()
        assert isinstance(X, (np.ndarray, torch.Tensor)), 'Input to the fitness function must be an array or tensor'
        assert s == (1,self.xdims) or ms == (self.xdims,), 'genome is the wrong shape'
        istensor = isinstance(X, torch.Tensor)
        if s == (1,self.xdims):
            feat1 =  0.5*(X[0,0]+X[0,1]) 
            feat2 =  (X[0,2]-1)**2
            if istensor:
                return torch.stack([feat1,feat2])
            else:
                return np.array([feat1,feat2])
        else:
            if istensor:
                feat1 = 0.5*(X[:,0]+X[:,1])
                feat2 = (X[:,2]-1)**2
                c = torch.cat((feat1.unsqueeze(-1), feat2.unsqueeze(-1)), dim=-1)
                d = torch.unbind(c, dim=-1)
                return torch.stack(d, dim = 1)
            else:
                feat1 = [0.5*(r[0]+r[1]) for r in X]
                feat2 = [(r[2]-1)**2 for r in X]
                combinedlist = zip(feat1, feat2)
                return [list(c) for c in combinedlist]
    
    def feat_fun(self, X):
        return(self.feature_fun(X))


