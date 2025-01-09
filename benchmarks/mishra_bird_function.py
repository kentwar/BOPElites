from benchmarks.BaseBenchmark import BaseExperiment
import numpy as np
import torch


class Mishra_bird_function(BaseExperiment):
    def __init__(self, feature_resolution, seed=100):

        kwargs = {
            'example_x': [0, 0],
            'Xconstraints': [[0, 1], [0, 1]],
            'featmins': [0, 0],
            'featmaxs': [10, 6],
            'lowestvalue': 0,
            'maxvalue': 190,  # max value is not very important
        }
        self._set_Xconstraints(np.array(kwargs['Xconstraints']))  # input x ranges [min,max]
        self.example_x = kwargs['example_x']
        self.xdims = len(kwargs['example_x'])
        self.fdims = len(kwargs['featmins'])
        self.featmins = kwargs['featmins']
        self.featmaxs = kwargs['featmaxs']
        self.feature_resolution = feature_resolution
        self.lowestvalue = kwargs['lowestvalue']
        self.maxvalue = kwargs['maxvalue']
        self.seed = seed
        self.name = 'mishra_bird_function'
        self.desc1name = 'x position of joints'
        self.desc2name = 'y position of joints'

    def mishra_bird_fitness(self, X):
        # Take the unit values and scale them to the correct range
        x = (X[:, 0] - 1) * 10
        y = (X[:, 1] - 1) * 6
        # Vectorised call to Mishra Bird function
        fit = np.sin(y) * np.exp((1 - np.cos(x)) ** 2) + np.cos(x) * np.exp((1 - np.sin(y)) ** 2) + (x - y) ** 2
        return (fit + 106.7)  # Ammend by a lower value that makes all fitness positive.

    def Torch_mishra_bird_fitness(self,X):
        # Take the unit values and scale them to the correct range
        if X.ndim == 1:
            x = (X[0] - 1) * 10
            y = (X[1] - 1) * 6
        else:
            x = (X[:, 0] - 1) * 10
            y = (X[:, 1] - 1) * 6
        
        # Vectorized call to Mishra Bird function
        if type(X) == torch.Tensor:
            fit = torch.sin(y) * torch.exp((1 - torch.cos(x)) ** 2) + torch.cos(x) * torch.exp((1 - torch.sin(y)) ** 2) + (x - y) ** 2
        else:
            fit = np.sin(y) * np.exp((1 - np.cos(x)) ** 2) + np.cos(x) * np.exp((1 - np.sin(y)) ** 2) + (x - y) ** 2
        return fit + 106.7  # Amend by a lower value that makes all fitness positive.

    def fitness(self, genotype):
        '''Fitness function '''
        #fit = self.mishra_bird_fitness(np.array(genotype).reshape(-1, self.xdims))
        fit = self.Torch_mishra_bird_fitness(genotype)
        return fit#[0]

    def fitness_fun(self, X):
        '''Function wrapper
        '''
        t = type(X)
        s = np.shape(X)
        ms = np.shape(X[0])
        if t == torch.Tensor:
            return(self.Torch_mishra_bird_fitness(X))
        assert t == np.ndarray, 'Input to the fitness function must be an array'
        assert s == (1, self.xdims) or ms == (self.xdims,), 'genome is the wrong shape'
        assert (X >= self.Xconstraints[:, 0]).all(), 'The point is outside the box constraints (lower bound)'
        assert (X <= self.Xconstraints[:, 1]).all(), 'The point is outside the box constraints (Upper bound)'
        ## Single Point   
        if s == (1, self.xdims):
            return (self.fitness(X))
        else:
            return ([self.fitness(x) for x in X])

    def torch_fitness(self, X):
        fit = self.Torch_mishra_bird_fitness(X)
        b = self.feature_fun(X)
        return fit, b

    def simple_feature1(self, X):
        x = (X[..., 0] - 1) * 10
        y = (X[..., 1] - 1) * 6
        return -x

    def simple_feature2(self, X):
        x = (X[..., 0] - 1) * 10
        y = (X[..., 1] - 1) * 6
        return -y

    def feature_fun(self, genomes):
        #genomes = torch.tensor(genomes, dtype=torch.float64)
        feat1 = self.simple_feature1(genomes)
        feat2 = self.simple_feature2(genomes)
        if type(feat1) == np.ndarray:
            return np.stack([feat1, feat2], axis=1)
        return torch.stack([feat1, feat2], dim=1)

    def feat_fun(self, genomes):
        return(self.feature_fun(genomes))

# domain = RobotArm(**kwargs)
