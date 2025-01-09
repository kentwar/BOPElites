from botorch.sampling import SobolEngine
import copy
from numpy import isnan
import torch

class BaseExperiment:
    '''
    A Base class for benchmark experiments
    '''
    def __init__(self, 
                feature_dims , 
                example = None,
                feature_resolution = None,
                ME_params = None , 
                seed = 100):
        '''
        The benchmark is where all details about the current experiment are 
        stored.
        '''
        
        #self.valid_ranges = constraints    #input x ranges [min,max]
        self.feat_dims = len( featmins )
        self.feat_mins = featmins
        self.feat_maxs = featmaxs
        self.valid_feat_ranges = [ [featmins[ i ], featmaxs[ i ] ] for i in range(len( featmins ))] 
        self.GP_Params = GP_params
        self.example = example_genome
        self.lowestvalue = 0
        self.maxvalue = 1
        self.feature_resolution = feature_resolution
        self.ME_params = ME_params
        self.seed = seed
        self.desc1name = 'desc1'
        self.desc2name = 'desc2'


    def _set_BOP_parameters(self):
        pass

    def _set_PM_parameters(self , mprob = 0, 
                                n_children = 2**6, 
                                mut_sigma = 0.1,
                                n_gens  = 2**7):
        '''
        These are the parameters used in generating the Prediction map 
        with MAP-Elites
        '''
        # mprob = Probability of performing 1 slice cross mutation
        # n_children = Number of children in Population
        # mut_sigma = Amount of Gaussian Noise to apply in mutation step       
        # n_gens = Number of generations to run MAP-Elites for
        self.ME_params = { 'mprob': mprob  , 
                        'n_children': n_children  , 
                        'mut_sigma' : mut_sigma , 
                        'n_gens' : n_gens }
        PM_params = { 'mprob': mprob  , 
                        'n_children': 2**7  , 
                        'mut_sigma' : mut_sigma , 
                        'n_gens' : 2**8 }

    def fitness_fun(self):
        pass

    def feature_fun(self):
        pass

    def _set_Xconstraints(self, Xconstraints):
        '''Sets contraints on the search domain'''
        self.Xconstraints = Xconstraints
    
    def get_constraints(self,):
        return(self.Xconstraints)
    
    def get_sample(self, n): 
        missing_points = copy.copy(n)
        init_samples = [] # [x, fit, desc]
        while missing_points > 0:
            self.sobolengine = SobolEngine(self.xdims, scramble=True, seed = self.seed)
            init_x = self.sobolengine.draw(n, dtype=torch.double)
            rescaled_x = self.rescale_X( init_x )
            init_samples , missing_points = self.sample(rescaled_x, missing_points, init_samples)
            print(f'sampled {n - missing_points} points of {n}')
        return(init_samples)

    def rescale_x(self, x):
        for count, dim in enumerate(x):
            min, max = self.Xconstraints[count]
            magnitude = max-min
            x[count] = dim * magnitude + min
        return(x)

    def rescale_X(self, X):
        rescaled_X = torch.empty(len(X), len(X[0]), dtype = torch.double)
        for count, x in enumerate(X):
            rescaled_X[count] = self.rescale_x(x)

        return( rescaled_X )


    def sample(self, rescaled_X, missing_points, init_samples = []):
        x = rescaled_X
        fitness = self.fitness_fun(rescaled_X.numpy().reshape(-1, self.xdims))
        descriptors = self.feature_fun(rescaled_X.numpy().reshape(-1, self.xdims))
        invalid = isnan(fitness) + isnan(descriptors).any()
        num_valid = min([missing_points,sum(~invalid)])
        samples = []
        for count in range(len(invalid)):
            if missing_points == 0:
                break
            if invalid[count]:
                pass
            else:
                samples.append([x[count], fitness[count], descriptors[count]])
                missing_points -= 1
        if init_samples == []:
            init_samples = samples
        else:
            init_samples += samples
        return(init_samples, missing_points)
    
    def evaluate_descriptors(self, rescaled_X):
        descriptors = self.feature_fun(rescaled_X.numpy().reshape(-1, self.xdims))
        return(descriptors)

    def get_Xsample(self, n ):
        sampler = SobolEngine(self.xdims, scramble=True)
        init_x = sampler.draw(n, dtype=torch.double)
        rescaled_x = self.rescale_X( init_x )
        return(rescaled_x)