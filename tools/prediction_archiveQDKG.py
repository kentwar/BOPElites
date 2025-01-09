import numpy as np
import torch, copy, sys
import itertools as it
from surrogates.GP import buildGP_from_XY
from tqdm import tqdm

class prediction_archive():
    '''
    Class for a prediction archive generator.

    Requires the domain, archive, and optimizer to be passed in.
    the prediction map must be passed the current archive
    to be used for prediction.
    '''
    def __init__(self, algorithm, PMoptimizer, acq_fun, sideloaded_points = None, return_added = False, **kwargs):
        self.algorithm = algorithm
        self.domain = algorithm.domain
        self.QDarchive = algorithm.QDarchive
        self.name = 'prediction_archive'
        self.dtype = torch.double
        self.optimizer = PMoptimizer        
        self.known_features = kwargs['known_features']
        try:
            self.return_pred = kwargs['return_pred']
        except:
            self.return_pred = False

        self.DGPs = None
        try: 
            self.fitGP = algorithm.fitGP
            
            if not self.known_features:
                try:
                    self.DGPs = algorithm.DGPs
                except:
                    print('No descriptor model found. known_features set to True')
                    self.known_features = True
                    
        except:
            print('No way to build models at this time')

        self.acq_fun = acq_fun

        try:
            n_children = kwargs['PM_params'][0]
            n_generations = kwargs['PM_params'][1]
            #print('found params')
        except:
            n_children = 2**8
            n_generations = 2**7
            print('using default params')
        
        if self.return_pred:
            self.true_pred_archive , self.pred_archive,  self.added = self.get_PA(n_children, n_generations, sideloaded_points= sideloaded_points, return_added = return_added)
        else:
            self.true_pred_archive = self.get_PA(n_children, n_generations, sideloaded_points= sideloaded_points)


    def get_PA(self,  n_children, n_iters, sideloaded_points = None, return_added = False):
        '''
        generates a prediction archive by running MAP-Elites over the surrogates
        INPUTS:
            archive: Initial archive to be used for prediction
            acq_fun: acquisition function to be used for prediction
        OUTPUTS:
            pred_archive: prediction archive 
        '''
        archive = self.QDarchive 

        acq_fun = self.acq_fun
        print('Generating prediction archive')
        pred_archive = archive.return_copy()
        # Seeding Prediction map

        pred_archive = self.seed_predmap(pred_archive, acq_fun, sideloaded_points)

        # Run MAP-Elites on prediction map
        # Use 2**7 children over 2**8 iterations
        if return_added:
            pred_archive, added = self.optimize_predmap(pred_archive, n_children, n_iters, return_added = return_added)
        else:
            pred_archive = self.optimize_predmap(pred_archive, n_children, n_iters)

        # return prediction map
        true_archive = self.pred_to_true_fit(pred_archive, self.domain.fitness_fun)
        if self.return_pred:
            if return_added:
                return(true_archive, pred_archive, added)
            else:
                return(true_archive, pred_archive)
        return(true_archive)
    
    def get_true_value(self):
        '''
        returns the true value of the prediction archive
        '''
        value = np.nansum(self.pred_archive.fitness)
        return(value)

    def seed_predmap(self, pred_archive, acqfun, sideloaded_points = None):
        '''
        seeds the prediction archive with the best points from the archive
        sideloaded archive is points that will be evaluated
        '''
        # Seeding Prediction map
        index_list = it.product(*[range(res) for res in self.domain.feature_resolution])
        for index in index_list:
            index = tuple(index)
            x = pred_archive.genomes[index]
            fit = acqfun(x.reshape(-1, self.domain.xdims))
            fitnan = torch.isnan(fit)
            x_nan = torch.isnan(x).any()
            if not (x_nan and fitnan):
                pred_archive.genomes[index] = x
                pred_archive.fitness[index] = fit

        if sideloaded_points is not None:
            for x in sideloaded_points:
                beh = self.domain.feature_fun(x.reshape(-1, self.domain.xdims))
                index = tuple(self.QDarchive.nichefinder(beh).numpy()[0])
                fit = acqfun(x.reshape(-1, self.domain.xdims))
                fitnan = torch.isnan(fit)
                x_nan = torch.isnan(x).any()
                if not (x_nan and fitnan) and (fit > pred_archive.fitness[index] or torch.isnan(pred_archive.fitness[index])):
                    pred_archive.genomes[index] = x
                    pred_archive.fitness[index] = fit
        return(pred_archive)

    def optimize_predmap(self, pred_archive,  n_children, n_iters, known_features = True, return_added = False):
        '''
        optimizes the prediction archive
        '''
        acqfun = self.acq_fun
        pred_optimizer = self.optimizer( self.domain , pred_archive)
        for i in range(n_iters): 
            pred_optimizer.archive = pred_archive
            children = pred_optimizer.run(n_children)
            children_y = acqfun(children)
            if self.known_features:
                children_desc = self.domain.feature_fun(children.reshape(-1, self.domain.xdims))
            else:
                children_desc = torch.stack([model(children).mean for model in self.DGPs]).T
            pred_obs = [[children[c], children_y[c], children_desc[c]] for c in range(children.shape[0])]
            added = pred_archive.updatearchive(pred_obs, 0, 0, return_added = return_added)
        
        if return_added:
            return(pred_archive, added)
            #return(pred_archive, children)
        return(pred_archive)

    def predict_descriptors(self, children):
        '''
        predicts the descriptor values for a set of children
        '''
        descriptors = [model(children)[0] for model in self.DGPs]
        return(descriptors)



    def pred_to_true_fit(self, archive, fitness):
        '''
        converts a prediction archive to a true archive
        '''
        true_archive = archive.return_copy()
        index_list = it.product(*[range(res) for res in self.domain.feature_resolution])
        for index in index_list:
            index = tuple(index)
            x = true_archive.genomes[index]
            fit_nan = torch.isnan(true_archive.fitness[index])
            genome_nan = torch.isnan(x).any()
            fit_not_nan = not fit_nan and not genome_nan
            if fit_not_nan:
                true_behaviour = self.domain.feature_fun(x.numpy().reshape(-1, self.domain.xdims))
                true_index = tuple(self.QDarchive.nichefinder(torch.tensor(true_behaviour)).numpy()[0])
                if true_index == index:
                    true_archive.genomes[index] = x
                    if type(true_archive.fitness) == torch.Tensor:
                        true_archive.fitness[index] = fitness(x.reshape(-1, self.domain.xdims))
                    else:
                        true_archive.fitness[index] = fitness(x.numpy().reshape(-1, self.domain.xdims))
                else:
                    true_archive.fitness[index] = 0
        return(true_archive)


    def createmodels(self, ):
        self.x = self.algorithm.x
        self.fitness = self.algorithm.fitness
        self.descriptors = self.algorithm.descriptors
        assert self.x.shape[0] > 0 , 'No points loaded in Algorithm'
        train_x = self.x.reshape(-1, self.x.shape[-1])
        #print(type(train_x))
        train_y = self.fitness.reshape(-1,1)
        #print(type(train_y))
        train_yd  = self.descriptors.reshape(-1, self.descriptors.shape[-1])
        self.fitGP = buildGP_from_XY(train_x, train_y, std = True)
        if not self.known_features:
            self.DGPs = buildGP_from_XY(train_x, train_yd)
        else:
            self.DGPs = None
            #print('Running with known Descriptors, no descriptor model trained')

    # def progressbar(self, it, prefix="", size=60, out=sys.stdout): # Python3.3+
    #     count = len(it)
    
    #     def show(j):
    #         x = int(size*j/count)
    #         print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count), 
    #                 end='\r', file=out, flush=True)
    #     show(0)
    #     for i, item in enumerate(it):
    #         yield item
    #         show(i+1)
    #     print("\n", flush=True, file=out)

