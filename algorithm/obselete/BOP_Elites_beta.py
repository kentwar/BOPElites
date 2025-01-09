'''
An algorithm consists of the following ingredients

1. Surrogates
2. An acquisition function
3. An archive
4. A sampling algorithm (for initial point selection)
5. An Optimizer
'''
from surrogates.GP import buildGP_from_XY
from algorithm.BOP_Elites import algorithm as BOP
import numpy as np
import logging , torch, os , pickle , random, copy, socket
import matplotlib.pyplot as plt



class algorithm(BOP):

    def __init__(self,  domain, QDarchive, acq_fun ,  optimizer,resolutions= None, pred_maker = None,seed = None, **kwargs):
        self.resolutions = resolutions
        self.domain = domain
        self.acq_fun = acq_fun
        self.QDarchive = QDarchive
        self.initpointdict = {}  # Stores the potential initial points
        self.descriptors = None
        self.fitness = None
        self.x = None
        self.progress = None
        self.kwargs = kwargs
        self.known_features = self.kwargs['known_features']
        self.test_mode = self.kwargs['test_mode']
        self.init_beta = self.kwargs['init_beta']
        self.beta = copy.copy(self.init_beta)
        self.set_seed(seed)
        self.initialise(10 * self.domain.xdims)
        self.start_data_saving()
        self.setup_logging()
        self.noFitProgress = 0
        # initialise optimizer
        self.optimizer = optimizer(self.acq_fun_eval.evaluate, self.domain, self.init_beta)


    def initialise(self, n: int):
        remote_server = socket.gethostname()
        print('Initialising Algorithm')
        init_sample = self.domain.get_sample(n)
        self.update_storage(init_sample)
        print(self.x[0].type())
        print('Creating models')
        self.createmodels()

        ## make in to points list
        print('initialising archive')
        points = [[self.x[c], self.fitness[c], self.descriptors[c]] for c in range(self.x.shape[0])]
        
        self.QDarchive.initialise(points, False)
        self.calculate_progress()
        print('Creating acquisition function')
        self.acq_fun_eval = self.acq()#_fun( self.fitGP, self.DGPs, self.domain, self.QDarchive)
        
        #setup standardisation
        ymean = self.fitness.mean(); ystd = self.fitness.std()
        self.acq_fun.set_fstar0(self.acq_fun,-ymean/ystd.item())

        self.pointdict = {}

    def iterate(self, n: int, iteration = None):
        self.printc('', line = True, log = True)
        

        if not iteration == None:
            self.printc(f'Iteration {iteration + 1}', log = True)
        self.printc('Finding diverse x0:', color = 'g', newline=True)
        x0 = self.get_diverse_initial_points(n)
        
        #Calculate Beta
        if iteration < len(self.beta_range):
            self.beta = self.beta_range[iteration]
        else:
            self.beta = 1

        self.printc(f'Beta: {self.beta}', log = True)
        # Run optimisation
        self.optimizer.set_obj(self.acq_fun_eval.evaluate, self.beta)
        X, F = self.optimizer.run_many(x0)
        bestpoint = X[np.argmax(F)]
        self.printc('Acquisition step:', color = 'g', newline=True)
        self.printc(f'Best point: {bestpoint}', log = True)
        self.printc(f'Acquisition value: {np.max(F)}', log = True)
        new_obs = self.evaluate_new_point(bestpoint)
        
        # Update surrogates, archive and storage.
        self.update_storage([new_obs])
        self.createmodels()
        self.acq_fun_eval = self.acq()
        self.optimizer.set_obj(self.acq_fun_eval.evaluate, self.beta)
        # Update standardisation
        ymean = self.fitness.mean(); ystd = self.fitness.std()
        self.acq_fun.set_fstar0(self.acq_fun,-ymean/ystd.item())
        
        # Turn behaviour in to a tensor and update archive
        new_obs[2] = torch.tensor(new_obs[2], dtype = torch.double)
        self.QDarchive.updatearchive([new_obs], ymean, ystd)

        # Provide terminal feedback
        self.printc('Observation step:', color = 'g', newline=True)
        current_fitness = self.calculate_fitness()
        self.printc(f'Current fitness: {current_fitness}', log = True)
        num_niches = self.QDarchive.get_num_niches()
        self.printc(f'Num filled regions: {num_niches}', log = True)
        self.save_data()

        self.standardise_stdarchive()

        last_index = self.QDarchive.nichefinder(self.descriptors[-1])
        
        try:
            del self.pointdict[tuple(last_index.numpy())]
        except:
            pass


            
    


    def run(self, n_restarts: int, max_iter: int):
        self.beta_range = np.linspace(self.init_beta, 1, int(np.ceil((max_iter-self.x.shape[0])/4)+1))
        self.restarts = n_restarts
        self.max_iter = max_iter
        for i in range(self.max_iter - self.x.shape[0]):
            self.iterate(n_restarts, i)

    def inherit(self, old_algorithm):
        '''
        inherits data from an old algorithm
        '''
        self.x = old_algorithm.x
        self.fitness = old_algorithm.fitness
        self.descriptors = old_algorithm.descriptors
        self.save_path = old_algorithm.save_path
        self.logger = old_algorithm.logger
        self.reinitialise()


    # check if stale
    
    ## Pred map
    # generate pred maps

#print part of a string a color
