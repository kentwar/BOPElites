'''
An algorithm consists of the following ingredients

1. Surrogates
2. An acquisition function
3. An archive
4. A sampling algorithm (for initial point selection)
5. An Optimizer
'''
import pdb


## Current State:
# Now can use Surrogates to find diverse initial points
# TODO: Get acquisition function working
# TODO: Make sure archive works
# TODO: Cutoff Value
# TODO: invalid regions 

from surrogates.GP import buildGP_from_XY
import numpy as np
import logging , torch, os , pickle , random, copy, sys, time, socket
import matplotlib.pyplot as plt
import psutil
from archives.archives import structured_archive
import itertools as it

class algorithm():

    def __init__(self,  domain, QDarchive, acq_fun , optimizer, resolutions = None, pred_maker = None, seed = None, **kwargs):
        self.resolutions = resolutions
        self.domain = domain
        self.acq_fun = acq_fun
        self.QDarchive = QDarchive
        self.initpointdict = {}  # Stores the potential initial points
        self.descriptors = None
        self.fitness = None
        self.x = None
        self.progress = None
        self.noFitProgress = 0
        self.noProgress = 0
        self.kwargs = kwargs
        self.known_features = self.kwargs['known_features']
        self.test_mode = self.kwargs['test_mode']
        self.init_beta = self.kwargs['init_beta']
        if 'stable_beta' not in self.kwargs.keys():
            self.stable_beta = False
        else:
            self.stable_beta = self.kwargs['stable_beta']
        self.beta = copy.copy(self.init_beta)
        self.set_seed(seed)
        self.initialise(10 * self.domain.xdims)
        self.start_data_saving()
        self.setup_logging()
        self.mispredict = 0
        self.mispredicted = 0

        # initialise optimizer
        self.optimizer = optimizer(self.acq_fun_eval.evaluate, self.domain, init_beta = 0)
        self.pred_maker = pred_maker


    def initialise(self, n: int):
        self.remote_server = socket.gethostname()
        torch.set_grad_enabled(False)
        print('Initialising Algorithm')
        init_sample = self.domain.get_sample(n)
        self.update_storage(init_sample)
        #print(self.x[0].type())
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
        self.standardise_stdarchive()
        self.pointdict = {}
        self.num_niches = self.QDarchive.get_num_niches()
        torch.set_grad_enabled(True)


    def calculate_fitness(self , update = True, archive = None):
        '''
        calculates the fitness of the points in the archive
        by summing over all fitnesses
        '''
        if archive == None:
            archive = self.QDarchive
        if self.progress.shape[0] == 0:
            self.calculate_progress()
            return(self.progress[-1])
        progress = archive.calculate_fitness()
        
        if update:
            self.progress = torch.cat((self.progress, torch.tensor(progress).unsqueeze(0)))
        return(progress)

    def calculate_progress(self):
        '''
        calculates the progress of the algorithm
        using the current points
        '''
        self.progress = torch.tensor([])
        self.progressarchive = copy.copy(self.QDarchive)
        self.progressarchive.fitness[:] = float('nan')
        # a range that counts in steps of 10
        for c,i in enumerate(self.x):
            new_obs = [i, self.fitness[c], self.descriptors[c]]
            ymean = self.fitness[:c].mean(); ystd = self.fitness[:c].std()
            self.progressarchive.updatearchive([new_obs], ymean, ystd)
            progress = np.nansum(self.progressarchive.fitness.flatten())
            self.progress = torch.cat((self.progress, torch.tensor([progress])))

    def iterate(self, n: int, iteration = None):
        self.printc('', line = True, log = True)
        

        if not iteration == None:
            n_evaluations = self.x.shape[0]
            self.printc(f'Iteration {iteration + 1}, {n_evaluations} evaluations', log = True)

        #Calculate Beta
        if iteration < len(self.beta_range):
            self.beta = self.beta_range[iteration]
        else:
            self.beta = 0

        self.printc(f'Beta: {self.beta}', log = True)
        # Get the best point
        self.optimizer.set_obj(self.acq_fun_eval.evaluate, self.beta)
        bestpoint, X, F = self.run_optimiser(n)
        
        # Evaluate new point
        
        new_obs = self.evaluate_new_point(bestpoint)
        predicted_region = self.predict_region(new_obs[0])

        ## adjust misprediction value
        misprediction_occured , total_value = self.acq_fun_eval.check_for_misprediction(new_obs[0])
        self.printc(f'Acquisition Value {total_value[0].item()}', log = True)



        # Update storage, surrogates, archive, acquisition function and optimizer.
        self.update_storage([new_obs])
        self.createmodels()
        # access training data of the fitness model
        ymean = self.fitness.mean(); ystd = self.fitness.std()
        new_obs[2] = torch.tensor(new_obs[2], dtype = torch.double)
        self.QDarchive.updatearchive([new_obs], ymean, ystd)
        self.acq_fun_eval = self.acq()
        self.optimizer.set_obj(self.acq_fun_eval.evaluate, self.beta)


            
        # Update standardisation
        self.acq_fun.set_fstar0(self.acq_fun,-ymean/ystd.item())
        
        # Save the best points for future optimisation runs
        self.save_best_points(X, F)


        self.printc('Observation step:', color = 'g', newline=True)
        current_fitness = self.calculate_fitness()

        #converged = total_value < 0.1 or 
        if self.progress[-1] == self.progress[-2]:
            self.noFitProgress +=1
        else:
            self.noFitProgress = 0
        if self.QDarchive.get_num_niches() < len(self.pointdict):
            self.noProgress = np.max([0, self.noProgress -1])
        elif self.num_niches == self.QDarchive.get_num_niches():
            self.noProgress +=1
        else:
            self.noProgress = 0
        #converged = self.noProgress > self.domain.xdims * self.domain.fdims 
        converged_by_regions = self.noProgress > 2 * np.sqrt(np.prod(self.QDarchive.feature_resolution))
        converged_by_fitness = self.noFitProgress > 2 * np.sqrt(np.prod(self.QDarchive.feature_resolution))
        converged = converged_by_regions or converged_by_fitness

        if self.noFitProgress > 5:
            self.printc('refreshing point storage', color = 'g', newline=True)
            self.pointdict = {}
            # new_sobol = self.domain.get_sample(np.prod(self.QDarchive.feature_resolution))
            # self.update_storage(new_sobol)
            # self.update_storage([new_obs])
            # self.createmodels()
            # # access training data of the fitness model
            # ymean = self.fitness.mean(); ystd = self.fitness.std()
            # new_obs[2] = torch.tensor(new_obs[2], dtype = torch.double)
            # self.QDarchive.updatearchive([new_obs], ymean, ystd)
            # self.acq_fun_eval = self.acq()
            # self.optimizer.set_obj(self.acq_fun_eval.evaluate, self.beta)


            
        # Update standardisation
        self.acq_fun.set_fstar0(self.acq_fun,-ymean/ystd.item())

        if converged:
            self.noProgress = 0
            resolutions = self.resolutions
            res_index = resolutions.index(self.QDarchive.feature_resolution)
            new_res_index = np.min([res_index + 1, len(resolutions) - 1])
            if resolutions[new_res_index] != self.QDarchive.feature_resolution:
                self.upscale(self.QDarchive, resolutions[new_res_index])

        # calculate misprediction
        no_improvement = self.progress[-1] == self.progress[-2]
        if misprediction_occured and no_improvement:
            self.mispredict +=1  
            
            self.QDarchive.mispredicted += np.max([self.mispredict, np.sqrt(self.QDarchive.feature_resolution[-1])])
             
            self.cutoff = self.acq_fun_eval.calculate_cutoff()
            self.printc(f'Misprediction detected, adapting cutoff to {self.cutoff}', color = 'r')
        else:
            self.mispredict = 0 #np.max([0, self.mispredict - 1])
            self.cutoff = self.acq_fun_eval.calculate_cutoff()
            self.printc(f'cutoff {self.cutoff}', log = True)
        print(f'self.mispredict {self.mispredict}')
        print(f'self.mispredicted {self.QDarchive.mispredicted}') 
        # Provide feedback to user
        self.terminal_feedback(new_obs, predicted_region, current_fitness)


        self.save_data()

        self.standardise_stdarchive()



    def terminal_feedback(self, new_obs, predicted_region, current_fitness):
        '''
        Provides terminal feedback
        '''
        self.printc(f'New point: {new_obs[0].numpy()}', log = True)
        self.printc(f'New fitness: {new_obs[1]}', log = True)
        self.printc(f'New Descriptor: {new_obs[2].numpy()}', log = True)
        prediction = predicted_region.numpy()[0]
        true = self.QDarchive.get_region(new_obs[0]).numpy()
        self.printc(f'Predicted region:  {prediction}, True region: {true}', log = True)
        self.printc(f'Current fitness: {current_fitness}', log = True)
        self.num_niches = self.QDarchive.get_num_niches()
        self.printc(f'Num filled regions: {self.num_niches}', log = True)

    def run_optimiser(self, n: int):
        '''
        Runs the optimiser to find the next point to evaluate
        '''
        pointnotfound = True

        while pointnotfound:
            self.printc('Finding diverse x0:', color = 'g', newline=True)
            
            x0 = self.get_diverse_initial_points(n)
            print(f'found points')
            print([self.acq_fun_eval.evaluate(x.reshape(-1,self.domain.xdims), 0).item() for x in x0])
            # Run optimisation
            self.printc('Acquisition step:', color = 'g', newline=True)
            
            X, F = self.optimizer.run_many(x0)
            bestpoint = X[np.argmax(F)]
            print(f'bestvalue {np.max(F)}')
            #print(np.array([self.acq_fun_eval.grad_vectorised_evaluate(torch.tensor(x).reshape(-1,4), 0).item() for x in X]))
            #print(np.array([self.acq_fun_eval.grad_vectorised_evaluate(torch.tensor(x).reshape(-1,4), 0).item() for x in X]) -np.array([self.acq_fun_eval.grad_vectorised_evaluate(x.reshape(-1,4), 0).item() for x in x0]) )
            if np.max(F) > 0 and not np.isnan(np.max(F)):
                pointnotfound = False
            else:
                self.QDarchive.nopointsfound += 1
                self.printc('Point not found, trying again', log = True)
        return(bestpoint, X,F)



    def evaluate_new_point(self, x):
        '''
        Evaluates a new point on the True functions
        '''
        new_fit = self.domain.fitness_fun(np.array(x).reshape(-1, self.domain.xdims))
        new_desc = self.domain.feature_fun(np.array(x).reshape(-1, self.domain.xdims))
        new_obs = [torch.tensor(x, dtype = torch.double) , new_fit, new_desc]
        return(new_obs)

    def update_storage(self, observations):
        x = torch.stack([p[0] for p in observations])
        fit = torch.tensor(np.array([p[1] for p in observations]))
        desc = torch.tensor(np.array([p[2] for p in observations]))
        self.update_x(x)
        self.update_fitness(fit)
        self.update_descriptors(desc)
        ## Remove stored points for the index we just found
        last_index = self.QDarchive.nichefinder(desc[-1])
        # try:
        #     del self.pointdict[tuple(last_index.numpy())]
        #     del self.oldpointdict[tuple(last_index.numpy())]
        #     print('deleted points stored from region just found')
        # except:
        #     pass

    def update_x(self, x):
        if self.x == None:
            self.x = x.double()
        else:
            self.x = torch.cat((self.x, x))
            
    
    def update_fitness(self, new_fitness):
        if self.fitness == None:
            self.fitness = new_fitness.double()
        else:
            self.fitness = torch.cat((self.fitness,new_fitness.double()))
            

    def update_descriptors(self, descriptors):       
        if self.descriptors == None:
            self.descriptors = descriptors.double()     
        else:
            self.descriptors = torch.cat((self.descriptors, descriptors.double()))


    def createmodels(self, ):
        torch.set_grad_enabled(True)
        assert self.x.shape[0] > 0 , 'No points loaded in Algorithm'
        train_x = self.x.reshape(-1, self.x.shape[-1])
        #print(type(train_x))
        train_y = self.fitness.reshape(-1,1)
        #print(type(train_y))
        train_yd  = self.descriptors.reshape(-1, self.descriptors.shape[-1])
        self.fitGP = buildGP_from_XY(train_x, train_y, std = True)
        if not self.known_features:
            self.DGPs = [buildGP_from_XY(train_x, train_y.reshape(-1,1), std = False) for train_y in train_yd.T]
        else:
            self.DGPs = None
            #print('Running with known Descriptors, no descriptor model trained')
        torch.set_grad_enabled(False)

    def run(self, n_restarts: int, max_iter: int):
        if self.stable_beta:
            self.beta_range = np.ones(max_iter) * self.beta
        else:
            self.beta_range = np.linspace(self.beta, 0, int((max_iter-self.x.shape[0])*2/10))
        self.restarts = n_restarts
        self.max_iter = max_iter
        for i in range(self.max_iter - self.x.shape[0]):
            if i % 10 == 0:
                self.iterate(n_restarts, i)
            else:
                self.iterate(5, i)


    def acq(self):
        return(self.acq_fun( self.fitGP, self.DGPs, self.domain, self.QDarchive))


    def get_diverse_initial_points(self, n: int):

        m = 10 if self.domain.fdims <2 else 1
        n_random_points = np.clip(m*(self.domain.xdims**2)*np.prod(self.domain.feature_resolution),10000,100000)
        random_x = self.domain.get_Xsample(int(n_random_points))  
        init_x = self.load_previous_points(random_x)  # Loads points from previous pointdict
        init_x = self.gen_elite_children(init_x) # Generates children of elite points
        self.keep_unique_points(init_x)      
        initpoints , initvals = self.select_initial_points(n)
        x0dict = self.pick_x0(initpoints, initvals, n)
        # return a tensor of all the items in the dictionary
        return(torch.stack(list(x0dict.values())))

    def gen_elite_children(self, init_points):
        indexes = it.product(*[range(i) for i in self.domain.feature_resolution])
        for index in indexes:
            if not np.isnan(self.QDarchive.fitness[tuple(index)]):
                elite = self.QDarchive.genomes[tuple(index)]
                children = elite + np.random.normal(0, 1/np.prod(self.domain.feature_resolution), size = (5, self.domain.xdims))
                lb = self.domain.Xconstraints[:,0]
                ub = self.domain.Xconstraints[:,1]
                children = np.clip(children, lb, ub)
                init_points = torch.cat((init_points,children))
        return(init_points)

    # def pick_x0(self, initpoints, initvals, n):
    #     # pick 80% of the best points
    #     x0dict = {}
    #     for i in range(int(n*0.8)):
    #         max_key = max(initvals, key=initvals.get)
    #         if max_key not in x0dict:
    #             x0dict[max_key] = initpoints[max_key]
    #             del initvals[max_key]
        
    #     ## Pick 20% randomly
    #     while len(x0dict) < n:
    #         random_key = random.choice(list(initpoints.keys()))
    #         if random_key not in x0dict:
    #             x0dict[random_key] = self.pointdict[random_key][np.random.randint(0, len(self.pointdict[random_key]))]
    #     return(x0dict)

    def pick_x0(self, initpoints, initvals, n):
        # pick 80% of the best points
        if n > len(initpoints):
            n = len(initpoints)
        x0dict = {}
        for i in range(int(n*0.5)):
            max_key = max(initvals, key=initvals.get)
            if max_key not in x0dict:
                x0dict[max_key] = initpoints[max_key]
                del initvals[max_key]
        ## Pick indexes that do not yet have a point
        all_indexes = list(it.product(*[range(i) for i in self.domain.feature_resolution]))
        empty_indexes = [i for i in all_indexes if np.isnan(self.QDarchive.fitness[tuple(i)])]
        found_empty_indexes = [i for i in empty_indexes if i in initpoints]
        counter = int(min(n*0.2, len(found_empty_indexes)))
        while min(len(found_empty_indexes), counter) > 0:
            random_key = random.choice(found_empty_indexes)
            if random_key in x0dict:
                found_empty_indexes.remove(random_key)
            if random_key not in x0dict:
                x0dict[random_key] = initpoints[random_key]
                found_empty_indexes.remove(random_key)
                counter -=1
        # for i in range(int(min(n*0.2, len(found_empty_indexes)))):
        #     random_key = random.choice(found_empty_indexes)
        #     if random_key not in x0dict:
        #         x0dict[random_key] = initpoints[random_key]
        ## Pick 20% randomly if we can't find empty indexes
        while len(x0dict) < min(n,len(self.pointdict)):
            random_key = random.choice(list(initpoints.keys()))
            if random_key not in x0dict:
                x0dict[random_key] = self.pointdict[random_key][np.random.randint(0, len(self.pointdict[random_key]))]
        return(x0dict)

    def predict_fit(self, x):
        '''
        provides an unstandardised prediction of the fitness
        '''
        fitness_prediction = self.fitGP(x).mean
        trainmu = self.fitness.mean()
        trainstd = self.fitness.std()
        unstd_fit = (fitness_prediction * trainstd) + trainmu
        return(unstd_fit)
        

    def load_previous_points(self, random_points):
        if len(self.pointdict) > 0:            
            points = [self.pointdict[key].reshape(-1) for key in self.pointdict]
            points = torch.cat(points)
            points = torch.cat([points,random_points.reshape(-1)])
            #print('Previous points Loaded from dict')
            try:
                points = torch.cat([points, self.saved_points.reshape(-1)])
                print('Previous points Loaded from previous optimisation')
            except:
                pass
            return(points.reshape(-1, self.domain.xdims))
        else:
            points = random_points
            print('Did Not load previous points')
        return(points)

    def keep_unique_points(self, points):
        try:
            self.pointdict = self.oldpointdict
        except:
            self.pointdict = {}
        if self.DGPs != None:
            region_list = torch.empty([0, self.domain.fdims], dtype = torch.int32) 
            print('estimating regions')
            mem = psutil.virtual_memory().total
            bs = int(mem/((np.prod(self.domain.feature_resolution))/6))
            bs = np.min([100000,bs]) # Batchsize
            index = [i*bs for i in range(int(np.ceil(len(points)/bs))+1)]       
            for c in range(len(index)-1):
                regions = self.predict_region(points[index[c]:index[c+1]])
                region_list = torch.cat([region_list, regions])
            region_list = [tuple(region.numpy()) for region in region_list]
        else:
            points = points[torch.randperm(len(points))]
            descriptors = self.domain.evaluate_descriptors(points)
            region_list = [tuple(self.QDarchive.nichefinder( d )[0].numpy()) for d in descriptors]
        #region_list = [tuple(r.tolist()) for r in region_list]
        for c,index in enumerate(region_list):
            if index not in self.pointdict.keys():
                self.pointdict[index] = points[c].unsqueeze(0)
            else:
                if points[c] not in self.pointdict[index]:
                    self.pointdict[index] = torch.cat((self.pointdict[index], points[c].unsqueeze(0)))

        num = sum( (self.pointdict[index].shape[0] for index in self.pointdict.keys()) )
        num_filled = len(self.pointdict.keys() )
        self.printc(f'{num} points in {num_filled} regions, selecting {self.restarts} initial points', log = True)
        return(np.copy(self.pointdict))


    def select_initial_points(self , n: int):
        '''
        Selects the initial points
        '''
        initpoints = {}
        initvals = {}
        new_pointdict = {}
        for index in self.pointdict:
            x = self.pointdict[index]
            ## Note we check stdfitness here, not fitness
            vals = self.acq_fun_eval.evaluate_init(x)
            is_above_zero = vals > 0 
            x= x[is_above_zero]
            vals = vals[is_above_zero]
            if len(x) > 0:
                x= x[vals.argsort(descending = True)][0:5]
                initpoints[index] = x[0]
                initvals[index] = vals.sort(descending = True).values[0]
                new_pointdict[index] = x
        for index in set(self.pointdict.keys()) - set(new_pointdict.keys()):
            del self.pointdict[index]
        self.pointdict = new_pointdict
        return(initpoints, initvals)

    def set_seed(self, seed):
        '''
        sets the seed for the random number generator
        '''
        if seed == None:
            self.seed = np.random.randint(100000)
        else:
            self.seed = seed
        self.domain.seed = self.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def save_best_points(self, X , F):
        '''
        Saves the best points from the previous optimisation for future
        runs, excluding the point that was previously chosen.
        '''
        self.oldpointdict = self.pointdict
        X = np.array(X)[np.argsort([f[0] for f in F])][:-1]
        X = np.unique(X, axis = 0)
        X = torch.tensor(X, dtype = torch.double).reshape(-1,self.domain.xdims)
        regions = [self.predict_region(x) for x in X]
        for c,region in enumerate(regions):
            region = tuple(region.tolist()[0])
            if region not in self.oldpointdict.keys():
                self.oldpointdict[region] = X[c].unsqueeze(0)
            else:
                if X[c] not in self.pointdict[region]:
                    self.oldpointdict[region] = torch.cat((self.oldpointdict[region], X[c].unsqueeze(0)))


    def start_data_saving(self):
        cwd = os.getcwd()
        domain = self.domain.name
        alg = self.acq_fun_eval.name
        res = self.resolutions[-1]
        if not self.test_mode:            
            self.save_path = f"{cwd}/experiment_data/{domain}/{alg}/{res}/{self.seed}"
            os.makedirs(self.save_path, exist_ok = True)
        else:
            self.save_path = f"{cwd}/experiment_data/{domain}/{alg}/{res}/Test/{self.seed}"
            os.makedirs(self.save_path, exist_ok = True) 
        print('Saving data to: ', self.save_path)
               

    def setup_logging(self):
        # Set up logging
        filename = f'{self.save_path}/log.txt'
        writemode = 'a' if os.path.exists(filename) else 'w'
        logging.basicConfig(filename=f'{self.save_path}/log.txt', filemode=writemode, format='%(message)s')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  
        self.printc('Starting logger', color = 'y',line = True)
        self.printc(f'Domain: {self.domain.name}')
        self.printc(f'Acquisition function: {self.acq_fun_eval.name}', log = True)
        self.printc(f'Seed: {self.seed}', log = True)
        self.printc(f'Running on: {self.remote_server}', log = True)

    def printc(self, text, start = 0, end = 1000, color = 'w', line = False, newline = False, log = False):
        if color == 'y':
            color = '93m'
        elif color == 'bold':
            color = '37m'
        elif color == 'g':
            color = '92m'
        elif color == 'b':
            color = '94m'
        else:
            color = '0m'
        if newline:
            if log:
                self.logger.info('')
            print('')
        if line: 
            if log:
                self.logger.info('--------------------------------------------------------')
            print('--------------------------------------------------------')
        if log:
            self.logger.info(text)
        print(text[:start] + '\033[' + color + text[start:end] + '\033[0m' + text[end:])

    def standardise_fitness(self, fit):
        '''
        Standardises the fitness
        '''
        fit2 = fit - self.y.mean()
        stdfit = fit2 / self.y.std()
        return(stdfit)

    def save_data(self):
        mydir = self.save_path
        fitness_file = f'{mydir}/fitness.pkl'
        descriptors_file = f'{mydir}/descriptors.pkl'
        x_file = f'{mydir}/x.pkl'
        point_dict_file = f'{mydir}/pointdict.pkl'
        param_file = f'{mydir}/params.pkl'
        params = {'seed':self.seed, 
                  'domain':self.domain.name, 
                  'acq_fun':self.acq_fun_eval.name, 
                  'resolutions':self.resolutions, 
                  'test_mode':self.test_mode, 
                  'remote_server':self.remote_server,
                  'mispredicted':self.mispredicted}

        savefiles = {fitness_file:self.fitness, 
                        descriptors_file: self.descriptors,
                        x_file : self.x,
                        point_dict_file : self.pointdict,
                        param_file : params}
        for file in savefiles:
            with open(file, 'wb') as f:
                pickle.dump(savefiles[file], f)


    def standardise_stdarchive(self):
        '''
        Standardises the standardised archive
        '''
        ymean = self.fitness.mean()
        ystd = self.fitness.std()
        self.QDarchive.stdfitness = (self.QDarchive.fitness - ymean)/ystd
        self.acq_fun.set_fstar0(self.acq_fun,-ymean/ystd.item())

    def predict_region(self, x):
        '''
        Vectorized function that returns the index of the niches 
        with maximal probability that x belongs to that niche 
        '''
        return(self.acq_fun_eval.predict_region(x))

    def conformbeh(self,  beh ):
        '''
        conforms behaviour to the domain bounds
        '''
        lb = [bound for bound in self.domain.featmins ]
        ub = [bound for bound in self.domain.featmaxs ]
                
        beh = np.clip(beh , a_min = lb, a_max = ub)
        
        return( list(beh) ) 

    def load_data(self, save_path, n = None, models = True):
        ''' Loads the data from the save path'''
        mydir = save_path
        self.QDarchive.flush()
        fitness_file = f'{mydir}/fitness.pkl'
        descriptors_file = f'{mydir}/descriptors.pkl'
        x_file = f'{mydir}/x.pkl'
        point_dict_file = f'{mydir}/pointdict.pkl'
        params_file = f'{mydir}/params.pkl'
        self.params = {}
        loadfiles = {fitness_file:self.fitness,
                        descriptors_file: self.descriptors,
                        x_file : self.x,
                        point_dict_file : self.pointdict,
                        params_file : self.params
                        }
        for file in loadfiles:
            try:
                with open(file, 'rb') as f:
                
                    loadfiles[file] = pickle.load(f)
            except:
                print(f'unable to load {file}')
        if n != None:
            self.fitness = loadfiles[fitness_file][:n]
            self.descriptors = loadfiles[descriptors_file][:n]
            self.x = loadfiles[x_file][:n]
        else:
            self.fitness = loadfiles[fitness_file]
            self.descriptors = loadfiles[descriptors_file]
            self.x = loadfiles[x_file]
        self.pointdict = loadfiles[point_dict_file]
        self.params = loadfiles[params_file]
        self.reinitialise(models = models)

    def reinitialise(self, n = None, models = True):
        if models:
            print('Creating models')
            self.createmodels()

        print('initialising archive')
        points = [[self.x[c], self.fitness[c], self.descriptors[c]] for c in range(self.x.shape[0])]
        self.QDarchive.flush()
        self.QDarchive.initialise(points, False)

        print('Creating acquisition function')
        self.acq_fun_eval = self.acq()#_fun( self.fitGP, self.DGPs, self.domain, self.QDarchive)
        self.beta = self.init_beta
        self.optimizer.set_obj(self.acq_fun_eval.evaluate, beta = self.init_beta)
        self.standardise_stdarchive()
        self.calculate_progress()
        try:
            self.resolutions = self.params['resolutions']
            self.mispredicted = self.params['mispredicted']
            print('loaded resolutions and mispredicted value')
        except:
            print('unable to load resolutions and mispredicted value')
        #setup standardisation

    def plot_archive2d(self, save = True, save_path = None, archive = None, text = ''):
        '''
        Plots the archive
        '''
        if save_path == None:
            save_path = self.save_path
        if archive == None:
            archive = self.QDarchive
        fig, ax = plt.subplots()
        # plot a heatmap
        vmin = self.domain.lowestvalue
        vmax = self.domain.maxvalue
        im = ax.imshow(archive.fitness, cmap = 'viridis', interpolation = 'nearest', vmin = vmin, vmax = vmax)
        ax.set_xlabel(self.domain.desc1name)
        ax.set_ylabel(self.domain.desc2name)
        fig.colorbar(im)
        fit_val = self.calculate_fitness(update = False, archive = archive)
        fit_val = str(np.round(fit_val,2))
        ax.set_title(f'Fitness = {fit_val} for {self.x.shape[0]} samples, {self.domain.name} function')
        if save:
            fig.savefig(f'{save_path}/{text}archive.png')
        else:
            plt.show()
    
    def plot_convergence(self,save = True, save_path = None):
        '''
        Plots the progress of the algorithm
        '''
        if save_path == None:
            save_path = self.save_path
        fig, ax = plt.subplots()
        x = np.arange(len(self.x))
        self.calculate_progress()
        y = self.progress
        ax.plot(x,y, label = self.acq_fun_eval.name)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fitness')
        ax.set_title(f'Progress of {self.domain.name} function')
        ax.vlines(10*self.domain.xdims, 0, max(self.progress), colors = 'r', linestyles = 'dashed', label = 'initial points')
        ax.set_xscale('log')
        ax.set_yscale('log')
        fit_val = self.calculate_fitness(update = False)
        fit_val = str(np.round(fit_val,2))
        ax.set_title(f'Convergence of {self.domain.name} function, Fitness: {fit_val} for {self.x.shape[0]} samples, ')
        plt.legend()
        if save:
            fig.savefig(f'{save_path}/archive.png')
        else:
            plt.show()

    def upscale(self, archive, resolution):
        '''
        Upscales the new BOP to a higher resolution archive.
        '''
        current_resolution = self.domain.feature_resolution
        self.printc(f'Upscaling from {current_resolution} to {resolution}')
        self.domain.feature_resolution = resolution
        mispredicted = self.QDarchive.mispredicted
        nopointsfound = self.QDarchive.nopointsfound
        self.QDarchive = structured_archive(self.domain)
        self.reinitialise()
        self.QDarchive.mispredicted = mispredicted 
        self.QDarchive.nopointsfound = nopointsfound


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
