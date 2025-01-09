'''
An algorithm consists of the following ingredients

1. Surrogates
2. An acquisition function
3. An archive
4. A sampling algorithm (for initial point selection)
5. An Optimizer
'''
from surrogates.GP import buildGP_from_XY
import numpy as np
import logging , torch, os , pickle , random, copy, socket
import matplotlib.pyplot as plt
from archives.archives import structured_archive
import psutil
import itertools as it


class algorithm():

    def __init__(self,  
                domain, 
                QDarchive, 
                acq_fun ,  
                optimizer, 
                resolutions = None ,
                seed = None, 
                **kwargs):
            
        self.resolutions = resolutions
        self.domain = domain
        self.acq_fun = acq_fun
        self.name = 'BOP_Elites'
        self.QDarchive = QDarchive
        self.initpointdict = {}  # Stores the potential initial points
        self.descriptors = None
        self.fitness = None
        self.x = None
        self.progress = None
        self.kwargs = kwargs
        self.known_features = self.kwargs['known_features']
        self.test_mode = self.kwargs['test_mode']
        self.set_seed(seed)
        self.initialise(10 * self.domain.xdims)
        self.start_data_saving()
        self.setup_logging()
        self.noFitProgress = 0

        # initialise optimizer
        self.optimizer = optimizer(self.acq_fun_eval.evaluate, self.domain)


    def initialise(self, n: int):
        self.remote_server = socket.gethostname()
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
            self.printc(f'Iteration {iteration + 1}', log = True)
        self.printc('Finding diverse x0:', color = 'g', newline=True)
        x0 = self.get_diverse_initial_points(n)
        
        # Run optimisation
        X, F = self.optimizer.run_many(x0)
        bestpoint = X[np.argmax(F)]
        self.printc('Acquisition step:', color = 'g', newline=True)
        self.printc(f'Acquisition value: {np.max(F)}', log = True)
        new_obs = self.evaluate_new_point(bestpoint)
        
        # Update surrogates, archive and storage.
        self.update_storage([new_obs])
        self.createmodels()
        self.acq_fun_eval = self.acq()
        self.optimizer.set_obj(self.acq_fun_eval.evaluate)
        # Update standardisation
        ymean = self.fitness.mean(); ystd = self.fitness.std()
        self.acq_fun.set_fstar0(self.acq_fun,-ymean/ystd.item())


        # Turn behaviour in to a tensor and update archive
        new_obs[2] = torch.tensor(new_obs[2], dtype = torch.double)
        self.QDarchive.updatearchive([new_obs], ymean, ystd)

        # Save the best points for future optimisation runs
        self.save_best_points(X, F)
        
        current_fitness = self.calculate_fitness()
        # Provide terminal feedback
        self.terminal_feedback(new_obs, current_fitness)
        

        #converged = np.max(F) < 1e-2
        if self.progress[-1] == self.progress[-2]:
            self.noFitProgress += 1
        else:
            self.noFitProgress = np.max([0, self.noFitProgress - 1])
        converged_by_fitness = self.noFitProgress > self.domain.fdims

        if converged_by_fitness:
            # Check if the archive has converged
            # If converged, upscale the archive
            self.printc('Converged by fitness', color = 'r', log = True)
            self.noFitProgress = 0
            resolutions = self.resolutions
            res_index = resolutions.index(self.QDarchive.feature_resolution)
            new_res_index = np.min([res_index + 1, len(resolutions) - 1])
            if resolutions[new_res_index] != self.QDarchive.feature_resolution:
                self.upscale(self.QDarchive, resolutions[new_res_index])
        self.save_data()

        self.standardise_stdarchive()

        last_index = self.QDarchive.nichefinder(self.descriptors[-1])
        
        try:
            del self.pointdict[tuple(last_index.numpy())]
        except:
            pass

    def terminal_feedback(self, new_obs, current_fitness):
        '''
        Provides terminal feedback
        '''
        self.printc(f'New point: {new_obs[0].numpy()}', log = True)
        self.printc(f'New fitness: {new_obs[1]}', log = True)
        self.printc(f'New Descriptor: {new_obs[2].numpy()}', log = True)
        true = self.QDarchive.get_region(new_obs[0]).numpy()
        self.printc(f'True region: {true}', log = True)
        self.printc(f'Current fitness: {current_fitness}', log = True)
        self.num_niches = self.QDarchive.get_num_niches()
        self.printc(f'Num filled regions: {self.num_niches}', log = True)

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

    def run(self, n_restarts: int, max_iter: int):
   
        self.restarts = n_restarts
        self.max_iter = max_iter
        for i in range(self.max_iter - self.x.shape[0]):
            self.iterate(n_restarts, i)


    def acq(self):
        return(self.acq_fun( self.fitGP, self.DGPs, self.domain, self.QDarchive))


    def get_diverse_initial_points(self, n: int):

        m = 10 if self.domain.fdims <2 else 1
        n_random_points = np.clip(m*(self.domain.xdims**2)*np.prod(self.domain.feature_resolution),10000,100000)
        random_x = self.domain.get_Xsample(int(n_random_points))  
        init_x = self.load_previous_points(random_x)  # Loads points from previous pointdict
        init_x = self.gen_elite_children(init_x)
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

    def pick_x0(self, initpoints, initvals, n):
        # pick 80% of the best points
        x0dict = {}
        for i in range(int(n*0.8)):
            max_key = max(initvals, key=initvals.get)
            if max_key not in x0dict:
                x0dict[max_key] = initpoints[max_key]
                del initvals[max_key]
        
        ## Pick 20% randomly
        while len(x0dict) < n:
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

    def save_best_points(self, X , F):
        '''
        Saves the best points from the previous optimisation for future
        runs, excluding the point that was previously chosen.
        '''

        X = np.array(X)[np.argsort([f[0] for f in F])][:-1]
        X = np.unique(X, axis = 0)
        X = torch.tensor(X, dtype = torch.double).reshape(-1,self.domain.xdims)
        descriptors = self.domain.evaluate_descriptors(X)
        regions = [torch.stack([self.QDarchive.nichefinder( d )] )[0][0] for d in descriptors]
        for c,region in enumerate(regions):
            region = tuple(region.tolist())
            if region not in self.pointdict.keys():
                self.pointdict[region] = X[c].unsqueeze(0)
            else:
                if X[c] not in self.pointdict[region]:
                    self.pointdict[region] = torch.cat((self.pointdict[region], X[c].unsqueeze(0)))

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
        region_list = []
        if self.DGPs != None: 
            print('estimating regions')
            mem = psutil.virtual_memory().total
            bs = int(mem/((np.prod(self.domain.feature_resolution))/6))
            bs = np.min([100000,bs]) # Batchsize
            index = [i*bs for i in range(int(np.ceil(len(points)/bs))+1)]       
            for c in range(len(index)-1):
                regions = self.V_predict_region(points[index[c]:index[c+1]], fmodels, mymap)
                region_list = region_list + regions
        else:
            points = points[torch.randperm(len(points))]
            descriptors = self.domain.feature_fun(points)
            region_list = [torch.stack([self.QDarchive.nichefinder( d )] )[0] for d in descriptors]
        #region_list = [tuple(r.tolist()) for r in region_list]
        for c,index in enumerate(region_list):
            index = tuple(index[0].numpy())
            if index not in self.pointdict.keys():
                self.pointdict[index] = points[c].unsqueeze(0)
            else:
                if points[c] not in self.pointdict[index]:
                    self.pointdict[index] = torch.cat((self.pointdict[index], points[c].unsqueeze(0)))

                    #self.pointdict[index] = torch.stack((new_point,self.pointdict[index].reshape(-1,  self.domain.xdims  )), dim = 1)
        
        for key in self.pointdict:
            self.pointdict[key] = self.pointdict[key].reshape(-1,  self.domain.xdims  )

        num = sum( len(self.pointdict[index]) for index in self.pointdict.keys() )
        num_filled = len(self.pointdict.keys() )
        self.printc(f'{num} points  in {num_filled} regions, selecting {self.restarts} initial points', log = True)
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
            vals = self.acq_fun_eval.evaluate(x)
            is_above_zero = vals > 0 
            x= x[is_above_zero.squeeze(0)]
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
        

    def start_data_saving(self):
        cwd = os.getcwd()
        domain = self.domain.name
        alg = self.acq_fun_eval.name
        resolution = str(self.resolutions[-1])

        if not self.test_mode:            
            self.save_path = f"{cwd}/experiment_data/{domain}/{alg}/{resolution}/{self.seed}"
            os.makedirs(self.save_path, exist_ok = True)
        else:
            self.save_path = f"{cwd}/experiment_data/{domain}/{alg}/{resolution}/Test/{self.seed}"
            os.makedirs(self.save_path, exist_ok = True) 
        

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
        savefiles = {fitness_file:self.fitness, 
                        descriptors_file: self.descriptors,
                        x_file : self.x}
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

    def load_data(self, save_path, n = None):
        ''' Loads the data from the save path'''
        mydir = save_path
        self.QDarchive.flush()
        fitness_file = f'{mydir}/fitness.pkl'
        descriptors_file = f'{mydir}/descriptors.pkl'
        x_file = f'{mydir}/x.pkl'
        loadfiles = {fitness_file:self.fitness,
                        descriptors_file: self.descriptors,
                        x_file : self.x}
        for file in loadfiles:
            with open(file, 'rb') as f:
                loadfiles[file] = pickle.load(f)
        if n != None:
            self.fitness = loadfiles[fitness_file][:n]
            self.descriptors = loadfiles[descriptors_file][:n]
            self.x = loadfiles[x_file][:n]
        else:
            self.fitness = loadfiles[fitness_file]
            self.descriptors = loadfiles[descriptors_file]
            self.x = loadfiles[x_file] 
        print('Creating models')
        self.createmodels()

        print('initialising archive')
        points = [[self.x[c], self.fitness[c], self.descriptors[c]] for c in range(self.x.shape[0])]
        self.QDarchive.initialise(points, False)

        print('Creating acquisition function')
        self.acq_fun_eval = self.acq()#_fun( self.fitGP, self.DGPs, self.domain, self.QDarchive)
        self.optimizer.set_obj(self.acq_fun_eval.evaluate)
        self.standardise_stdarchive()
        self.calculate_progress()
        #setup standardisation

    def reinitialise(self, n = None):
        print('initialising archive')
        points = [[self.x[c], self.fitness[c], self.descriptors[c]] for c in range(self.x.shape[0])]
        self.QDarchive.flush()
        self.QDarchive.initialise(points, False)

        print('Creating models')
        self.createmodels()

        print('Creating acquisition function')
        self.acq_fun_eval = self.acq()#_fun( self.fitGP, self.DGPs, self.domain, self.QDarchive)
        self.optimizer.set_obj(self.acq_fun_eval.evaluate)
        self.standardise_stdarchive()
        self.calculate_progress()

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

    def upscale(self, archive = None, resolution = None):
        '''
        Upscales the new BOP to a higher resolution archive.
        '''
        if archive == None:
            archive = self.QDarchive
        if resolution == None:
            resolution = self.domain.feature_resolution
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
