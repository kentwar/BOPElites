'''
An algorithm consists of the following ingredients

1. Surrogates
2. An acquisition function
3. An archive
4. A sampling algorithm (for initial point selection)
5. An Optimizer
'''
import numpy as np
import logging , torch, os , pickle , random
import matplotlib.pyplot as plt



class algorithm():

    def __init__(self,  domain, QDarchive, optimizer, seed = None, **kwargs):
        self.domain = domain
        self.QDarchive = QDarchive
        self.initpointdict = {}  # Stores the potential initial points
        self.descriptors = None
        self.fitness = None
        self.x = None
        self.name = 'MAP-Elites'
        self.progress = None
        self.kwargs = kwargs
        self.optimizer = optimizer
        self.test_mode = self.kwargs['test_mode']
        self.optimizer = optimizer( self.domain , self.QDarchive)
        self.set_seed(seed)
        self.initialise(10 * self.domain.xdims)
        self.start_data_saving()
        self.setup_logging()

        # initialise optimizer
        



    def initialise(self, n: int):
        print('Initialising Algorithm')
        init_sample = self.domain.get_sample(n)
        self.update_storage(init_sample)
        print(self.x[0].type())
        print('Creating models')

        ## make in to points list
        print('initialising archive')
        points = [[self.x[c], self.fitness[c], self.descriptors[c]] for c in range(self.x.shape[0])]
        
        self.QDarchive.initialise(points, False)
        self.calculate_progress()

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
        self.progressarchive = self.QDarchive.return_copy()
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
            self.printc(f'number of observations: {self.x.shape[0]}', log = True)
        
        # Run MAP-Elites
        X = self.optimizer.run(n)
        new_obs = self.evaluate_new_points(X)
               
        # Turn behaviour in to a tensor and update archive
        self.QDarchive.updatearchive(new_obs, 0, 0)
        self.update_storage(new_obs)
        # Provide terminal feedback
        self.printc('Observation step:', color = 'g', newline=True)
        current_fitness = self.calculate_fitness()
        self.printc(f'Current fitness: {current_fitness}', log = True)
        num_niches = self.QDarchive.get_num_niches()
        self.printc(f'Num filled regions: {num_niches}', log = True)
        self.save_data()

        #last_index = self.QDarchive.nichefinder(self.descriptors[-1])

    def evaluate_new_point(self, x):
        '''
        Evaluates a new point on the True functions
        '''
        new_fit = self.domain.fitness_fun(np.array(x).reshape(-1, self.domain.xdims))
        new_desc = self.domain.feature_fun(np.array(x).reshape(-1, self.domain.xdims))
        new_obs = [torch.tensor(x, dtype = torch.double) , new_fit, new_desc]
        return(new_obs)

    def evaluate_new_points(self, x):
        '''
        Evaluates a new point on the True functions
        '''
        new_fit = self.domain.fitness_fun(np.array(x).reshape(-1, self.domain.xdims))
        new_desc = self.domain.feature_fun(np.array(x).reshape(-1, self.domain.xdims))
        new_desc = torch.tensor(new_desc, dtype = torch.double) 
        new_obs = [[torch.tensor(x[c], dtype = torch.double) , new_fit[c], new_desc[c]] for c in range(len(x))]
        return(new_obs)

    def update_storage(self, observations):
        x = torch.stack([p[0] for p in observations])
        fit = torch.tensor(np.array([p[1] for p in observations]))
        desc = torch.stack([torch.tensor(p[2]) for p in observations])
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

    def run(self, n_children: int, max_iter: int):
        self.n_children = n_children
        self.max_iter = max_iter
        n_gens = int((self.max_iter - self.x.shape[0])/self.n_children)
        final_run = self.max_iter - n_gens*self.n_children - self.x.shape[0]
        for i in range(n_gens):
            self.iterate(n_children, i)
        self.iterate(final_run, n_gens)


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
        alg = self.optimizer.name
        fdims = str(self.QDarchive.feature_resolution[0])
        if not self.test_mode:            
            self.save_path = f"{cwd}/experiment_data/{domain}/{alg}/{fdims}/{self.seed}"
            os.makedirs(self.save_path, exist_ok = True)
        else:
            self.save_path = f"{cwd}/experiment_data/{domain}/{alg}/{fdims}/Test/{self.seed}"
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
        #self.printc(f'Acquisition function: {self.acq_fun_eval.name}', log = True)
        self.printc(f'Seed: {self.seed}', log = True)

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
        self.standardise_stdarchive()
        self.calculate_progress()
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
        ax.invert_yaxis()
        if save:
            fig.savefig(f'{save_path}/{text}archive.png')
            return(fig, ax)
        else:
            plot.show()
    
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
        ax.plot(x,y, label = self.name)
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