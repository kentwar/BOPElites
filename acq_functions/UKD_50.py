import numpy as np
from acq_functions.base_acq import BASEacq
import torch
from torch.distributions import Normal
import itertools as it
import scipy.stats


class UKD_50(BASEacq):
    '''
    This is the BOP acquisition function that predicts the 
    region membership from the Descriptor GPs (DGPs)
    '''
    def __init__(self, fitGP, DGPs, Domain, QDarchive):        
        self.fitGP = fitGP
        self.DGPs = DGPs
        self.Domain = Domain
        self.QDarchive = QDarchive
        self.dtype = torch.double
        self.name = 'UKD_50'

    def posteriormean(self, x,min=False , return_mean = False):
        '''
        This function calculates the posterior mean
        
        INPUT :
        model   : GPmodel   - A GP model from which we will estimate.
        x       : Float     - an x value to evaluate
        
        OUTPUT :
        meanvect : vector    - Vector of mean predictions
        '''
        with torch.no_grad():
            x = x.double()
            
            self.fitGP.eval()
            posterior = self.fitGP.posterior(x)
            mean = posterior.mean
            meanvect = mean.expand(mean.shape[0],1 )

        return(meanvect)
           
    def evaluate(self, x, fstar = None):
        # if len(x.shape) == 1:
        #     x = x.double()
        #     post_mean_prob = self.region_probabilities2(x)
        #     #transformed_prob = self.transform(post_mean_prob)
        #     fitness = self.posteriormean(x.reshape(-1,self.Domain.xdims))
        #     std = self.fitGP.train_targets.std().item()
        #     mean = self.fitGP.train_targets.mean().item()
        #     adjust = torch.zeros(1)
        #     adjust[post_mean_prob < 0.5] = - (0 - mean)/std
        #     # un standardise based on fitness in the fitGP
        #     #std = self.fitGP.train_targets.std().item()
        #     #mean = self.fitGP.train_targets.mean().item()
        #     #fitness = (fitness*std) + mean
        #     #fitness = fitness * post_mean_prob
        #     #fitness = fitness * transformed_prob
        #     #fitness = (fitness - mean) / std
        #     #adjusted_fitness = self.adjust_fitness(fitness, transformed_prob)
        #     #adjusted_fitness = self.adjust_fitness(fitness, post_mean_prob)
        #     fitness = fitness - adjust
        #     return(fitness)
        #else:
        return(self.evaluate2(x, fstar = fstar))

    def evaluate2(self, x, fstar = None):
        x = x.double()
        post_mean_prob = self.region_probabilities2(x).squeeze(-1)
        
        fitness = self.posteriormean(x.reshape(-1,self.Domain.xdims))
        # Less than 50 tails off, everything else is 0.5
        # adjust = torch.ones(post_mean_prob.shape, dtype= torch.double) 
        # adjust[post_mean_prob < 0.5] = post_mean_prob[post_mean_prob < 0.5] / 0.5
        # un standardise based on fitness in the fitGP

        #fitness = fitness * post_mean_prob
        #fitness = fitness * transformed_prob
        #fitness = (fitness - mean) / std
        #adjusted_fitness = self.adjust_fitness(fitness, adjust)
        adjusted_fitness = self.adjust_fitness(fitness, post_mean_prob)
        #adjusted_fitness = self.adjust_fitness(fitness, transformed_prob)
        return(adjusted_fitness)

    def transform(self, probs):
        '''
        This function transforms the posterior probability into a relaxed sigmoid
        '''
        beta_dist = scipy.stats.beta(7,7)
        betas = beta_dist.cdf(probs)
        return(betas)

    def adjust_fitness(self, fitness, transformed_prob):
        '''
        adjusts fitness based on sigmoid, assuming the fitness is 
        standardised.
        '''
        meany = self.fitGP.train_targets.mean().item()
        stdy = self.fitGP.train_targets.std().item()
        stdmaxy = (self.fitGP.train_targets.max().item() - meany)/stdy
        stdminy = (self.fitGP.train_targets.min().item() - meany)/stdy
        std_range = ((stdmaxy - stdminy) - self.fitGP.train_targets.mean().item())/self.fitGP.train_targets.std().item()
        n_probs = transformed_prob.shape[0]
        prob_infeasible = torch.ones(n_probs) - transformed_prob 
        adjustment = prob_infeasible * std_range
        adjusted_fitness = torch.sub(fitness , adjustment.unsqueeze(-1))
        return(adjusted_fitness)

    def sp(self, x):
        '''
        shape point, get points in the right shape to work
        '''
        return(np.array(x).reshape(-1,self.Domain.xdims))

    def predict_region(self, x):
        '''
        Vectorized function that returns the index of the niches 
        with maximal probability that x belongs to that niche 
        '''
        #torchz = torch.from_numpy(x).double()
        xdims = self.Domain.xdims
        mulist = [model(x.reshape(-1,xdims)).mean.detach().numpy() for model in self.DGPs]       
        mulist = np.array(mulist).T
        mulist = self.conformbeh(mulist)
        niches = [self.QDarchive.nichefinder(mu) for mu in mulist]
        return(torch.stack(niches))

    def conformbeh(self,  beh ):
        '''
        conforms behaviour to the domain bounds
        '''
        lb = [bound for bound in self.Domain.featmins ]
        ub = [bound for bound in self.Domain.featmaxs ]
                
        beh = np.clip(beh , a_min = lb, a_max = ub)
        
        return( list(beh) ) 

    def region_probabilities( self, x ):
        '''
        returns the probability that x belongs to the region that is maximally
        likely to be the region that x belongs to 

        '''
        predicted_region = self.predict_region(x).squeeze(1)
        fdims = self.Domain.fdims
        # Get the posterior mean and variance of x for each descriptor model
        probslist = []
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        #Get mean and variance
        _mus, _vars = self.getmuvar(x)       
        featdimslist = self.get_univariate_descriptor_probs(_mus, _vars, predicted_region)   
        ## Create a univariate gaussian for each descriptor dimension       
        if fdims == 1:
            nprobs = np.array([featdimslist[i] for i in enumerate(predicted_region)])
        else:      
            nprobs = np.product([featdimslist[0][i] for i in range(len(featdimslist[0]))], axis = 1)[0]    
            #nprobs = np.array(np.multiply(*[featdimslist[c][int(i.numpy())] for c,i in enumerate(predicted_region[0])]))
        probslist.append(nprobs)
        probs_tensor = torch.tensor(nprobs)
        return(probs_tensor)       

    def region_probabilities2(self, x):
        '''
        returns the probability that x belongs to the region that is maximally
        likely to be the region that x belongs to
        '''
        predicted_region = self.predict_region(x)
        fdims = self.Domain.fdims
        # Get the posterior mean and variance of x for each descriptor model
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Get mean and variance
        _mus, _vars = self.getmuvar(x) 
        
        # Calculate the probabilities of being in the feasible region
        probs_list = []
        for i in range(_mus.shape[1]):
            prob = 1
            for j in range(fdims):
                # Obtain the boundaries of the feasible region for the current dimension
                lower_bound, upper_bound = self.QDarchive.edges[j][predicted_region[i,0,j]], self.QDarchive.edges[j][predicted_region[i,0,j] + 1]

                # Calculate the CDF values for the lower and upper bounds
                cdf_lower = scipy.stats.norm.cdf(lower_bound, loc=_mus[j, i], scale=np.sqrt(_vars[j, i]))
                cdf_upper = scipy.stats.norm.cdf(upper_bound, loc=_mus[j, i], scale=np.sqrt(_vars[j, i]))

                # Calculate the probability of being within the feasible region for the current dimension
                prob *= (cdf_upper - cdf_lower)
            
            # Append the calculated probability for the current point x
            probs_list.append(prob)
        
        probs_tensor = torch.tensor(probs_list)
        return probs_tensor

    def calculate_cutoff(self):
        d = self.Domain.xdims
        t = self.fitGP.train_targets.shape[0]
        n = np.prod(self.Domain.feature_resolution)
        mis = self.QDarchive.mispredicted
        npf = self.QDarchive.nopointsfound
        cutoff = 0.5*(2/n)**((10*d)/(t + mis - 2*npf))**0.5
        return(cutoff)

    def find_neighbouring_regions( self, known_region):
        '''
        returns the indices of the neighbouring regions
        '''
        regions = []
        known_region = tuple(known_region.numpy())
        for c , dim in enumerate(known_region):
            if dim-1 >=0 and dim+1 < self.Domain.feature_resolution[c]:
                regions.append([dim-1, dim, dim+1])
            else:
                if dim-1 <0:
                    regions.append([ dim, dim+1])
                else:
                    regions.append([dim-1,  dim])
        return(list(it.product(*regions)))

    def queryDGP(self, model, x):
        model.eval()
        posterior = model.posterior(x)
        mean = posterior.mean
        sigma = posterior.variance.clamp_min(1e-9)
        return(mean.double(), sigma.double())

    def getmuvar(self, x):
        '''
        Returns the posterior mean and variance of x
        '''
        mus_ = []
        vars_ = []
        for model in self.DGPs:
            mu, var = self.queryDGP(model, x)
            mus_.append(mu)
            vars_.append(var)
        mus_ = torch.stack(mus_)
        vars_ = torch.stack(vars_)
        return(mus_, vars_)

    def get_univariate_descriptor_probs2(self, mus_, var_, regions):
        '''
        returns region probabilities from the univariate descriptor distributions
        for each nieghbouring region
        '''
        edges = self.QDarchive.edges
        fdims = self.Domain.fdims
        featdimslist = []
        regions = np.array(regions).T
        for n_desc in range(fdims):
            mu = mus_[n_desc]
            var = var_[n_desc]
            distro = scipy.stats.norm( loc = mu , scale = np.sqrt( var ) )        
            cdfvals = [distro.cdf( edge ) for edge in edges[n_desc]]          
            featdimprobs = {i: ( cdfvals[i+1]-cdfvals[i] ) for i in regions[n_desc] } 
            featdimslist.append(featdimprobs)
        return(featdimslist)

    def get_univariate_descriptor_probs(self, mus_, var_, regions):
        '''
        returns region probabilities from the univariate descriptor distributions
        for each neighbouring region
        '''
        edges = self.QDarchive.edges
        fdims = self.Domain.fdims
        featdimslist = []
        #regions = np.array(regions).T

        probvals = []       
        regions = regions
        for i in range(regions.shape[1]):
            distro = scipy.stats.norm( loc = mus_[i] , scale = np.sqrt( var_[i] ) ) 
            l_edges = regions[:,i]
            u_edges = regions[:,i] + 1
            l_cdfvals = [distro.cdf( edges[i][e] ) for e in l_edges]          
            u_cdfvals = [distro.cdf( edges[i][e] ) for e in u_edges]
            probvals.append(np.array(u_cdfvals) - np.array(l_cdfvals))
            
        return(probvals)
    
    def get_univariate_descriptor_probs2(self, mus_, var_, regions):
        '''
        returns region probabilities from the univariate descriptor distributions
        for each nieghbouring region
        '''
        edges = self.QDarchive.edges
        fdims = self.Domain.fdims
        featdimslist = []
        #regions = np.array(regions).T

        probvals = []       
        regions = regions.squeeze(0)
        regions = regions.squeeze(1)
        for i in range(regions.shape[1]):
            distro = scipy.stats.norm( loc = mus_[i] , scale = np.sqrt( var_[i] ) ) 
            l_edges = regions[:,i]
            u_edges = regions[:,i] + 1
            l_cdfvals = [distro.cdf( edges[i][e] ) for e in l_edges]          
            u_cdfvals = [distro.cdf( edges[i][e] ) for e in u_edges]
            probvals.append(np.array(u_cdfvals) - np.array(l_cdfvals))
            
        return(probvals)