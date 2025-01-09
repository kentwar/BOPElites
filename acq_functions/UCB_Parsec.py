import numpy as np
from acq_functions.base_acq import BASEacq
import torch
from torch.distributions import Normal

class GPUCB_Parsec(BASEacq):
    
    def __init__(self, dragGP, liftGP, DGPs, Domain, QDarchive, beta = 4, name = 'SAIL'):
        self.dragGP = dragGP
        self.liftGP = liftGP
        self.DGPs = DGPs
        self.Domain = Domain
        self.QDarchive = QDarchive
        self.dtype = torch.double
        self.name = name
        self.beta = beta
        self.name = 'UCB_Parsec'


    def UCB(self, x,min=False , return_mean = False):
        '''
        This function calculates the Expected Improvement (EGO) acquisition function
        
        INPUT :
        model   : GPmodel   - A GP model from which we will estimate.
        x       : Float     - an x value to evaluate
        fstar   : Float     - Current best value in region
        min     : Boolean   - determines if the function is a minimisation or maximisation problem
        
        OUTPUT :
        ei       : Float     - returns the estimated improvement]
        meanvect : vector    - Vector of mean predictions
        '''
        with torch.no_grad():
            #x = torch.from_numpy(np.array( [ [ x ] ] ).reshape( -1 , self.Domain.xdims ))
            x = x.double()
            
            self.dragGP.eval()
            posterior = self.dragGP.posterior(x)
            mean = posterior.mean
            sigma = posterior.variance.clamp_min(1e-9).sqrt()
            meanvect = mean.expand(mean.shape[0],1 )
            #val = torch.sub(meanvect.t(),fstar)
            #u = torch.div(val.t() ,sigma).t()
            #if min == True:
            #    u = -u
            #normal = Normal(torch.zeros_like(u), torch.ones_like(u))
            #ucdf = normal.cdf(u)
            #updf = torch.exp(normal.log_prob(u))

            UCB = meanvect + (sigma * self.beta)
        if return_mean:
            return( UCB, meanvect )
        else:
            return( UCB )

            
    def evaluate(self, x, beta = 4):
        if type(x) == torch.Tensor:
            x = x.reshape(-1,self.Domain.xdims)
        else:
            x = torch.tensor(x, dtype = self.dtype).reshape(-1,self.Domain.xdims);
        pred_lift = self.liftGP(x).mean
        UCB_val = self.UCB(x)
        fitness = torch.stack([self.Domain.calc_fitness(x[c], UCB_val.detach()[c], pred_lift.detach()[c]) for c in range(x.shape[0])]).squeeze(-1)
        return(fitness)
     

    def sp(self, x):
        '''
        shape point, get points in the right shape to work
        '''
        return(np.array(x).reshape(-1,self.Domain.xdims))

    def init_x(self):
        pass

    def getmu(self, x):
        '''
        Returns the posterior mean and variance of x
        '''
        mus = []
        vars = []
        for model in self.DGPs:
            mu, var = self.queryDGP(model, x)
            mus.append(mu)
        return(mus)

    def queryDGP(self, model, x):
        '''
        Returns the posterior mean descriptors and variance of x
        '''
        model.eval()
        posterior = model.posterior(x)
        mean = posterior.mean
        sigma = posterior.variance.clamp_min(1e-9)
        return(mean.double(), sigma.double())

    def getmuvar(self, x):
        '''
        Returns the posterior mean and variance of x
        '''
        mus = []
        vars = []
        for model in self.DGPs:
            mu, var = self.queryDGP(model, x)
            mus.append(mu)
            vars.append(var)
        return(mus, vars)

    def get_univariate_descriptor_probs(self, mus_, var_, regions):
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
    
    def predict_region(self, x):
        '''
        Vectorized function that returns the index of the niches 
        with maximal probability that x belongs to that niche 
        '''
        #torchz = torch.from_numpy(x).double()
        xdims = self.Domain.xdims
        fdims = self.Domain.fdims
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
