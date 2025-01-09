import numpy as np
from acq_functions.base_acq import BASEacq
import torch
from torch.distributions import Normal

class GPmean(BASEacq):
    
    def __init__(self, fitGP, DGPs, Domain, QDarchive):
        self.fitGP = fitGP
        self.DGPs = DGPs
        self.Domain = Domain
        self.QDarchive = QDarchive
        self.dtype = torch.double
        self.name = 'MEAN'


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
        if type(x) == torch.Tensor:
            x = x.reshape(-1,self.Domain.xdims)
        else:
            x = torch.tensor(x, dtype = self.dtype).reshape(-1,self.Domain.xdims);
        
        return(self.posteriormean(x))
     

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




