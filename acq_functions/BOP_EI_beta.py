import numpy as np
from acq_functions.base_acq import BASEacq
import torch
from torch.distributions import Normal
import itertools as it

class BOP_EI_beta(BASEacq):
    
    def __init__(self, fitGP, DGPs, Domain, QDarchive):
        self.fitGP = fitGP
        self.DGPs = DGPs
        self.Domain = Domain
        self.QDarchive = QDarchive
        self.dtype = torch.double
        self.name = 'BOP_EI_KF_beta'


    def EI(self, x,fstar, beta, min=False , return_mean = False , ):
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
            
            self.fitGP.eval()
            posterior = self.fitGP.posterior(x)
            mean = posterior.mean
            sigma = posterior.variance.clamp_min(1e-9).sqrt()
            meanvect = mean.expand(mean.shape[0],1 )
            val = torch.sub(meanvect.t(),fstar)
            u = torch.div(val.t() ,sigma).t()
            if min == True:
                u = -u
            normal = Normal(torch.zeros_like(u), torch.ones_like(u))
            ucdf = normal.cdf(u)
            updf = torch.exp(normal.log_prob(u))
            ei = (sigma * ( beta *updf + u * ucdf).t()).t()
        if return_mean:
            return( ei, meanvect )
        else:
            return( ei )
            
    def evaluate(self, x, beta = 1, fstar = None ):
        if type(x) == torch.Tensor:
            x = x.reshape(-1,self.Domain.xdims)
        else:
            x = torch.tensor(x, dtype = self.dtype).reshape(-1,self.Domain.xdims);
        
        if x.shape[0] == 1:
            val = self.evaluate_single(x, fstar, beta)
        else:
            val = self.vectorised_evaluate(x, fstar, beta)
        return(val) 

    def vectorised_evaluate(self, x, fstar, beta):
        fstar = self.findfstar(x)
        ei = self.EI(x, fstar, beta)
        return(ei)

    def evaluate_single(self, x , fstar, beta):
        fstar = self.findfstar(x)
        ei = self.EI(x, fstar, beta)
        return(ei)        

    def findfstar(self, x):
        # identify region
        if x.shape[0] > 1:
            x = x[0,:]
        descriptors = torch.tensor(self.Domain.feature_fun(self.sp(x)))
        region_index = self.QDarchive.nichefinder(descriptors)
        index = tuple(region_index.numpy())
        # get fstar from archive
        fstar = self.QDarchive.stdfitness[index]
        if torch.isnan(fstar) or fstar == None:
            fstar = self.stdfstar0
        fstar = torch.tensor([fstar], dtype = self.dtype)
        
        return(fstar)

    def set_fstar0(self, fstar0):
        self.stdfstar0 = fstar0

    def sp(self, x):
        '''
        shape point, get points in the right shape to work
        '''
        return(np.array(x).reshape(-1,self.Domain.xdims))

    def init_x(self):
        pass




