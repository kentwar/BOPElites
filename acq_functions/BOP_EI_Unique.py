import numpy as np
from acq_functions.base_acq import BASEacq
import torch
from torch.distributions import Normal
import itertools as it

class BOP_EI_KF(BASEacq):
    
    def __init__(self, fitGP, DGPs, Domain, QDarchive):
        self.fitGP = fitGP
        self.DGPs = DGPs
        self.Domain = Domain
        self.QDarchive = QDarchive
        self.dtype = torch.double
        self.name = 'BOP_EI_KF_Unique'


    def EI(self, x,fstar,min=False , return_mean = False):
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

            ei = (sigma * (updf + u * ucdf).t()).t()
        if return_mean:
            return( ei, meanvect )
        else:
            return( ei )
            
    def evaluate(self, x, fstar = None):
        if type(x) == torch.Tensor:
            x = x.reshape(-1,self.Domain.xdims)
        else:
            x = torch.tensor(x, dtype = self.dtype).reshape(-1,self.Domain.xdims);
        
        if x.shape[0] == 1:
            val = self.evaluate_single(x, fstar)
        else:
            val = self.vectorised_evaluate(x, fstar)
        if self.unexplored(x):
            val = val * 0.5
        return(val)

    def unexplored(self, x):
        if x.shape[0] > 1:
            x = x[0]
        descriptors = torch.tensor(self.Domain.feature_fun(self.sp(x)))
        region_index = self.QDarchive.nichefinder(descriptors)
        index = tuple(region_index.numpy())
        radius = self.expand(index, self.Domain.feature_resolution, rad = 1)
        neighbours = torch.tensor([self.QDarchive.fitness[index] for index in radius])
        if torch.isnan(neighbours).all():
            return(False)
        else:
            return(True)

    def expand(self, index , fr,  rad =1):
        '''
        Takes a point an index and finds the outer wall of the hypercube 
        surrounding the index with distance rad

        fr = feature_resolution
        '''
        l_bounds = [np.max([0,i-rad]) for i in index]
        u_bounds = [np.min([i+rad, fr[c]-1]) for c,i in enumerate(index)]
        radii = [range(l_bounds[i],u_bounds[i]+1) for i in range(len(l_bounds))]
        radius_index = []
        vertices = [[r[0],r[-1]] for r in radii]
        for c,i in enumerate(vertices):
            temp = [r for r in radii]
            temp[c] = i   
            rad =  list(it.product(*temp))
            radius_index += list(rad)
        
        ### Get rid of duplicates
        radius_index.append(tuple(index))
        radius_index = list(set(radius_index))

        return(radius_index)  

    def vectorised_evaluate(self, x, fstar):
        fstar = self.findfstar(x)
        ei = self.EI(x, fstar)
        return(ei)

    def evaluate_single(self, x , fstar):
        fstar = self.findfstar(x)
        ei = self.EI(x, fstar)
        return(ei)        

    def findfstar(self, x):
        # identify region
        if x.shape[0] > 1:
            x = x[0,:]
        descriptors = torch.tensor(self.Domain.feature_fun(self.sp(x)))
        region_index = self.QDarchive.nichefinder(descriptors)
        if region_index.shape[0] == 1:
            region_index = region_index[0]
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




