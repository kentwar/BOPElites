import numpy as np
from acq_functions.base_acq import BASEacq
import torch
from torch.distributions import Normal
import itertools as it
import scipy.stats
import copy

# This is a specific implementation of the BOP_UKD acquisition function
# For using with the Parsec domain.

class BOP_EI_Parsec(BASEacq):
    '''
    This is the BOP acquisition function that predicts the 
    region membership from the Descriptor GPs (DGPs)
    '''
    def __init__(self, dragGP, liftGP, DGPs, nanmodel, Domain, QDarchive):        
        self.dragGP = dragGP
        self.liftGP = liftGP
        self.DGPs = DGPs
        self.nanmodel = nanmodel
        self.Domain = Domain
        self.QDarchive = QDarchive
        self.dtype = torch.double
        self.name = 'BOP_UKD_PARSEC'
        self.num_regions = np.product(self.QDarchive.feature_resolution)

    def EI(self, x, fstar, beta, min=False , return_mean = False):
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
        #with torch.no_grad():
            #x = torch.from_numpy(np.array( [ [ x ] ] ).reshape( -1 , self.Domain.xdims ))
        x = x.double()
        
        self.dragGP.eval()
        self.liftGP.eval()
        posterior = self.dragGP.posterior(x)
        liftmean = self.liftGP.posterior(x).mean
        mean = posterior.mean
        sigma = posterior.variance.clamp_min(1e-9).sqrt()
        if x.shape[0] == self.Domain.xdims:
            x = x.unsqueeze(0) 
        predicted_fitness = torch.stack([self.Domain.calc_fitness(x[c],mean[c], liftmean[c]) for c in range(len(x))])
        val = torch.sub(predicted_fitness,fstar)
        u = torch.div(val ,sigma)
        if min == True:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))

        ei = (sigma * ( updf + u * ucdf)) #* (1 + beta *  desc_var.t()) 
        if return_mean:
            return( ei, meanvect )
        else:
            return( ei )
            
    def vec_EI(self, x,  fstar_tensor, beta, min=False , return_mean = False):
        '''
        This function calculates the Expected Improvement (EGO) acquisition function
        in a vectorised format

        INPUT :
        model   : GPmodel   - A GP model from which we will estimate.
        x       : Float     - an x value to evaluate
        fstar   : Float     - Current best value in region
        min     : Boolean   - determines if the function is a minimisation or maximisation problem
        
        OUTPUT :
        ei       : Float     - returns the estimated improvement]
        meanvect : vector    - Vector of mean predictions
        '''
        x = x.double()
        
        self.dragGP.eval()
        self.liftGP.eval()
        posterior = self.dragGP.posterior(x)
        liftmean = self.liftGP.posterior(x).mean
        mean = posterior.mean
        sig = posterior.variance.clamp_min(1e-9).sqrt()
        predicted_fitness = self.Domain.calc_fitness(x,mean, liftmean)
        # mean = posterior.mean
        # sig = posterior.variance.clamp_min(1e-9).sqrt()
        val = torch.sub(predicted_fitness,fstar_tensor)
        u = torch.div(val ,sig)
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))

        ei = (sig * ( updf + u * ucdf)) * (1 + beta *  sig.t())
        if return_mean:
            return( ei, meanvect )
        else:
            return( ei )


    # def grad_vectorised_evaluate(self, x, beta):
    #     # Find cluster of regions for each x
    #     #neighbouring_regions = self.grad_get_regions(x)
    #     neighbouring_regions = self.grad_get_ALL_regions(x)

    #     # Get probability of being in each region, for each x
    #     probs = self.grad_region_probabilities(x, neighbouring_regions)
    #     print('check')
    #     fstars = self.QDarchive.stdfitness.reshape(-1).repeat(x.shape[0],1)
        
    #     # Turn fstars into a long tensor

    #     fstar_tensor = fstars.reshape(-1,1).double()
    #     fstar_is_nan = torch.isnan(fstar_tensor)
    #     fstar_tensor[fstar_is_nan] = self.stdfstar0

    #     # Calculate EI and times by probability

    #     fitness = self.vec_EI( x, fstar_tensor, beta = beta)
    #     prob_tens = torch.tensor(probs, dtype = torch.double).reshape_as(fitness)
       
    #     product = prob_tens * fitness
    #     product_reshaped = product.reshape(x.shape[0], -1)

    #     #value = product_reshaped.sum(1).double() * (1 + beta * entropy / self.entropynorm)
    #     value = product_reshaped.sum(1).double() * (1 + entropy / self.entropynorm)
    #     return(value)

    def grad_vectorised_evaluate(self, x, beta):
        # Find cluster of regions for each x
        #region = self.Domain.feature_function(x)
        region = self.QDarchive.get_region(x)
        fstar = self.QDarchive.fitness[region].unsqueeze(0)
        
        # Turn fstars into a long tensor

        fstar_is_nan = torch.isnan(fstar)
        fstar[fstar_is_nan] = torch.tensor(0) #self.stdfstar0

        # Calculate EI and times by probability

        drag_EI = self.vec_EI( x, fstar, beta = beta)
       
        pred_lift = self.liftGP(x).mean
        value = self.Domain.calc_fitness(x, drag_EI, pred_lift)
        print('grad')
        return(value)

    # def grad_get_regions(self, x):
    #     # Find cluster of regions
    #     central_regions = self.predict_region(x)
    #     # Find probabilities of each region
    #     fdims = self.Domain.fdims
    #     # List all neighbouring regions
    #     neighbouring_regions = [self.find_neighbouring_regions(region) for region in central_regions]
    #     ## Check that each region has the same number of neighbours
    #     pad = 3**fdims
    #     for c,neighbours in enumerate(neighbouring_regions):
    #         if len(neighbours) < pad:
    #             neighbouring_regions[c] = neighbouring_regions[c] + [tuple([-1]*fdims)] * (pad - len(neighbours))
    #     neighbouring_regions = torch.tensor(neighbouring_regions, dtype = torch.int64)  
    #     return(neighbouring_regions)

    # def grad_get_ALL_regions(self, x):
    #     all_regions = torch.tensor(list(it.product(*[range(featdim) for featdim in self.Domain.feature_resolution])))
    #     neighbouring_regions = [all_regions for point in x]
    #     neighbouring_regions = torch.stack(neighbouring_regions)
    #     return(neighbouring_regions)

    # def grad_region_probabilities(self, x, neighbouring_regions):
    #             # Get the posterior mean and variance of x for each descriptor model
    #     _mus, _vars = self.getmuvar(x)
    #     _mus = torch.stack(_mus)  # Now they are 
    #     _vars = torch.stack(_vars)

    #     edges = self.QDarchive.edges
    #     fdims = self.Domain.fdims

    #     neighbouring_regions = np.array(neighbouring_regions)

    #     # Allow for single evaluations
    #     if len(neighbouring_regions.shape) == 2:
    #         neighbouring_regions = neighbouring_regions.reshape(1,neighbouring_regions.shape[0],neighbouring_regions.shape[1])
    #     probs_array = np.zeros(neighbouring_regions.shape)

    #     ## we make probs an array of n x fdims where n is the number of points
    #     ## and n is the number of neighbouring regions
    #     for n_desc in range(fdims):
    #         mu = _mus[n_desc]
    #         var = _vars[n_desc]
    #         distro = scipy.stats.norm( loc = mu.detach() , scale = np.sqrt( var.detach() ) )  
    #         cdfvals = distro.cdf(edges[n_desc])
    #         featdimprobs = np.diff(cdfvals, axis=-1)
    #         for c , region_list in enumerate(neighbouring_regions):
    #             indexes = region_list.T[n_desc]
    #             probs = featdimprobs[c][indexes]
    #             probs_array[c][:,n_desc] = probs
    #     # Then multiply by probability in each dimension to get probability of 
    #     # being in each region
    #     probs = np.prod(probs_array, axis = 2)
    #     return(probs)

    def evaluate(self, x, beta = 0, fstar = None):
        
        if type(x) == torch.Tensor:
            x = x.reshape(-1,self.Domain.xdims)
        else:
            x = torch.tensor(x, dtype = self.dtype).reshape(-1,self.Domain.xdims);
        
        if x.shape[0] == 1:
            # check if a duplicate
            return(self.evaluate_single(x,  beta))
        else:
            ## As multiple points are only ever evaluated in the point finding
            ## process, we assume duplicates are unimportant.
            return(self.grad_vectorised_evaluate(x, beta))

    def evaluate_init(self, x, beta = 1, fstar = None):
        
        if type(x) == torch.Tensor:
            x = x.reshape(-1,self.Domain.xdims)
        else:
            x = torch.tensor(x, dtype = self.dtype).reshape(-1,self.Domain.xdims);
        
        if x.shape[0] == 1:
            # check if a duplicate
            return(self.evaluate_single(x,  beta, fstar))
        else:
            ## As multiple points are only ever evaluated in the point finding
            ## process, we assume duplicates are unimportant.
            return(self.vectorised_evaluate(x, beta))

    def vectorised_evaluate(self, x, beta):
        # Find cluster of regions
        descriptors = self.Domain.feature_fun(x.numpy())
        regions = self.QDarchive.nichefinder(descriptors)

        # Convert to tensor
        fstar_list = [self.QDarchive.fitness[tuple(region)] for region in regions]
        
        fstar_tensor = torch.stack(fstar_list).reshape(-1,1).double()
        fstar_is_nan = torch.isnan(fstar_tensor)
        fstar_tensor[fstar_is_nan] = torch.tensor([0]) #self.stdfstar0
        # Calculate EI and times by probability
        drag_EI = self.EI( x, fstar_tensor ,beta)
        pred_lift = self.liftGP(x).mean
        value = [self.Domain.calc_fitness(x[c], drag_EI[c], pred_lift[c]).item() for c in range(x.shape[0])]
        p = x.numpy().reshape(-1, self.Domain.xdims)
        prob_feas = self.nanmodel.predict_proba(p)[:,0]
        value = value * prob_feas
        return(value)  

    # def vectorised_evaluate(self, x, beta):
    #     # Find cluster of regions
    #     descriptors = self.Domain.feature_functions(x)
    #     regions = self.QDarchive.nichefinder(descriptors)
        
    #     central_region = self.predict_region(x)[0]
    #     region_probs = self.region_probabilities(x, central_region)

    #     # Find fstars for each region        
    #     fstar_dict = self.find_all_fstar(x, region_probs)

    #     # Convert to tensor
    #     fstar_list = [fstar_dict[region] for region in region_probs]
    #     fstar_tensor = torch.stack(fstar_list).reshape(-1,1)

    #     # Calculate EI and times by probability
    #     fitness = self.EI( x, fstar_tensor ,beta)
    #     probs = [p for p in region_probs.values()]
    #     prob_tens = torch.tensor(probs, dtype = torch.double).squeeze()
    #     product = prob_tens * fitness
    #     value = torch.sum(product, axis = 0)
    #     return(value) 

    # def evaluate_single(self, x , beta, fstar = None):
    #     # Find cluster of regions
    #     central_region = self.predict_region(x)[0]
    #     limited = False
    #     if limited:
    #         regions = self.find_neighbouring_regions(central_region)
    #         regions = torch.tensor(regions).unsqueeze(0)
    #     else:
    #         regions = self.grad_get_ALL_regions(central_region.unsqueeze(0))
    #     region_probs = self.grad_region_probabilities(x, regions)
    #     #region_probs = self.region_probabilities(x, central_region)

    #     # Find fstars for each region        
    #     #fstar_dict = self.find_all_fstar(x, region_probs)
    #     fstar_tensor = torch.tensor([self.QDarchive.stdfitness[tuple(region)].item() for region in regions[0]],dtype = torch.double)
    #     no_fstar = torch.isnan(fstar_tensor) 
    #     fstar_tensor[no_fstar] = self.stdfstar0.double()
    #     probs = region_probs#[p for p in region_probs.values()]
    #     prob_tens = torch.tensor(probs, dtype = torch.double)
    #     if torch.sum(prob_tens) == 0:
    #         return(torch.tensor([0], dtype = torch.double))
    #     probiszero = prob_tens == 0
    #     fstar_tensor = fstar_tensor[~probiszero[0]]
        
    #     if limited:
    #         prob_tens = prob_tens[~probiszero] / torch.sum(prob_tens[~probiszero])
    #     else:
    #         prob_tens = prob_tens[~probiszero] #/ torch.sum(prob_tens[~probiszero])
    #     # Calculate EI and times by probability
    #     fitness = self.vec_EI( x, fstar_tensor , beta)
    #     entropy = -torch.sum(prob_tens * torch.log(prob_tens), axis = 0)
    #     value = torch.sum(prob_tens.T * fitness[0,:].T) * (1 + beta * (entropy / - self.entropynorm))    
    #     return(value.unsqueeze(0))   

    def evaluate_single(self, x , beta, fstar = None):
        # Find cluster of regions for each x
        region = self.QDarchive.get_region(x)
        fstar = self.QDarchive.fitness[tuple(region.numpy()[0])].unsqueeze(0).double()
        
        # Turn fstars into a long tensor

        fstar_is_nan = torch.isnan(fstar)
        fstar[fstar_is_nan] = torch.tensor([0])

        # Calculate EI and times by probability

        drag_EI = self.vec_EI( x, fstar, beta = beta)
       
        pred_lift = self.liftGP(x).mean
        value = self.Domain.calc_fitness(x, drag_EI, pred_lift)
        p = x.numpy().reshape(-1, self.Domain.xdims)
        prob_feas = self.nanmodel.predict_proba(p)[:,0]
        value = value * prob_feas
        return(value)

    # def check_for_misprediction(self, x ):
    #     # Find cluster of regions
    #     print(x.shape)
    #     assert (x.shape[0] == 1 and x.shape[1] == self.Domain.xdims) or x.shape[0] == self.Domain.xdims, "x must be a single point"
    #     x = x.unsqueeze(0)
    #     central_region = self.predict_region(x)[0]
    #     region_probs = self.region_probabilities(x, central_region)

    #     # Find fstars for each region        
    #     fstar_dict = self.find_all_fstar(x, region_probs)

    #     # Calculate EI and times by probability
    #     values_dict = {}

    #     for c,region in enumerate(region_probs):            
    #         values_dict[region] =  self.EI( x, fstar_dict[region], 1 ) * region_probs[region]
                 
    #     total_value = sum(values_dict.values())
        
    #     for c,region in enumerate(region_probs):            
    #         values_dict[region] =  values_dict[region] / total_value

    #     ## Provide feedback on the best region
    #     best_region = max(values_dict, key=values_dict.get)
    #     best_prob = region_probs[best_region]
    #     bestval = values_dict[best_region].numpy()[0][0]*total_value.item()
    #     valpercent = values_dict[best_region].numpy()[0][0]*100
    #     toral_val = sum(values_dict.values())
    #     print(f'Max probability {best_prob} in {best_region}, best value {bestval}, {valpercent}% of total')
    #     if region_probs[best_region] < 0.5 and values_dict[best_region] > 0.5:
    #         return(True, total_value)
    #     else:
    #         return(False, total_value)
     

    # def find_all_fstar(self, x, region_probs = None):
    #     '''
    #     Finds the fstar for some index/s in the archive
    #     '''
    #     fstar_dict = {}
    #     for region in region_probs:
    #         # find existing elite in archive
    #         fstar = self.QDarchive.stdfitness[region]
    #         # If no elite exists, use the default fstar
    #         if torch.isnan(fstar) or fstar == None:
    #             fstar = self.stdfstar0
    #         fstar_dict[region] = torch.tensor([fstar], dtype = self.dtype)

    #     return(fstar_dict)

    def set_fstar0(self, fstar0):
        self.stdfstar0 = fstar0

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
        xdims = self.Domain.xdims
        fdims = self.Domain.fdims
        
        if fdims == 1:
            mus = self.DGPs[0](x.reshape(-1, xdims)).mean.detach().numpy()
            mus = mus.reshape(-1, fdims)
        else:
            mus = [model(x.reshape(-1, xdims)).mean.detach().numpy() for model in self.DGPs]
            mus = np.array(mus)                 
        mus = self.conformbeh(mus)
        niches = [self.QDarchive.nichefinder(mu) for mu in mus]
        xdim = x.shape[0] if len(x.shape) > 1 else 1
        return torch.stack(niches).reshape(xdim, fdims)

    def conformbeh(self,  beh ):
        '''
        conforms behaviour to the domain bounds
        '''
        lb = [bound for bound in self.Domain.featmins ]
        ub = [bound for bound in self.Domain.featmaxs ]

        beh = beh.T       
        beh = np.clip(beh , a_min = lb, a_max = ub)
        
        return( list(beh) ) 

    def region_probabilities( self, x , known_region):
        '''
        returns the probability that x belongs to each region around the 
        known_region as a dictionary
        '''
        regions = self.find_neighbouring_regions(known_region)
        fdims = self.Domain.fdims
        # Get the posterior mean and variance of x for each descriptor model
        _mus, _vars = self.getmuvar(x)
        ## Create a univariate gaussian for each descriptor dimension
        featdimslist = self.get_univariate_descriptor_probs(_mus, _vars, regions)
        if fdims == 1:
            nprobs = np.array([[featdimslist[c][i] 
                    for c,i in enumerate(region)]
                    for region in regions] )
        else:           
            nprobs = np.array([np.multiply(*[featdimslist[c][i] 
                        for c,i in enumerate(region)]) 
                        for region in regions] 
            )

        # normalise the probabilities
        if self.Domain.fdims == 1:
            probsum = np.sum(nprobs, axis = 0)[0][0]
        else:
            probsum = np.sum(nprobs,axis = 0)[0]
        if probsum !=0:
            nprobs = nprobs / probsum
        
        # Implement Cutoff value
        cutoff = self.calculate_cutoff()
        nprobs[nprobs < cutoff] = 0
        probs_dict = {region: nprobs[i] for i, region in enumerate(regions)}
        return(probs_dict)        

    def calculate_cutoff(self):
        d = self.Domain.xdims
        t = self.fitGP.train_targets.shape[0]
        n = np.prod(self.Domain.feature_resolution)
        mis = self.QDarchive.mispredicted
        npf = self.QDarchive.nopointsfound
        cutoff = 0.5*(2/n)**((10*d)/(t + mis - 2*npf))**0.5
        return(cutoff)

    # def find_neighbouring_regions( self, known_region):
    #     '''
    #     returns the indices of the neighbouring regions
    #     '''
    #     # Vectorise the find_neighbouring_regions function


    #     regions = []
    #     known_region = tuple(known_region.numpy())
    #     for c , dim in enumerate(known_region):
    #         if dim-1 >=0 and dim+1 < self.Domain.feature_resolution[c]:
    #             regions.append([dim-1, dim, dim+1])
    #         else:
    #             if dim-1 <0:
    #                 regions.append([ dim, dim+1])
    #             else:
    #                 regions.append([dim-1,  dim])
    #     return(list(it.product(*regions)))

    def find_neighbouring_regions(self, known_region):
        '''
        returns the indices of the neighbouring regions
        '''
        # Vectorise the find_neighbouring_regions function

        regions = []
        known_region = tuple(known_region.numpy())
        for c , dim in enumerate(known_region):
            disc = 5 # How many in each dimension. 5 -> 5^2 = 25 regions around the point
            dims = np.array(np.linspace(dim -(disc -1)/2, dim + (disc -1)/2, disc), dtype = int)
            dims = dims[dims >= 0]
            dims = dims[dims < self.Domain.feature_resolution[c]]
            regions.append(dims)
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
            distro = scipy.stats.norm( loc = mu.detach() , scale = np.sqrt( var.detach() ) )      
            cdfvals = [distro.cdf( edge ) for edge in edges[n_desc]]          
            featdimprobs = {i: ( cdfvals[i+1]-cdfvals[i] ) for i in regions[n_desc] } 
            featdimslist.append(featdimprobs)
        return(featdimslist)

    def V_get_univariate_descriptor_probs(self, mus_, var_, regions):
        '''
        returns region probabilities from the univariate descriptor distributions
        for each nieghbouring region
        '''
        
        edges = self.QDarchive.edges
        fdims = self.Domain.fdims
        featdimslist = []
        regions = np.array(regions).T
        # Vectorise this loop
        probs_array = np.zeros(regions.shape)

        for n_desc in range(fdims):
            mu = _mus[n_desc]
            var = _vars[n_desc]
            distro = scipy.stats.norm( loc = mu.detach() , scale = np.sqrt( var.detach() ) )  
            cdfvals = distro.cdf(edges[n_desc])
            featdimprobs = np.diff(cdfvals, axis=-1)
            for c , region_list in enumerate(ne):
                indexes = region_list.T[n_desc]
                probs = featdimprobs[c][indexes]
                probs_array[c][:,n_desc] = probs
            probs = np.prod(probs_array, axis = 2)