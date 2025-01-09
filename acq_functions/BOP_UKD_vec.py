import numpy as np
from acq_functions.base_acq import BASEacq
import torch
from torch.distributions import Normal
import itertools as it
import scipy.stats

class BOP_UKD_vec(BASEacq):
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
        self.name = 'BOP_UKD_vec'

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
        
        self.fitGP.eval()
        posterior = self.fitGP.posterior(x)
        desc_var = torch.sum(torch.stack([DGP.posterior(x).variance.clamp_min(1e-9).sqrt() for DGP in self.DGPs]), axis = 0)
        mean = posterior.mean
        sigma = posterior.variance.clamp_min(1e-9).sqrt()
        meanvect = mean.expand(mean.shape[0],fstar.shape[0] )
        val = torch.sub(meanvect.t(),fstar)
        u = torch.div(val.t() ,sigma).t()
        if min == True:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))

        ei = (sigma * ( updf + u * ucdf).t()).t() * (1 + beta *  desc_var.t()) 
        if return_mean:
            return( ei, meanvect )
        else:
            return( ei )
            
    def vec_EI(self, x, fstar_tensor, min=False , return_mean = False):
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
        #with torch.no_grad():
            #x = torch.from_numpy(np.array( [ [ x ] ] ).reshape( -1 , self.Domain.xdims ))
        x = x.double()
        
        self.fitGP.eval()
        posterior = self.fitGP.posterior(x)
        #desc_var = torch.sum(torch.stack([DGP.posterior(x).variance.clamp_min(1e-9).sqrt() for DGP in self.DGPs]), axis = 0)
        mean = posterior.mean
        sigma = posterior.variance.clamp_min(1e-9).sqrt()
        #n_fstar = 3**self.Domain.fdims
        n_fstar = np.product(self.Domain.feature_resolution)
        meanvect = mean.repeat(1, n_fstar).reshape(n_fstar*mean.shape[0],-1)
        sig = sigma.repeat(1,n_fstar).reshape(n_fstar*sigma.shape[0],-1)
        val = torch.sub(meanvect,fstar_tensor)
        u = torch.div(val ,sig)
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))

        ei = (sig * ( updf + u * ucdf)) 
        if return_mean:
            return( ei, meanvect )
        else:
            return( ei )

    def evaluate(self, x, beta = 1, fstar = None):
        
        if type(x) == torch.Tensor:
            #x = x.reshape(-1,self.Domain.xdims, requires_grad = True)
            pass
        else:
            x = torch.tensor(x, dtype = self.dtype).reshape(-1,self.Domain.xdims);
        
        if x.shape[0] == 1:
            # check if a duplicate
            x = x.reshape(-1,self.Domain.xdims)
            return(self.evaluate_single(x,  beta, fstar))
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

    def vectorised_evaluate(self, x, beta = 1):
        # Find cluster of regions        
        central_region = self.predict_region(x)[0]
        region_probs = self.region_probabilities(x, central_region)

        # Find fstars for each region        
        fstar_dict = self.find_all_fstar(x, region_probs)

        # Convert to tensor
        fstar_list = [fstar_dict[region] for region in region_probs]
        fstar_tensor = torch.stack(fstar_list).reshape(-1,1)

        # Calculate EI and times by probability
        fitness = self.EI( x, fstar_tensor ,beta)
        probs = [p for p in region_probs.values()]
        prob_tens = torch.tensor(probs, dtype = torch.double).squeeze()
        product = prob_tens * fitness
        value = torch.sum(product, axis = 0)
        return(value)  



    def grad_vectorised_evaluate(self, x, beta):
        # Find cluster of regions for each x
        #neighbouring_regions = self.grad_get_regions(x)
        neighbouring_regions = self.grad_get_ALL_regions(x)

        # Get probability of being in each region, for each x
        probs = self.grad_region_probabilities(x, neighbouring_regions)
        #cutoff = self.calculate_cutoff()
        #probs[probs < cutoff] = 0

        # Find fstars for each region 
        #fstars = np.zeros(neighbouring_regions.shape[:2])
        #for c, neighbour in enumerate(neighbouring_regions):
        #    fstar_list = [self.QDarchive.stdfitness[tuple(i)] for i in neighbour]
        #    fstars[c] = torch.stack(fstar_list).squeeze()   
        
        ### THIS is 
        fstars = self.QDarchive.stdfitness.reshape(-1).repeat(x.shape[0],1)
        
        # Turn fstars into a long tensor
        #fstar_tensor = torch.tensor(fstars, dtype = torch.double).reshape(-1,1)  
        fstar_tensor = fstars.reshape(-1,1).double()
        fstar_is_nan = torch.isnan(fstar_tensor)
        fstar_tensor[fstar_is_nan] = self.stdfstar0

        # Calculate EI and times by probability

        fitness = self.vec_EI( x, fstar_tensor)
        prob_tens = torch.tensor(probs, dtype = torch.double).reshape_as(fitness)
        # normalise probabilities
        #prob_tens_reshaped = prob_tens.reshape(-1,3**self.Domain.fdims)
        prob_tens_reshaped = prob_tens.reshape(-1,np.product(self.Domain.feature_resolution))
        prob_sums = torch.sum(prob_tens_reshaped, axis = 1)
        #prob_sums = prob_sums.repeat(3**self.Domain.fdims,1).T.reshape(-1)
        prob_sums = prob_sums.repeat(np.product(self.Domain.feature_resolution),1).T.reshape(-1)
        #prob_tens = torch.div(prob_tens.T , prob_sums).T
        
        product = prob_tens * fitness #* (1+beta)
        #product_reshaped = product.reshape(-1,3**self.Domain.fdims)
        product_reshaped = torch.stack([product[i:i+x.shape[0]] for i in range(0,product.shape[0],x.shape[0])])
        #product_reshaped = product.reshape(x.shape[0], -1)
        value = product_reshaped.sum(1)
        return(value)

    def grad_get_regions(self, x):
        # Find cluster of regions
        central_regions = self.predict_region(x)
        # Find probabilities of each region
        fdims = self.Domain.fdims
        # List all neighbouring regions
        neighbouring_regions = [self.find_neighbouring_regions(region) for region in central_regions]
        ## Check that each region has the same number of neighbours
        pad = 3**fdims
        for c,neighbours in enumerate(neighbouring_regions):
            if len(neighbours) < pad:
                neighbouring_regions[c] = neighbouring_regions[c] + [tuple([-1]*fdims)] * (pad - len(neighbours))
        neighbouring_regions = torch.tensor(neighbouring_regions, dtype = torch.int64)  
        return(neighbouring_regions)

    def grad_get_ALL_regions(self, x):
        all_regions = torch.tensor(list(it.product(*[range(featdim) for featdim in self.Domain.feature_resolution])))
        neighbouring_regions = [all_regions for point in x]
        neighbouring_regions = torch.stack(neighbouring_regions)
        return(neighbouring_regions)

    def grad_region_probabilities(self, x, neighbouring_regions):
                # Get the posterior mean and variance of x for each descriptor model
        _mus, _vars = self.getmuvar(x)
        _mus = torch.stack(_mus)  # Now they are 
        _vars = torch.stack(_vars)

        edges = self.QDarchive.edges
        fdims = self.Domain.fdims

        neighbouring_regions = np.array(neighbouring_regions)
        probs_array = np.zeros(neighbouring_regions.shape)

        ## we make probs an array of n x fdims where n is the number of points
        ## and n is the number of neighbouring regions
        for n_desc in range(fdims):
            mu = _mus[n_desc]
            var = _vars[n_desc]
            distro = scipy.stats.norm( loc = mu.detach() , scale = np.sqrt( var.detach() ) )  
            cdfvals = distro.cdf(edges[n_desc])
            featdimprobs = np.diff(cdfvals, axis=-1)
            for c , region_list in enumerate(neighbouring_regions):
                indexes = region_list.T[n_desc]
                probs = featdimprobs[c][indexes]
                probs_array[c][:,n_desc] = probs
        # Then multiply by probability in each dimension to get probability of 
        # being in each region
        probs = np.prod(probs_array, axis = 2)
        return(probs)






    def evaluate_single(self, x , beta, fstar = None):
        # Find cluster of regions
        central_region = self.predict_region(x)[0]
        region_probs = self.region_probabilities(x, central_region)

        # Find fstars for each region        
        fstar_dict = self.find_all_fstar(x, region_probs)

        # Convert to tensor
        fstar_list = [fstar_dict[region] for region in region_probs]
        fstar_tensor = torch.stack(fstar_list).reshape(-1,1)

        # Calculate EI and times by probability
        fitness = self.EI( x, fstar_tensor , beta)
        probs = [p for p in region_probs.values()]
        prob_tens = torch.tensor(probs, dtype = torch.double)
        value = torch.sum(prob_tens.T * fitness.T)#reshape(len(region_probs),)).unsqueeze(0)
        return(value.unsqueeze(0))   

    def check_for_misprediction(self, x ):
        # Find cluster of regions
        print(x.shape)
        assert (x.shape[0] == 1 and x.shape[1] == self.Domain.xdims) or x.shape[0] == self.Domain.xdims, "x must be a single point"
        x = x.unsqueeze(0)
        central_region = self.predict_region(x)[0]
        region_probs = self.region_probabilities(x, central_region)

        # Find fstars for each region        
        fstar_dict = self.find_all_fstar(x, region_probs)

        # Calculate EI and times by probability
        values_dict = {}

        for c,region in enumerate(region_probs):            
            values_dict[region] =  self.EI( x, fstar_dict[region], 1 ) * region_probs[region]
                 
        total_value = sum(values_dict.values())

        for c,region in enumerate(region_probs):            
            values_dict[region] =  values_dict[region] / total_value

        ## Provide feedback on the best region
        best_region = max(values_dict, key=values_dict.get)
        best_prob = region_probs[best_region]
        bestval = values_dict[best_region].numpy()[0][0]*total_value.item()
        valpercent = values_dict[best_region].numpy()[0][0]*100
        toral_val = sum(values_dict.values())
        print(f'max_value region prob. {best_prob} in {best_region}, contributing: {bestval}, {valpercent}% of total')
        if region_probs[best_region] < 0.5 and values_dict[best_region] > 0.5:
            return(True, total_value)
        else:
            return(False, total_value)
     

    def find_all_fstar(self, x, region_probs = None):
        '''
        Finds the fstar for some index/s in the archive
        '''
        fstar_dict = {}
        for region in region_probs:
            # find existing elite in archive
            fstar = self.QDarchive.stdfitness[region]
            # If no elite exists, use the default fstar
            if torch.isnan(fstar) or fstar == None:
                fstar = self.stdfstar0
            fstar_dict[region] = torch.tensor([fstar], dtype = self.dtype)

        return(fstar_dict)

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
        probsum = np.sum(nprobs,axis = 0)
        probsum[probsum == 0] +=1
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

    def find_neighbouring_regions( self, known_region):
        '''
        returns the indices of the neighbouring regions
        '''
        # Vectorise the find_neighbouring_regions function


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

    def get_univariate_descriptor_probs(self, mus_, var_, regions):
        '''
        returns region probabilities from the univariate descriptor distributions
        for each nieghbouring region
        '''
        
        edges = self.QDarchive.edges
        fdims = self.Domain.fdims
        featdimslist = []
        regions = np.array(regions).T
        # Vectorise this loop
        for n_desc in range(fdims):
           mu = mus_[n_desc]
           var = var_[n_desc]
           distro = scipy.stats.norm( loc = mu.detach() , scale = np.sqrt( var.detach() ) )      
           cdfvals = [distro.cdf( edge ) for edge in edges[n_desc]]          
           #cdfvals = distro.cdf(edges).T 
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
