import numpy as np
from acq_functions.base_acq import BASEacq
import torch
from torch.distributions import Normal
import itertools as it
import scipy.stats

class BOP_UKD(BASEacq):
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
        self.name = 'BOP_UKD'

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
        #with torch.no_grad():
            #x = torch.from_numpy(np.array( [ [ x ] ] ).reshape( -1 , self.Domain.xdims ))
        x = x.double()
        
        self.fitGP.eval()
        posterior = self.fitGP.posterior(x)
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
            ## check if a duplicate
            if x in self.fitGP.train_inputs[0]:
                return(torch.tensor([0]))
            return(self.evaluate_single(x, fstar))
        else:
            ## As multiple points are only ever evaluated in the point finding
            ## process, we assume duplicates are unimportant.
            return(self.vectorised_evaluate(x))

    def vectorised_evaluate(self, x):
        # Find cluster of regions
        central_region = self.predict_region(x)[0]
        region_probs = self.region_probabilities(x, central_region)

        # Find fstars for each region        
        fstar_dict = self.find_all_fstar(x, region_probs)

        # Convert to tensor
        fstar_list = [fstar_dict[region] for region in region_probs]
        fstar_tensor = torch.stack(fstar_list).reshape(-1,1)

        # Calculate EI and times by probability
        fitness = self.EI( x, fstar_tensor )
        probs = [p for p in region_probs.values()]
        prob_tens = torch.tensor(probs, dtype = torch.double).squeeze()
        product = prob_tens * fitness
        value = torch.sum(product, axis = 0)
        return(value)  

    def evaluate_single(self, x , fstar = None):
        # Find cluster of regions
        
        central_region = self.predict_region(x)[0]
        region_probs = self.region_probabilities(x, central_region)

        # Find fstars for each region        
        fstar_dict = self.find_all_fstar(x, region_probs)

        # Convert to tensor
        fstar_list = [fstar_dict[region] for region in region_probs]
        fstar_tensor = torch.stack(fstar_list).reshape(-1,1)

        # Calculate EI and times by probability
        fitness = self.EI( x, fstar_tensor )
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
            values_dict[region] =  self.EI( x, fstar_dict[region] ) * region_probs[region]
                 
        total_value = sum(values_dict.values())

        for c,region in enumerate(region_probs):            
            values_dict[region] =  values_dict[region] / total_value

        ## Provide feedback on the best region
        best_region = max(values_dict, key=values_dict.get)
        best_prob = region_probs[best_region]
        bestval = values_dict[best_region].numpy()[0][0]*total_value.item()
        valpercent = values_dict[best_region].numpy()[0][0]*100
        print(f'Max probability {best_prob} in {best_region}, best value {bestval}, {valpercent}% of total')
        if region_probs[best_region] < 0.5 and values_dict[best_region] > 0.5:
            return(True)
        else:
            return(False)
     

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
            nprobs = np.array([np.multiply(*[featdimslist[c][int(i)] 
                        for c,i in enumerate(region)]) 
                        for region in regions] 
            )

        # normalise the probabilities
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