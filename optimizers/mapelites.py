import numpy as np
import torch

class MAPelites():

    def __init__(self, domain, archive):

        ## Define the problem 

        self.xl = np.array(domain.Xconstraints)[:,0]
        self.xu = np.array(domain.Xconstraints)[:,1]
        self.xdims = domain.xdims
        self.max = max
        self.domain = domain
        self.archive = archive
        self.name = 'MAP-Elites'

    def create_children(self, batchsize = 1, parents = None ):
        '''
        Creates children by evolutionary strategies 
        '''
        ## Select random parents
        parents = self.get_parents(batchsize)
        try:
            randindex1 = np.random.randint( 0 , parents.shape[0], batchsize ) 
            randindex2 = np.random.randint( 0 , parents.shape[0], batchsize ) 
        except:
            print('random parent selection failed, saving with index 0')
            randindex1 = np.array([0] * batchsize)
            randindex2 = np.array([0] * batchsize)
        parent_sample1 = parents[randindex1,:]
        parent_sample2 = parents[randindex2,:]

        directions = parent_sample2 - parent_sample1
        # Create children
        pertubation_strength = 0.1
        pertubation = np.random.normal(0, 1, (batchsize, self.xdims)) * pertubation_strength
        
        line_mutation = directions * np.random.normal(0, 1, (batchsize, self.xdims)) * 0.1
        children = parent_sample2 + pertubation + line_mutation
        # clip to bounds
        children = np.clip(children, self.xl, self.xu)

        return(children)

    def get_parents(self, batchsize = 1):
        '''
        Selects parents from the archive
        '''
        flatarchive = self.archive.genomes.flatten()
        flatnonnans = torch.isnan(flatarchive)
        parents = flatarchive[~flatnonnans].reshape(-1,self.domain.xdims)

        return(parents)

    def run(self, n_children = 1):
        '''
        runs map-elites with a batchsize of n

        returns a tensor of children n * xdims
        '''
        children = self.create_children(n_children)
        return(children)


