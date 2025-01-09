import numpy as np
import math
import itertools as it
import torch
import copy
from sklearn.neighbors import NearestNeighbors

class structured_archive:
    '''
    A base class for a structured Quality-Diversity archive
    '''
    def __init__(self, domain):
        '''
        niche_sizes : a d dimensional list of niche_sizes
        domain      : a domain class object 
        '''
        
        # Grab info from domain
        self._init_from_domain(domain)
        self.domain = domain

        #Add niche boundaries/edges to archive
        self._initialize_archive_edges()
        
        # Create empty array objects to store Fitness, Genomes nans, models
        self.blank_archive        = torch.empty( self.feature_resolution ) #Useful for niche membership calculations
        self.blank_archive[ : ]   = float('nan')
        self.flush() # Flushes the archive of all data
        self.valid_ranges = torch.tensor(domain.Xconstraints)
        self.genomes = np.clip(self.genomes, self.valid_ranges[:,0], self.valid_ranges[:,1])
        self.dtype = torch.double
        self.mispredicted = 0
        self.nopointsfound = 0

    def flush(self):
        '''
        Flushes the archive of all data
        '''
        self.fitness     = torch.clone(self.blank_archive)
        self.means       = torch.clone(self.blank_archive)
        self.stdmeans    = torch.clone(self.blank_archive)
        self.observed_Ys = np.empty(self.feature_resolution,dtype=object)
        # Standardised versions
        self.stdfitness  = torch.clone(self.blank_archive)
        self.stdfitness[ : ]   = float('nan')
        self.std_Ys      = np.empty(self.feature_resolution,dtype=object)
        self.std_means   = torch.clone(self.blank_archive)
        self.nanstore    = []
        self.models      = []
        # Establish empty place holder for genomes
        self.genomes = torch.empty( ( *self.feature_resolution , self.xdims ) , dtype = torch.double )
        self.genomes[:,:] = np.float64('nan')

    def _init_from_domain(self, domain):
        '''
        initialise data from a domain class object
        '''
        try:
            self.xdims = len(domain.example_x)
            self.fdims = len(domain.feature_resolution)
            self.fmins = domain.featmins
            self.fmaxs = domain.featmaxs
            self.feature_resolution = domain.feature_resolution
        except: 
            raise FileNotFoundError("There was an error loading data from domain")

    def _initialize_archive_edges(self ): 
        '''
        Creates the edges used to define regions
        '''
        self.edges = []
        # Define edges in each descriptor dimension
        for i in range( self.fdims ):
            edge_boundaries = np.linspace( self.fmins[ i ] , 
                                           self.fmaxs[ i ],
                                           self.feature_resolution[ i ] + 1 )
            self.edges.insert( i, edge_boundaries ) 
        
        # Convert to array
        self.edges = np.array( self.edges )
    
    def iterdims(self ):
        '''
        Produce an iterable list of indexes in the solution archive
        '''
        genome_dims = [ range( i ) for i in self.feature_resolution  ] 
        return(it.product( *genome_dims ))

    # def nichefinder(self, behaviour):
    #     '''
    #     identifies the niche points belong to
    #     WARNING: nichefinder will conform behaviours to the dimensions
    #     and will not report points that lie outside the valid range
    #     '''
    #     ## Check if it's a single behaviour or a list of behaviours
    #     if len(behaviour.shape) == 1:
    #         behaviour = behaviour.reshape(-1,self.fdims)
    #     niches = np.empty((0,self.fdims))
    #     for count, edge in enumerate(self.edges):   
    #         digitized = np.digitize(behaviour[:, count], edge[1:-1], right=False)
    #         niches = np.vstack([niches , digitized])
    #     return(torch.tensor(niches).int().T)

    # def nichefinder(self, behaviour):
    #     '''
    #     identifies the niche points belong to
    #     WARNING: nichefinder will conform behaviours to the dimensions
    #     and will not report points that lie outside the valid range
    #     '''
    #     if self.domain.fdims == 1:
    #         behaviour = behaviour.reshape(-1, self.fdims)
    #         digitized = [np.digitize(behaviour[:, count], edge[1:-1], right=False) for count, edge in enumerate(self.edges)]
    #         niches = np.vstack(digitized)
    #         return torch.tensor(niches).int().T
        
    #     ## Check if it's a single behaviour or a list of behaviours
    #     if len(behaviour.shape) == 1:
    #         behaviour = behaviour.reshape(-1,self.fdims)
    #     niches = np.empty((0,behaviour.shape[0]))
    #     for count, edge in enumerate(self.edges):
    #         digitized = np.digitize(behaviour[:, count], edge[1:-1], right=False)
    #         niches = np.vstack([niches , digitized])
    #     return torch.tensor(niches).int().T

    def nichefinder(self, behaviour):
        '''
        identifies the niche points belong to, returns -1 if it falls outside the bounds.
        '''
        if self.domain.fdims == 1:
            behaviour = behaviour.reshape(-1, self.fdims)
            digitized = [np.digitize(behaviour[:, count], edge[1:-1], right=False) for count, edge in enumerate(self.edges)]
            niches = np.vstack(digitized)
            niches[(behaviour < edge[0]).nonzero()] = -1
            niches[(behaviour > edge[-1]).nonzero()] = -1
            return torch.tensor(niches).int().T

        ## Check if it's a single behaviour or a list of behaviours
        if len(behaviour.shape) == 1:
            behaviour = behaviour.reshape(-1,self.fdims)
        niches = np.empty((0,behaviour.shape[0]))
        for count, edge in enumerate(self.edges):
            digitized = np.digitize(behaviour[:, count], edge[1:-1], right=False)
            digitized[(behaviour[:, count] < edge[0]).nonzero()] = -1
            digitized[(behaviour[:, count] > edge[-1]).nonzero()] = -1
            niches = np.vstack([niches , digitized])
        output = torch.tensor(niches).int().T
        return (output)

    def nichefinder(self, behaviour):
        '''
        identifies the niche points belong to, returns -1 if it falls outside the bounds.
        '''
        if type(behaviour) == np.ndarray:
            behaviour = torch.tensor(behaviour)
        if self.domain.fdims == 1:
            behaviour = behaviour.reshape(-1, self.fdims)
            digitized = [torch.bucketize(behaviour[:, count].contiguous(), torch.as_tensor(edge[1:-1]), right=False) for count, edge in enumerate(self.edges)]
            niches = np.vstack(digitized)
            niches[(behaviour < edge[0]).nonzero()] = -1
            niches[(behaviour > edge[-1]).nonzero()] = -1
            return torch.tensor(niches).int().T

        ## Check if it's a single behaviour or a list of behaviours
        if len(behaviour.shape) == 1:
            behaviour = behaviour.reshape(-1,self.fdims)
        niches = None
        for count, edge in enumerate(self.edges):
            digitized = torch.bucketize(behaviour[:, count].contiguous(), torch.tensor(edge[1:-1]), right=False)
            digitized[(behaviour[:, count] < edge[0]).nonzero()] = -1
            digitized[(behaviour[:, count] > edge[-1]).nonzero()] = -1
            if niches == None:
                niches = digitized
            else:
                niches = torch.stack([niches , digitized])
        output = niches.int().T
        return (output)

    def get_region(self, x):
        '''
        Returns the region that a point belongs to
        '''
        if isinstance(x, torch.Tensor):
            descriptor = self.domain.feature_fun(x.reshape(-1, self.xdims))
            #descriptor = self.domain.feature_fun(x.numpy().reshape(-1, self.xdims))
            region = self.nichefinder(descriptor)
        else:
            descriptor = self.domain.feature_fun(x.reshape(-1, self.xdims))
            region = self.nichefinder(torch.tensor(descriptor))
        return(region)

    def updatearchive( self, new_points , ymean, ystd, verbose = False, return_added = False):
        '''
        Adds new observations to the archive and calculates the 
        percentage of points that were accepted in to the archive
        Also updates the standardised fitness values (used with std GP)
        '''
        accepted_n = 0
        
        index_list = self._getindexlist(new_points)
        filtered_indices = [i for i, tup in enumerate(index_list) if -1 not in tup]
        index_list = [index_list[i] for i in filtered_indices]
        new_points = [new_points[i] for i in filtered_indices]
        accepted_points = {}
        if len(index_list) == 0:
            return(0)
        for count, point in enumerate(new_points):
            index = index_list[count]
            if np.isnan( self.fitness[ index ].detach() ):
                self.genomes[ index ] = point[ 0 ]
                self.fitness[ index ] = point[ 1 ].detach()
                accepted_points[index] = point[0]
                accepted_n += 1
            else:
                if point[ 1 ] > self.fitness[ index ]:
                    self.genomes[ index ] = point[ 0 ]
                    self.fitness[ index ] = point[ 1 ]
                    if index not in accepted_points.keys():
                        accepted_points[index] = point[0]
                    else:
                        accepted_points[index] = torch.cat([accepted_points[index], point[0]], dim=0)
                    accepted_n += 1
                else:
                    region_not_updated = index not in accepted_points.keys()
                    behaviour = self.domain.feature_fun(self.genomes[index].reshape(-1, self.xdims)).detach()
                    if not hasattr(self, 'arch_beh'):
                        #print('rebuilding archive behaviours')
                        archive_behaviours = self.domain.feat_fun(self.genomes.reshape(-1,self.domain.xdims)).detach()
                        self.arch_beh = archive_behaviours[~torch.isnan(archive_behaviours).any(axis=1)].detach().numpy()                    
                    archive_behaviours = self.arch_beh
                    novelty = novelty_score(archive_behaviours, behaviour, 1)
                    #distance_to_point = (behaviour - point[2] > 2/torch.tensor(self.feature_resolution)/3).any()
                    #value_cond = (point[1]/self.fitness[index]) > 0.9
                    novelty_condition = novelty > 0
                    #if distance_to_point and region_not_updated:
                    search_space_distance = torch.norm(self.genomes[index] - point[0])
                    
                    if region_not_updated and (novelty_condition or search_space_distance > np.sqrt(self.domain.xdims)):
                        if index not in accepted_points.keys():
                            accepted_points[index] = point[0]
                        else:
                            accepted_points[index] = torch.cat([accepted_points[index], point[0]], dim=0)

        self.stdfitness = (self.fitness - ymean)/ystd
        
        #self._updatepointmean(new_points)

        if return_added:
            if verbose:
                return( accepted_n / len(new_points), accepted_points )
            else:
                return( accepted_points )

        if verbose:
            if count == 0:
                return(0)
            else:
                return( accepted_n / len(new_points) )

    def initialise( self, initial_points , standardize=False):
        '''Takes initial random points and correctly allocates them in the 
        solution archive only using the best performing points
        '''
        if standardize:
            yvals = [p[1] for p in initial_points]
            stdvals = (yvals-np.mean(yvals))/(np.std(yvals)+0.0001)
            for i in range(len(initial_points)):
                xs = [p[0] for p in initial_points]
                feats = [o[2] for o in initial_points]
                initial_points = [[xs[i],
                                    stdvals[i],
                                    feats[i]] for i in range(len(yvals))]
          
        index_list = self._getindexlist(initial_points)
        filtered_indices = [i for i, tup in enumerate(index_list) if -1 not in tup]
        index_list = [index_list[i] for i in filtered_indices]
        initial_points = [initial_points[i] for i in filtered_indices]
        for count, point in enumerate(initial_points):  
            index = index_list[count]            
            if torch.isnan( self.fitness[ index ] ):
                self.genomes[ index ] = point[ 0 ]
                self.fitness[ index ] = point[ 1 ]
            else:
                if point[ 1 ] > self.fitness[ index ]:
                    self.genomes[ index ] = point[ 0 ]
                    self.fitness[ index ] = point[ 1 ]
            self._add_observation(point[1], index )
        pass
        #self.obsmean = np.nanmean(self.fitness.flatten())
        #self.obsstd = np.nanstd(self.fitness.flatten()) 
        #if len(initial_points)>0:
        #    self._initialisemeans(initial_points) 

    def zoom_initialise( self, initial_points , standardize=False):
        '''Takes initial random points and correctly allocates them in the 
        solution archive only using the best performing points
        '''
        if standardize:
            yvals = [p[1] for p in initial_points]
            stdvals = (yvals-np.mean(yvals))/(np.std(yvals)+0.0001)
            for i in range(len(initial_points)):
                xs = [p[0] for p in initial_points]
                feats = [o[2] for o in initial_points]
                initial_points = [[xs[i],
                                    stdvals[i],
                                    feats[i]] for i in range(len(yvals))]
        index_list = self._getindexlist(initial_points)
        filtered_indices = [i for i, tup in enumerate(index_list) if -1 not in tup]
        initial_points = [initial_points[i] for i in filtered_indices]
        index_list = self._getindexlist(initial_points)
        for count, point in enumerate(initial_points):  
            index = index_list[count]            
            if torch.isnan( self.fitness[ index ] ):
                self.genomes[ index ] = point[ 0 ]
                self.fitness[ index ] = point[ 1 ]
            else:
                if point[ 1 ] > self.fitness[ index ]:
                    self.genomes[ index ] = point[ 0 ]
                    self.fitness[ index ] = point[ 1 ]
            self._add_observation(point[1], index )
        self.obsmean = np.nanmean(self.fitness.flatten())
        self.obsstd = np.nanstd(self.fitness.flatten()) 

        self._initialisemeans(initial_points) 

    def update_stdfitness(self, mean, std):
        '''
        Updates the standardised fitness values in the archive
        '''
        self.stdfitness = (self.fitness - mean)/std

    def _add_observation(self,  yval , index):
        '''
        adds a new point in to the list of observed values for a region
        '''
        if np.shape(self.observed_Ys[index]) == ():
            self.observed_Ys[index] = [yval.detach().numpy()]
        else:
            self.observed_Ys[index] = np.append(self.observed_Ys[index], yval.detach().numpy())

    def _initialisemeans( self, points , standardize = False):
        '''Calculates the mean observed value for each niche and populates
        self.means with those mean values. 
        '''

        if standardize:
            s_points = []
            ys = torch.tensor([p[1] for p in points])
            std_y = (ys - ys.mean())/(ys.std())
            for i,p in enumerate(points):
                s_points.append([p[0],std_y[i],p[2]])
        index_list = self._getindexlist(points)        
        for c, p in enumerate(points):
            index = index_list[c]            
            self.observed_Ys[ index ].append(p[1])
            yvals_in_region = torch.tensor(self.observed_Ys[ index ], dtype = torch.double)
            self.means[index] = torch.mean(yvals_in_region)

    # def _getindexlist(self, points):
    #     behaviours = torch.stack([p[2] for p in points])
    #     index_list = self.nichefinder(behaviours)
    #     indexes = [tuple(index.numpy()) for index in index_list]
    #     return(indexes)   
    def _getindexlist(self, points):
        behaviours = torch.stack([p[2] for p in points])
        index_list = self.nichefinder(behaviours)
        indexes = [tuple(index.numpy()) for index in index_list]
        return(indexes)

    def _updatepointmean( self, newpoints):
        index_list = self._getindexlist(newpoints)
        for count, point in enumerate(newpoints):
            index = index_list[count]
            if self.observed_Ys[index] == None:
                self.observed_Ys[index] = [point[1]]
            else:
                self.observed_Ys[ index ].append(point[1])
            self.means[index] = torch.mean(torch.tensor(self.observed_Ys[index]))
    
    def calculate_fitness(self):
        '''Calculates the fitness of each point in the archive
        '''
        return(np.nansum(self.fitness))
    
    def get_num_niches(self):
        '''Calculates the fitness of each point in the archive
        '''
        return(torch.sum(~torch.isnan(self.fitness.flatten())))

    def return_copy(self):
        '''
        Returns a copy of the archive object
        deepcopy can fail due to tensor leaf nodes,
        so we need to detach the tensors before copying
        '''
        copy_archive = structured_archive(self.domain)
        copy_archive.genomes = self.genomes.detach().clone()
        copy_archive.fitness = self.fitness.detach().clone()
        copy_archive.stdfitness = self.stdfitness.detach().clone()
        copy_archive.means = self.means.detach().clone()
        copy_archive.observed_Ys = self.observed_Ys
        #copy_archive.obsstd = self.obsstd
        return(copy_archive)


def novelty_score(archive_B, candidate_b, k):
    """
    Calculate the novelty score for a new point x using an efficient k-nearest neighbor search.
    
    Parameters:
    - archive_B: np.ndarray, shape (n_points, n_features), the existing set of behaviours.
    - candidate_b: np.ndarray, shape (n_features,), the new behaviours.
    - k: int, the number of nearest neighbors to consider for the novelty score.
    
    Returns:
    - float, the novelty score of the new point x.
    """
    # Initialize the NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(archive_B)
    
    # Reshape x to shape (1, n_features) and find k nearest neighbors
    distances, _ = nn.kneighbors(candidate_b.reshape(1,-1), n_neighbors=k)
    
    # Calculate the novelty score as the average of the k nearest neighbor distances
    novelty_score = np.mean(distances)
    
    return novelty_score
# class domain:
    
#     def __init__(self):
#         self.example_genome = np.array([1,2,3])
#         self.feature_resolution = [10,10]
#         self.feat_mins = [0,0]
#         self.feat_maxs = [1,1]
#         self.valid_ranges = [[0,1]*len(self.example_genome)]

#my_domain = domain()
#QDarchive = structured_archive(my_domain)
