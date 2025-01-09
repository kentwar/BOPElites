from benchmarks.BaseBenchmark import BaseExperiment
import math
import numpy as np
import os 
import copy
import torch
from benchmarks.pyAFT.ffdFoil import *
from benchmarks.pyAFT.foil_eval import *
import benchmarks.pyAFT.parsec as prs
from numpy import isnan
from botorch.sampling import SobolEngine

class Parsec(BaseExperiment):
    def __init__(self, feature_resolution , seed = None):
        
        kwargs =    {
            'example_x' : [0]*10 ,
            'Xconstraints' : [[0, 1 ]]*10 ,
            'featmins' : [0,0] ,
            'featmaxs' : [1,1] ,
            'lowestvalue' : 0 ,
            'maxvalue' : 1 ,
            }
        self._set_Xconstraints(np.array(kwargs['Xconstraints']))    #input x ranges [min,max]
        self.example_x = kwargs['example_x']
        self.xdims = len(kwargs['example_x']) 
        self.fdims = len(kwargs['featmins']) 
        self.featmins = kwargs['featmins']
        self.featmaxs = kwargs['featmaxs']
        self.feature_resolution = feature_resolution
        self.lowestvalue = kwargs['lowestvalue']
        self.maxvalue = kwargs['maxvalue']
        self.seed = seed
        self.name = 'Parsec'
        self.desc1name = 'X_up'
        self.desc2name = 'Z_up'
        self.fitness_fun = self.ffd_fit()
        self.foil = prs.Airfoil()
        self.feasibility_data = None
        # Evaluate base foil
        self.base_coord = self.foil.express()
        self.basefit , self.basefeats = evalFoil(self.base_coord )
        self.baselift , self.basearea = self.basefeats

    def ffd_fit(self):
        foil = prs.Airfoil()

        def ffd_true_fit( params , verbose = False):
            new_coords = foil.express( params = params )
            # Evaluate deformed foil
            drag , feats = evalFoil( new_coords )
            lift , area = feats   
        
            if lift < self.baselift:
                lift_penalty = (lift/self.baselift)**2
            else: lift_penalty = 1

            #if area > basearea:
            area_penalty = ( 1 - np.abs( area - self.basearea )/self.basearea )**7
            #else: area_penalty = 1
            
            true_fitness = drag * lift_penalty * area_penalty 

            if verbose:
                print('base lift ', self.baselift )
                print('base area ', self.basearea ,'\n')
                print( 'new lift  ' , lift)
                print( 'new area  ' , area , '\n')
                print( 'lift pen =' , lift_penalty)
                print( 'areapen =' , area_penalty)
                print( '-log(cd) = ' , drag , '\n')
                print( 'final fitness = ' , true_fitness )
            return( true_fitness +0.1, drag, lift )
        
        def true_fit_runner( X , verbose = False ):
            t = type(X)
            s = np.shape(X)
            ms = np.shape(X[0])
            assert t == np.ndarray, 'Input to the fitness function must be an array'
            assert s == (1,self.xdims) or ms == (self.xdims,), 'genome is the wrong shape'
            ## Single Point
            if s == (1,self.xdims):
                return( ffd_true_fit(X , verbose) )
            else:
                fitness = [ffd_true_fit(x) for x in X]
                return(fitness)            

        return( true_fit_runner )
        
    

    def feature_fun(self, X ):
        t = type(X)
        s = np.shape(X)
        ms = np.shape(X[0])
        assert t == np.ndarray, 'Input to the fitness function must be an array'
        assert s == (1,self.xdims) or ms == (self.xdims,), 'genome is the wrong shape'
        x_up = [x[1] for x in X]
        z_up = [x[2] for x in X]
        if s == (1,self.xdims):
            return( np.array([x_up[0],z_up[0]]) )
        else:
            return( np.array(list(zip(x_up,z_up))) )

    def sample(self, rescaled_X, missing_points, init_samples = []):
        x = rescaled_X
        fitness_drag_lift = self.fitness_fun(rescaled_X.numpy().reshape(-1, self.xdims))
        fit_drag_lift = np.array(fitness_drag_lift)
        fitness = fit_drag_lift[:,0]
        drag = fit_drag_lift[:,1]
        lift = fit_drag_lift[:,2]
        descriptors = self.feature_fun(rescaled_X.numpy().reshape(-1, self.xdims))
        # This produces a binary indicator for "fit is nan" OR "desc is nan"
        invalid = isnan(fitness) + isnan(descriptors).any()
        #num_valid = min([missing_points,sum(~invalid)])
        if self.feasibility_data == None:
            self.feasibility_data = [[x[c] , int(invalid[c])] for c in range(len(x))]
        else:
            new_data = [[x[c] , int(invalid[c])] for c in range(len(x))]
            self.feasibility_data.extend(new_data)
        samples = []

        for count in range(len(invalid)):
            if missing_points == 0:
                break
            if invalid[count]:
                pass
            else:
                samples.append([x[count], fitness[count],drag[count], lift[count], descriptors[count]])
                missing_points -= 1
        if init_samples == []:
            init_samples = samples
        else:
            init_samples += samples
        return(init_samples, missing_points)

    ### Tools to predict the fitness of a point

    def polyArea(self, x,y): # Shoelace formula
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))) 

    def calc_lift_pen(self, lift):   
        lift_pen = (lift <= self.baselift) * (lift/self.baselift)**2 + (lift > self.baselift)
        return(lift_pen)

    def calc_area(self, airfoil):
        airfoil = self.foil.express( params = airfoil)
        x = airfoil[0,:]
        y = airfoil[1,:]
        area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))) 
        return(area)

    def calc_area_pen(self, area):
        area_penalty = ( 1 - np.abs( area - self.basearea )/self.basearea )**7
        return(area_penalty)

    def calc_fitness(self, x, pred_drag, pred_lift):
        area = self.calc_area(x)
        lift_pen = self.calc_lift_pen(pred_lift)
        area_pen = self.calc_area_pen(area)
        return(np.multiply(np.multiply(pred_drag, lift_pen), area_pen) )

    def vec_calc_fitness(self, x, pred_drag, pred_lift):
        area = self.calc_area(x)
        lift_pen = self.calc_lift_pen(pred_lift)
        area_pen = self.calc_area_pen(area)
        return(np.multiply(np.multiply(pred_drag, lift_pen), area_pen) )

    def get_sample(self, n): 
        missing_points = copy.copy(n)
        init_samples = [] # [x, fit, desc]
        while missing_points > 0:
            self.sobolengine = SobolEngine(self.xdims, scramble=True, seed = self.seed)
            init_x = self.sobolengine.draw(n, dtype=torch.double)
            rescaled_x = self.rescale_X( init_x )
            init_samples , missing_points = self.sample(rescaled_x, missing_points, init_samples)
            print(f'sampled {n - missing_points} points of {n}')
        return(init_samples)