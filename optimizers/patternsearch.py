from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem 
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
import numpy as np
import sys
from tqdm import tqdm

class PatternSearchOptimizer():

    def __init__(self, objective_function, domain, init_beta = None, max = True, termination = None):
        self.algorithm = None
        self.set_termination(termination) 

        ## Define the problem 

        self.xl = np.array(domain.Xconstraints)[:,0]
        self.xu = np.array(domain.Xconstraints)[:,1]
        self.xdims = domain.xdims
        self.max = max
        self.set_obj(objective_function)


    def set_termination(self, termination = None):
        if termination == None:
            self.termination = DefaultSingleObjectiveTermination(
                    xtol=1e-8,
                    cvtol=1e-6,
                    ftol=1e-4,
                    n_max_gen=100,
                    n_max_evals=1000
                    ) 
        else:
            self.termination = termination

    def pymoosettings(self,x0):
        self.algorithm = PatternSearch(x0 = x0)

    def run(self, x0):
        x0 = np.array(x0)
        self.pymoosettings(x0)
        res = minimize(self.problem ,
                       self.algorithm,
                       termination=self.termination,
                       seed=1,
                       verbose=False)
        return(res)

    def run_many(self, x0):
        X = []
        F = []
        for x in tqdm(x0):
            x = np.array(x)
            res = self.run(x)
            X.append(res.X)
            F.append(-res.F)
        return(X, F)

    def set_obj(self, acq, beta = None):
        if not beta == None:
            print('beta is not used in PatternSearchOptimizer')
        if self.max:
            def obj(x):
                return -acq(x)
        else:    
            obj = acq
        def wrapped(x):
            return(obj(x)[0].double().item())
            
        self.problem = FunctionalProblem(self.xdims,
                                        objs = wrapped,
                                        xl=self.xl,
                                        xu=self.xu
                                        )


