from pymoo.util.termination.default import SingleObjectiveDefaultTermination
#from scipy.optimize import minimize
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem 
from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
import numpy as np
import sys
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
import pathos.multiprocessing as mp

class PatternSearchOptimizer():

    def __init__(self, objective_function, domain, init_beta, max = True, termination = None):
        self.algorithm = None
        self.set_termination(termination) 

        ## Define the problem 

        self.xl = np.array(domain.Xconstraints)[:,0]
        self.xu = np.array(domain.Xconstraints)[:,1]
        self.xdims = domain.xdims
        self.max = max
        self.set_obj(objective_function, init_beta)


    def set_termination(self, termination = None):
        if termination == None:
            self.termination = SingleObjectiveDefaultTermination(
                    x_tol=1e-8,
                    cv_tol=1e-6,
                    f_tol=1e-4,
                    nth_gen=5,
                    n_last=20,
                    n_max_gen=100,
                    n_max_evals=1500
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
        f = self.run
        #with parallel_backend('multiprocessing'):
        # = Parallel(n_jobs= 6,prefer='processes')(delayed(f)(np.array(x)) for x in x0)
        p = mp.Pool(2)
        result = p.map(f, x0)
        X  = [res.X for res in result]
        F  = [-res.F for res in result]
        return(X, F)

    def set_obj(self, acq, beta):
        if self.max:
            def obj(x):
                return -acq(x, beta)
        else:    
            obj = acq(x , beta)
        def wrapped(x):
            return(obj(x)[0].double().item())
        self.problem = FunctionalProblem(self.xdims,
                                        objs = wrapped,
                                        xl=self.xl,
                                        xu=self.xu
                                        )
