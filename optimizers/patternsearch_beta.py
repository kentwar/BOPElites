#from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pymoo.termination.default import DefaultSingleObjectiveTermination
#from scipy.optimize import minimize
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem 
# from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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
            # self.termination = SingleObjectiveDefaultTermination(
            #         x_tol=1e-8,
            #         cv_tol=1e-6,
            #         f_tol=1e-4,
            #         nth_gen=5,
            #         n_last=20,
            #         n_max_gen=100,
            #         n_max_evals=1500
            #         ) 
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
        for x in x0:
            X.append(x.numpy())
            F.append(-self.problem.evaluate(x.numpy()))
        return(X, F)

    # def run_many(self, x0, n_workers=4):
    #     with ThreadPoolExecutor(max_workers=n_workers) as executor:
    #         results = list(tqdm(executor.map(self.run, x0), total=len(x0)))

    #     X = []
    #     F = []
    #     for res in results:
    #         X.append(res.X)
    #         F.append(-res.F)

    #     for x in x0:
    #         X.append(x.numpy())
    #         F.append(-self.problem.evaluate(x.numpy()))
    #     return(X, F)

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
