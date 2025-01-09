from pymoo.util.termination.default import SingleObjectiveDefaultTermination
#from scipy.optimize import minimize
import numpy as np
import sys
from tqdm import tqdm
from scipy.optimize import minimize
import torch

class Scipy_optimizer():

    def __init__(self, objective_function, domain, beta = 0, max = True, termination = None):
        self.algorithm = None


        ## Define the problem 
        self.domain = domain
        self.xl = np.array(domain.Xconstraints)[:,0]
        self.xu = np.array(domain.Xconstraints)[:,1]
        self.xdims = domain.xdims
        self.max = max
        self.set_obj(objective_function)


    def optimize_acqf_and_get_observation(acq_func):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.domain.Xconstraints,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        # observe new values 
        new_x = candidates.detach()
        return new_x

    def optimize(self, x0, fun):

        res = minimize(
            fun=fun,
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            bounds=self.domain.Xconstraints)
        return(res)

    def run(self, x0):
        x0 = np.array(x0)
        res = self.optimize(x0,self.problem,)
        return(res)

    def run_many(self, x0):
        X = []
        F = []
        for x in tqdm(x0):
            x = np.array(x)
            res = self.run(x)
            X.append(res.x)
            F.append(-res.fun)
        return(X, F)

    def set_obj(self, acq , beta = None):
        if self.max:
            def obj(x):
                val = -acq(x)
                val.backward()
                return (-acq(x), x.grad)
        else:    
            def obj(x):
                val = acq(x)
                val.backward()
                print('min')
                return (val, x.grad)
        def wrapped(x):
            with torch.enable_grad():
                x = torch.tensor(x, requires_grad=True, dtype=torch.double)

                return(obj(x))
            
        self.problem = wrapped


