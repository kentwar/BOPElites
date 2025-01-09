from pymoo.util.termination.default import SingleObjectiveDefaultTermination
#from scipy.optimize import minimize
import numpy as np
import sys
from tqdm import tqdm
from scipy.optimize import minimize
import torch
from scipy.optimize import Bounds

class LBFGSB_optimizer():

    def __init__(self, objective_function, domain, init_beta = 0, max = True, termination = None):
        self.algorithm = None


        ## Define the problem 
        self.domain = domain
        self.xl = np.array(domain.Xconstraints)[:,0]
        self.xu = np.array(domain.Xconstraints)[:,1]
        self.xdims = domain.xdims
        self.max = max
        self.set_obj(objective_function)

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
        initial_x = x0.requires_grad_().double()
        shapeX = x0.shape
        bounds = self.make_scipy_bounds(initial_x)
        ## Arrify
        x0 = self._arrayify(x0.view(-1))

        def f(x):
            X = (
                torch.from_numpy(x)
                .to(initial_x).double()
                .view(shapeX)
                .contiguous()
                .requires_grad_(True)
            )
            with torch.enable_grad():
                acqval = self.problem(X).sum()
            # compute gradient w.r.t. the inputs (does not accumulate in leaves)
            gradf = self._arrayify(torch.autograd.grad(acqval, X)[0].contiguous().view(-1))
            fval = acqval.item()

            return fval, gradf

        #init_EI = torch.tensor([-self.problem(torch.tensor(x).reshape(-1,4)).detach().item() for x in x0.reshape(-1,4)])
        #print(init_EI.reshape(-1))  

        res = minimize(
            f,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options = {'disp': True})
        optim_x = torch.from_numpy(res.x).to(initial_x).view(shapeX).contiguous()
        optim_EI = torch.tensor([-self.problem(x.reshape(-1,self.domain.xdims)).detach().item() for x in optim_x])        #print(optim_EI.reshape(-1))

        return(optim_x, optim_EI)


    def set_obj(self, acq , beta = None):
        if self.max:
            def obj(x):
                val = -acq(x)
                #val.backward()
                return (val)#, x.grad)
        else:    
            def obj(x):
                val = acq(x)
                #val.backward()
                #print('min')
                return (val)#, x.grad)
        def wrapped(x):
            return(obj(x))
            
        self.problem = wrapped


    def _arrayify(self, X: torch.Tensor) -> np.ndarray:
        r"""Convert a torch.Tensor (any dtype or device) to a numpy (double) array.

        Args:
            X: The input tensor.

        Returns:
            A numpy array of double dtype with the same shape and data as `X`.
        """
        return X.cpu().detach().contiguous().double().clone().numpy()

    def make_scipy_bounds(self, X):

        def _expand(bounds, X):
            if not torch.is_tensor(bounds):
                bounds = torch.tensor(bounds)
            ebounds = bounds.expand_as(X)
            return self._arrayify(ebounds).flatten()
        lb = self.domain.Xconstraints[:,0]
        ub = self.domain.Xconstraints[:,1]
        lb = _expand(bounds=lb, X=X)
        ub = _expand(bounds=ub, X=X)
        return Bounds(lb=lb, ub=ub, keep_feasible=True)
