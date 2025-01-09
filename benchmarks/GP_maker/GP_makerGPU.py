import torch
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from scipy.optimize import minimize

dtype = torch.double

class synthetic_GP():
    '''
    A synthetic GP problem
    '''
    def __init__(self, noise_std: Optional[float] = None,
                 negate: Optional[bool] = False,
                 kernel_str: Optional[str] = "Matern",
                 hypers_ls: Optional[float] = 0.5,
                 seed: Optional[int] = 1,
                 bounds = [-1.787691461612756, 2.735910248722252],
                 dim: Optional[int] = 1) -> None:    
        self.tkwargs = {
        "dtype": torch.double,
        "device": "cpu",
        }   
        dtype = torch.double
        self.seed = seed
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.hypers_ls = hypers_ls.to( **self.tkwargs)

        if kernel_str == "RBF":
            self.covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=self.dim).to( **self.tkwargs)#.cuda(), 
            ).to( **self.tkwargs)#.cuda()

        elif kernel_str == "Matern":
            self.covar_module = ScaleKernel(
                MaternKernel(ard_num_dims=self.dim, **self.tkwargs).to( **self.tkwargs), **self.tkwargs
            ).to( **self.tkwargs)#.cuda()

        self.covar_module.base_kernel.lengthscale = self.hypers_ls.to( **self.tkwargs)
        self.covar_module.outputscale = torch.tensor(1.).to( **self.tkwargs)
        self.generate_function()
        #self.find_optimal_value(max = True)
        #self.find_optimal_value(max = False)
        self.y_bounds = [bounds]
        self.y_min = self.y_bounds[0][0]
        self.y_max = self.y_bounds[0][1]

        ## list all tensors with their devices
     


    def generate_function(self):
        print('generating test function with seed: ' +str(self.seed))
        torch.manual_seed(self.seed)
        self.x_base_points = torch.rand(size=(50, self.dim), **self.tkwargs )
        mu = torch.zeros((1,self.x_base_points.shape[0])).to(**self.tkwargs)
        C = self.covar_module.forward(self.x_base_points, self.x_base_points).to(**self.tkwargs)
        C += torch.eye(C.shape[0]).to( **self.tkwargs) * 1e-3
        mvn = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=C)
        Z = mvn.rsample(sample_shape=(1, len(mu))).view( C.shape[0], 1)
        invC = torch.inverse(C)
        self.invCZ = torch.matmul(invC, Z).to(**self.tkwargs)

    def evaluate_true(self, X: torch.tensor) -> torch.tensor:
        ks = self.covar_module.forward(X.to(**self.tkwargs), self.x_base_points).to(**self.tkwargs)
        out = torch.matmul(ks, self.invCZ)
        return out
    
    def evaluate_norm(self, X: torch.tensor) -> torch.tensor:
        
        val = self.evaluate_true(X)
        dif = self.y_max-self.y_min
        norm_val = (val - self.y_min)/dif
        return(norm_val)

    def find_optimal_value(self, max = True):
        r"""The global minimum (maximum if negate=True) of the function."""

        bounds = torch.cat(
            [torch.zeros((1, self.dim),**self.tkwargs), torch.ones((1, self.dim),**self.tkwargs)]
            , dim = 0
        ).to(**self.tkwargs)

        X_initial_conditions_raw = torch.rand((100000,  self.dim)).to(**self.tkwargs)
        X_initial_conditions_raw = torch.cat([X_initial_conditions_raw, bounds])
        mu_val_initial_conditions_raw = self.evaluate_true(X_initial_conditions_raw).squeeze()
        best_k_indices = torch.argsort(mu_val_initial_conditions_raw, descending=max)[:5]
        X_initial_conditions = X_initial_conditions_raw[best_k_indices, :]

        self.best_val = 9999
        def wrapped_evaluate_true_fun(X, max):
            X = torch.Tensor(X).to(**self.tkwargs)
            ub_condition = X <= bounds[1] + 1e-4
            lb_condition = X >= bounds[0] - 1e-4
            overall_condition = torch.prod(ub_condition * lb_condition)
            if overall_condition:
                if max:
                    val = -self.evaluate_true(X).to(**self.tkwargs).squeeze().detach().numpy()
                else:
                    val = self.evaluate_true(X).to(**self.tkwargs).squeeze().detach().numpy()
                if val < self.best_val:
                    self.best_val = val
                return val
            else:
                return 999


        res = [minimize(wrapped_evaluate_true_fun, args = max, x0=x0, method='nelder-mead', tol=1e-9) for x0 in X_initial_conditions]
        if max:
            self.x_max = res[0]["x"]
            self.y_max = -res[0]["fun"]
            return(self.x_max, self.y_max)
        else:
            self.x_min = res[0]["x"]
            self.y_min = res[0]["fun"]
            return(self.x_min, self.y_min)
        


    def plotGP(self):
        try:
            optmax = [self.x_max,self.y_max]
            optmin = [self.x_min,self.y_min]
        except:
            optmax = self.optimize_optimal_value(max = True)
            optmax = self.optimize_optimal_value(max = False)
        if self.dim > 2 :
            print('cannot print more than 2 dimensions')
        if self.dim == 1:
            X_plot = torch.rand(100000,self.dim).sort(dim=0).values
            fval = myGP.evaluate_true(X_plot).detach()
            plt.plot(X_plot, fval/self.y_max, color = 'black')
            plt.scatter(optmax[0],optmax[1]/self.y_max ,color = 'red')
            plt.scatter(optmin[0],optmin[1]/self.y_max ,color = 'green')
            plt.show()
        if self.dim == 2:
            X_plot = torch.rand(100000,self.dim)
            fval = myGP.evaluate_true(X_plot).detach()
            plt.scatter(X_plot[:,0], X_plot[:,1], c=fval ,s = 2.5)
            plt.scatter(optmax[0][0],optmax[0][1], color = 'red',marker='*')
            plt.scatter(optmin[0][0],optmin[0][1], color = 'green',marker='*')
            plt.show()


if __name__ == '__main__':

    myGP = synthetic_GP(seed = 25, 
                        dim = 1,
                        hypers_ls = 0.5)
    myGP.evaluate_true(torch.tensor([0.3,0.2]))
    myGP.plotGP()
    print(myGP.y_bounds)