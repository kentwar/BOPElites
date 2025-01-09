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
                 dim: Optional[int] = 1) -> None:    
        dtype = torch.double
        self.seed = seed
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.hypers_ls = hypers_ls

        if kernel_str == "RBF":
            self.covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=self.dim),
            )

        elif kernel_str == "Matern":
            self.covar_module = ScaleKernel(
                MaternKernel(ard_num_dims=self.dim),
            )

        self.covar_module.base_kernel.lengthscale = self.hypers_ls.to(dtype=dtype)
        self.covar_module.outputscale = torch.tensor(1.,)

        self.generate_function()
        self.find_optimal_value(max = True)
        self.find_optimal_value(max = False)
        self.y_bounds = [self.y_min,self.y_max]

    def generate_function(self):
        print('generating test function with seed: ' +str(self.seed))
        torch.manual_seed(self.seed)
        self.x_base_points = torch.rand(size=(50, self.dim))
        mu = torch.zeros((1,self.x_base_points.shape[0])).to(dtype=dtype)
        C = self.covar_module.forward(self.x_base_points, self.x_base_points).to(dtype=dtype)
        C += torch.eye(C.shape[0]) * 1e-3
        mvn = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=C)
        Z = mvn.rsample(sample_shape=(1, len(mu))).view( C.shape[0], 1)
        invC = torch.inverse(C)
        self.invCZ = torch.matmul(invC, Z)

    def evaluate_true(self, X: torch.tensor) -> torch.tensor:
        
        if X.dim() == 1:
            if self.dim > 1:
                X = X.unsqueeze(0)
            ks = self.covar_module.forward(X, self.x_base_points).to(dtype=dtype)
            out = torch.matmul(ks, self.invCZ)
            return(out)
        
        # Step 1: Identify NaN indices
        nan_mask = torch.isnan(X)
        any_nan = nan_mask.any(dim=1)  # Assuming X is 2D and NaNs should be checked per row
        # Prepare the output tensor filled with NaNs
        out = torch.full((X.shape[0], self.invCZ.shape[1]), float('nan'), dtype=dtype)

        # if any_nan.all():
        #     # If all are NaNs, return the NaN-filled tensor immediately
        #     return out

        # Step 2: Compute output only for non-NaN entries
        # Ensure the computation skips NaN rows
        non_nan_indices = (~any_nan).nonzero(as_tuple=True)[0]
        X_non_nan = X[non_nan_indices]

        # Perform your computation only on non-NaN parts
        ks_non_nan = self.covar_module.forward(X_non_nan, self.x_base_points).to(dtype=dtype)
        out_non_nan = torch.matmul(ks_non_nan, self.invCZ)

        # Step 3: Place computed values back, skipping NaN positions
        out[non_nan_indices] = out_non_nan


        return out
    
    def evaluate_norm(self, X: torch.tensor) -> torch.tensor:
        # Remove nans
        #X = X[~torch.isnan(X)].reshape(-1,self.dim)
        val = self.evaluate_true(X)
        dif = self.y_max-self.y_min
        norm_val = (val - self.y_min)/dif
        return(norm_val)

    def find_optimal_value(self, max = True):
        r"""The global minimum (maximum if negate=True) of the function."""

        bounds = torch.cat(
            [torch.zeros((1, self.dim)), torch.ones((1, self.dim))]
            , dim = 0
        ).to(dtype=dtype)

        X_initial_conditions_raw = torch.rand((100000,  self.dim)).to(dtype=dtype)
        X_initial_conditions_raw = torch.cat([X_initial_conditions_raw, bounds])
        mu_val_initial_conditions_raw = self.evaluate_true(X_initial_conditions_raw).squeeze()
        best_k_indices = torch.argsort(mu_val_initial_conditions_raw, descending=max)[:5]
        X_initial_conditions = X_initial_conditions_raw[best_k_indices, :]

        self.best_val = 9999
        def wrapped_evaluate_true_fun(X, max):
            X = torch.Tensor(X).to(dtype=dtype)
            if X.dim() < self.dim:
                X = X.unsqueeze(0)
            ub_condition = X <= bounds[1] + 1e-4
            lb_condition = X >= bounds[0] - 1e-4
            overall_condition = torch.prod(ub_condition * lb_condition)
            if overall_condition:
                if max:
                    val = -self.evaluate_true(X).to(dtype=dtype).squeeze().detach().numpy()
                else:
                    val = self.evaluate_true(X).to(dtype=dtype).squeeze().detach().numpy()
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