import torch
import numpy as np
def acq_fun(x):
    acqval = torch.sin(x)
    acqval = acqval 
    print(acqval)
    return acqval

X = torch.rand(10, 4, requires_grad=True).double()
acqval = acq_fun(X).sum()
gradf = torch.autograd.grad(acqval, X)[0].contiguous().view(-1)
print(gradf)