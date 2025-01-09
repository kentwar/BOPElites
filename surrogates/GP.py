'''
This implements building a GP model from the botorch library.
'''

import numpy as np
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model

def buildGP_from_XY(x_train, y_train, std = True):
    '''
    Trains a GP from the observation set of the BOP-Elites codebase

    n_observations = [x*d , fitness, features*m]*n
    '''   
    if std:
        model = CreateModel(x_train, standardize(y_train))
    else:
        model = CreateModel(x_train, y_train)
    return(model)

def CreateModel(xtrain, ytrain):
    '''
    Creates and trains a GpyTorch GP model.
    '''
    model = FixedNoiseGP(xtrain, ytrain, train_Yvar = torch.full_like(ytrain, 1e-6))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll);
    return(model)