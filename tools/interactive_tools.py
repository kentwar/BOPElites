
import torch
import numpy as np
import itertools as it
import math

def dist(p , q):
    d = math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))
    return(d)

def select_zoom_region(algorithm):
    # Generate random points
    points = torch.rand(algorithm.domain.fdims)
    
    # Generate random rectangle shape
    rectangle_shape = torch.rand(algorithm.domain.fdims) / 4
    
    # Calculate the boundaries of the rectangle
    rectangle = torch.stack([points - rectangle_shape, points + rectangle_shape], dim=1)
    
    # Clamp the rectangle between 0 and 1
    rectangle = torch.clamp(rectangle, 0, 1) 
    
    # Calculate the bottom left corner of the rectangle
    bottom_left = rectangle[:, 0]
    
    # Calculate the height and width of the rectangle
    # height = rectangle[1, 1] - rectangle[1, 0]
    # width = rectangle[0, 1] - rectangle[0, 0]
    max_x = rectangle[0, 1]
    max_y = rectangle[1, 1]
    
    # Return the bottom left corner, max_x  and max_y of the rectangle
    return bottom_left, max_x, max_y

def rescale_rectangle(rectangle, feature_resolution=[40, 40]):
    # Calculate the bottom left corner of the rectangle
    fr = torch.tensor(feature_resolution)
    bottom_left = rectangle[0] * fr
    
    # Calculate the height and width of the rectangle
    # width = rectangle[1] * fr[0]
    # exceeds_limit = torch.max(width + bottom_left[0] - fr[0], torch.tensor(0.0) )
    # width = width - exceeds_limit
    # height = rectangle[2] * torch.tensor(fr[1])
    # exceeds_limit = torch.max(height + bottom_left[1] - fr[1], torch.tensor(0.0) )
    max_x = rectangle[1] * fr[0]
    max_y = rectangle[2] * fr[1]
    # Return the bottom left corner, height and width of the rectangle
    return (bottom_left, max_x, max_y )


def first_multiple(tensor):
    # Extract the two floats from the tensor
    x, y = tensor

    # Set the initial multiple to 1
    multiple = 1

    # Loop until the sum of the two numbers multiplied by the multiple is greater than 100
    while multiple * x * multiple * y <= 100:
        multiple += 1
    
    # Return the first multiple that makes the sum of the two numbers greater than 100
    return multiple -1

def zoom_edges(edges, rectangle):
    f2lims = [rectangle[0][0], rectangle[0][0] + rectangle[1]]
    f1lims = [rectangle[0][1], rectangle[0][1] + rectangle[2]]
    # find the closest edges to the limits
    lowerf2lims = np.argmin(np.abs(edges[1] - f2lims[0].item()))
    upperf2lims = np.argmin(np.abs(edges[1] - f2lims[1].item())) +1
    f2lims = edges[1][lowerf2lims: upperf2lims]
    lowerf1lims = np.argmin(np.abs(edges[0] - f1lims[0].item()))
    upperf1lims = np.argmin(np.abs(edges[0] - f1lims[1].item())) +1
    f1lims = edges[0][lowerf1lims: upperf1lims]
    edges = [f1lims, f2lims]
    return(edges)


def MC_sample(z , x, model):
    # takes random z's defined on the univariate gaussian 
    # and returns the MC sample of the posterior
    # at those z values
    mean, var = model.posterior(x).mean, model.posterior(x).variance
    predictions = z*var.sqrt() + mean
    return predictions

def MC_position(z , x , models):
    # takes random z's defined on the univariate gaussian 
    # and returns the MC sample of the posterior
    # at those z values
    beh1 = MC_sample(z[0:int(z.shape[0]/2)], x, models[0])
    beh2 = MC_sample(z[int(z.shape[0]/2):], x, models[1])
    positions = list(it.product(beh1[0],beh2[0]))
    positions = torch.tensor(positions).squeeze(0)
    return(positions)
    #return torch.stack(predictions, axis = -1).squeeze(0).squeeze(1)

def MC_acq_func( x, fitz, featz, fit_model, feat_models, alpha, behdist, bestval , alg):
    # calculates a Montecarlo estimate of fit - alpha*distance
    # num_descriptor vals
    # We hold the z values constant so that the acquisition function is
    # deterministic

    mcfit = MC_sample(fitz, x, fit_model)[0] *alg.fitness.std() + alg.fitness.mean()
    positions = MC_position(featz, x, feat_models)
    distance = behdist(positions)
    acq_val = (mcfit - alpha*distance - bestval)
    acq_val_EI = torch.where(acq_val > 0, acq_val, torch.zeros_like(acq_val))
    acq_val = acq_val_EI.mean()
    return acq_val


def predict_position(models, x):
    sample = [model.posterior(x).sample() for model in models]
    return torch.stack(sample, axis = -1).squeeze(0).squeeze(1)

# calculate euclidean distance in n dimensions
def euclidean_distance(x, y):
    return torch.norm(x - y, dim = -1)

def euclidean_distance_vectorized(x, y):
    return torch.sqrt(torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=-1))

def gen_distance_func(target):
    def distance(position):
        #position = domain.BOtorch_feature_fun(x)
        if len(position.shape) == 1:
            position = position.unsqueeze(0)
        eucdist = torch.tensor([dist(p, target) for p in position])
        return eucdist
    return distance

# behaviour_distance = gen_distance_func(target)