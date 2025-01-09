import numpy as np
import sys, os
# Get the directory of the current file (__file__ is the pathname of the file from which the module was loaded)
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current file's directory
parent_directory = os.path.dirname(current_file_directory)

# Add the parent directory to sys.path
sys.path.insert(0, parent_directory)

#Botorch imports
from botorch.models.transforms import Standardize
from botorch import fit_gpytorch_mll, gen_candidates_torch
from botorch.acquisition import qKnowledgeGradient, qExpectedImprovement
from botorch.models.gp_regression import FixedNoiseGP
from botorch.sampling import SobolQMCNormalSampler
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from botorch.acquisition import qExpectedImprovement as qEI

from ribs.archives import GridArchive
from ribs.emitters import  GradientArborescenceEmitter, EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
import torch



from benchmarks.robotarm import RobotArm
from benchmarks.mishra_bird_function import Mishra_bird_function

domain = Mishra_bird_function(10)

def sphere(solution_batch):
    """Sphere function evaluation and measures for a batch of solutions.

    Args:
        solution_batch (np.ndarray): (batch_size, dim) batch of solutions.
    Returns:
        objective_batch (np.ndarray): (batch_size,) batch of objectives.
        measures_batch (np.ndarray): (batch_size, 2) batch of measures.
    """
    dim = solution_batch.shape[1]

    # Shift the Sphere function so that the optimal value is at x_i = 2.048.
    sphere_shift = 5.12 * 0.4

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 0.0
    worst_obj = (-5.12 - sphere_shift)**2 * dim
    raw_obj = np.sum(np.square(solution_batch - sphere_shift), axis=1)
    objective_batch = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # Calculate measures.
    clipped = solution_batch.copy()
    clip_mask = (clipped < -5.12) | (clipped > 5.12)
    clipped[clip_mask] = 5.12 / clipped[clip_mask]
    measures_batch = np.concatenate(
        (
            np.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
        ),
        axis=1,
    )

    return objective_batch, measures_batch

archive = GridArchive(
    solution_dim=2,
    dims=[10, 10],
    learning_rate=0.01,
    threshold_min=0.0,
    ranges=[(0, 10), (0,6)],
)

result_archive = GridArchive(
    solution_dim=2,
    dims=[10, 10],
    ranges=[(0, 10), (0,6)],
)

def obj_feat_bat(x):
    obs_x = torch.tensor(x).reshape(-1,2)
    fit , b  = domain.torch_fitness(obs_x.detach())
    return fit, b





x_random = np.random.rand(1000,2)
fit, b = domain.torch_fitness(torch.tensor(x_random))
#archive.add(x_random, fit.detach().numpy(), b.detach().numpy())
result_archive.add(x_random, fit.detach().numpy(), b.detach().numpy())

fit = fit.unsqueeze(-1)
train_y = torch.clone(fit).double().requires_grad_(True)
train_x = torch.tensor(x_random, dtype= torch.double).requires_grad_(True)
outcome_transform = Standardize(m=1)
train_yvar = torch.full_like(train_y, 1e-6, dtype=torch.double)
# train_x = train_x.detach()
# train_y = train_y.detach().double()
# train_yvar = train_yvar.detach().double()
model = FixedNoiseGP(train_x.detach(), train_y.detach(), train_yvar.detach(),outcome_transform=outcome_transform)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

emitters = [
    GradientArborescenceEmitter(
        archive,
        x0=np.array([0.3]*2),
        sigma0=0.01,
        ranker="imp",
        lr = 0.002,
        selection_rule="mu",
        restart_rule="basic",
        batch_size=36,
        #grad_opt = 'adam'
    ) for _ in range(10)
]
# emitters = [
#     EvolutionStrategyEmitter(
#         archive=archive,
#         x0=np.array([0.3]*2),
#         sigma0=1.0,  # Initial step size.
#         ranker="2imp",
#         batch_size=30,  # If we do not specify a batch size, the emitter will
#                         # automatically use a batch size equal to the default
#                         # population size of CMA-ES.
#     ) for _ in range(5)  # Create 5 separate emitters.
# ]


scheduler = Scheduler(archive, emitters, result_archive=result_archive)

class obj_feat:
    def __init__(self, model):
        self.tracker = 0 
        self.model = model
    def __call__(self, x):
        self.tracker += x.shape[0]
        obs_x = torch.tensor(x.reshape(-1,2), requires_grad=True)   
        fit , b  = domain.torch_fitness(obs_x)
        fit.sum().backward(retain_graph=True)
        grad_fit = obs_x.grad.clone()
        b1 = b[:,0]
        b1.sum().backward(retain_graph=True)
        grad_b1 = obs_x.grad
        #obs_x.grad.zero_()
        b2 = b[:,1]
        b2.sum().backward(retain_graph=True)
        grad_b2 = obs_x.grad
        grad_b = torch.stack([grad_b1, grad_b2]).transpose(0,1)
        #fit = self.model.posterior(obs_x).mean

        return fit.detach().squeeze().numpy(), grad_fit.squeeze().numpy(), b.detach().numpy(), grad_b.numpy()

    def evaluate(self, x):
        self.tracker += x.shape[0]
        obs_x = torch.tensor(x).reshape(-1,2)
        fit , b  = domain.torch_fitness(obs_x.detach())
        #fit = self.model.posterior(obs_x).mean.detach()
        return fit.detach().squeeze(), b.detach().squeeze() 

obj_feat_bat_grad = obj_feat(model = model)

for itr in range(5000):

    solution_batch = scheduler.ask_dqd()
    (objective_batch, objective_grad_batch, measures_batch, measures_grad_batch) = obj_feat_bat_grad(solution_batch)
    objective_grad_batch = np.expand_dims(objective_grad_batch, axis=1)
    jacobian_batch = np.concatenate((objective_grad_batch, measures_grad_batch), axis=1)
    scheduler.tell_dqd(objective_batch, measures_batch, jacobian_batch)
    
    solutions = scheduler.ask()

    fit, b = obj_feat_bat_grad.evaluate(solutions)

    objective_batch = fit.numpy()

    # get behaviours.
    measures_batch = b.detach().numpy()

    scheduler.tell(objective_batch, measures_batch)

import matplotlib.pyplot as plt
from ribs.visualize import grid_archive_heatmap

grid_archive_heatmap(result_archive)
plt.show()

print(f'{result_archive.stats}')
print(f'finished after {obj_feat_bat_grad.tracker} evaluations')

