from __future__ import annotations

import math
from abc import ABC, abstractmethod
from math import pi
from typing import Optional

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.test_functions.base import (
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)
from botorch.test_functions.synthetic import Branin, Levy
from botorch.utils.sampling import sample_hypersphere, sample_simplex
from botorch.utils.transforms import unnormalize
from scipy.special import gamma
from torch import Tensor
from torch.distributions import MultivariateNormal


class MOBOQD(MultiObjectiveTestProblem):
    r"""Two objective problem composed of a fitness function and a distance 
    in behaviour space.

    Fitness:

    The fitness can take the form of any function that returns a single valuation


    Distance:

    The distance is the euclidean distance between the estimated point and the 
    target.

    """

    dim = 10
    num_objectives = 2
    _bounds = [(0.0, 1.0), (0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0)]
    _ref_point = [0.0, 0.0]
    _max_hv = 1  # this is approximated using NSGA-II

    def __init__(self, fitness_fun, distance_fun, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)
        self.fitness_fun = fitness_fun
        self.distance_fun = distance_fun

    def evaluate_true(self, X: Tensor) -> Tensor:

        f1 = self.fitness_fun(X)
        f2 = self.distance_fun(X)
        return torch.stack([f1, f2], dim=-1, )