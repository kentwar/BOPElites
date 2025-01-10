# BOP-Elites-2024
A Working repository for BOP-Elites for collaborative development.

BOP-Elites is a true Bayesian Optimisation algorithm for Quality-Diversity optimisation. 
See the paper here https://ieeexplore.ieee.org/document/10472301 

# Quality-Diversity
BOP-Elites is a Quality Diversity algorithm. Learn more here
https://quality-diversity.github.io/

# Limitations and Use Case
## Highly sample efficient
BOP-Elites is sample efficient, using Bayesian Optimisation to choose where to sample to efficiently balance exploitation and exploration through its acquisition function.
## Lower input dimensionality
BOP-Elites utilises Bayesian Optimisation (an adaptation of Expected Improvement) using a Gaussian Process and is therefore only functional on problems with relative low dimensions (for QD usecases).  
## 1-2d feature space
BOP-Elites is only programmed to work with 1/2 d feature spaces. While the concept can be extrapolated to (slightly) higher dimensions the number of regions would explode making the probability function for region membership unworkable. BOP-Elites uses a uniformly discretised grid.


This is repurposed research code, made available for comparison for scientific purposes. Things may not work as intended because this was developed on linux, and I am now primarily using windows and there were some strange migration issues. 
so please contact me so we can make the work together 

### How to use:

There are executable self contained experiments in the tutorials folder. This is intended to show basic usage

## Requirements:

the requirements.yml allows for installation with conda. It is currently defined for windows install and may need tampering with to get it working with other OS

## Known Bugs

- There aren't many, which is only to say nobody has tried to use this yet, so we havent found them. 

1. the Parsec domain was setup using Xfoil which is old code and a bit fiddly. It didn't compile on my windows machine, so I zipped it up and have stored it in the pyAFT.zip file. In theory it should just unzip and run, but this may be a bit hopeful.

2. I expect the following will crop up: 
- Tensor shape for GPYTorch models. - Newer GPYtorch models seem to require a different shaped tensor, I noticed this broke the synthetic function maker so I have ficed it by unsqueezing the tensors. If you get a weird shape error this may be the reason.




BOP-Elites has been used in several applications:
https://dl.acm.org/doi/abs/10.1145/3583131.3590486
https://proceedings.mlr.press/v188/schneider22a.html
