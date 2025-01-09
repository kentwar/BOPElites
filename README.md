# BOP-Elites-2024
A Working repository for BOP-Elites for collaborative development.

This is repurposed research code, made available for comparison for scientific purposes. Things may not work as intended because this was developed on linux, and I am now primarily using windows and there were some strange migration issues. 
so please contact me so we can make the work together 

### How to use:

There are executable self contained experiments in the tutorials folder. This is intended to show basic usage

## Requirements:

the requirements.yml allows for installation with conda. It is currently defined for windows install and may need tampering with to get it working with other OS

## Known Bugs

- There aren't any, which is only to say nobody has tried to use this yet, so we havent found them. 

I expect the following will crop up: 
- Tensor shape for GPYTorch models. - Newer GPYtorch models seem to require a different shaped tensor, I noticed this broke the synthetic function maker so I have ficed it by unsqueezing the tensors. If you get a weird shape error this may be the reason. 
