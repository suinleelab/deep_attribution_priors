# Deep Attribution Prior Sample Code

This set of files provides code for training models using the Deep Attribution Priors
framework, introduced in _Learning Deep Attribution Priors Based On Prior Knowledge_.  
In `utils.py` we provide functions to train models both with and without deep attribution
priors.  In `two_moons.py` we demonstrate on the "Two Moons with Nuisance Features" task from
our paper how to bias the training of deep models using the DAPr framework.  We also
include a Jupyter notebook `two_moons.ipynb` to visualize the results of experiments run
using `two_moons.py`.