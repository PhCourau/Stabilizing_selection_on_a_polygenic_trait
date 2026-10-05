# Stabilizing selection on a polygenic trait
Philibert Courau https://www.normalesup.org/~pcourau/

This folder contains all of the code used to generate the simulations and figures used in the preprint:
  https://www.biorxiv.org/content/10.64898/2026.02.23.706325v1
except for Figure 1. The code is currently not very readable, sorry about that, but it should be running without problem. ALL PROGRAMS SHOULD BE RUN FROM THIS FOLDER.

## Files
The file simulate_population.py contains the functions necessary to run the simulations. The various files save_simulations each run simulations in a specific parameter setting and save them to a specific folder. Each of these files should be run as it is, except save_simulations.py which should be run three times, with line 10 reading N=50, N=100 and N=1000. Running such a file on my laptop takes between a few hours and three days. The file alpha50.npy, alpha100.npy, and alpha1000.npy contain the reference parameters for the simulations.

The file Figures.ipynb is a Python notebook which contains all the code necessary to generate all figures from the data, once the save_simulations files have been run.
