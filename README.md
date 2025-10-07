# Stabilizing selection on a polygenic trait
Philibert Courau https://www.normalesup.org/~pcourau/

This folder contains all of the code used to generate the simulations and figures used in my article, except for Figure 1. The code is currently not very readable, sorry about that, but it should be running without problem. ALL PROGRAMS SHOULD BE RUN FROM THIS FOLDER.


## Generating the figures
To generate Figure 2, first you must create an empty folder named sims_L100_N100 by running the command
      mkdir sims_L100_N100
Then you must go into save_simulations.py, on line 7 replace N=500 with N=100 and run save_simulations.py (this fills the folder sims_L100_N100).
Finally, run histz.py.

To generate Figure 3, first you must create 2 empty folders named sims_L100_N500 and sims_L100_N50 by running the command
      mkdir sims_L100_N500
      mkdir sims_L100_N50
Then you must run the file save_simulations.py to fill the folder sims_L100_N500 with the appropriate simulations (on my laptop, this runs for about 10 days). Then go into save_simulations.py, on line 7 replace N=500 with N=50 and rerun the file (this will fill the folder sims_L100_N50).
Finally, running the file plot_th_vs_sim.py will plot Figure 2 and Figure S5.

To generate Figure 4, run the file 

To generate Figure 5, first run
        mkdir contoursims/sims_L100_N500
Then run contoursims/save_simulations.py to generate the simulations, and contoursims/contourplot.py to obtain the plot.

To generate Figure S1 (illustrating the selection coefficient as a regression on logfitness), simply run fitness_regression.py.

To generate Figure S2 (illustrating time-averaging), run the file slowfast.py.

To generate Figure S3 (illustrating the breakdown for small mutation rates), first run
        mkdir breakdown_smallmut/sims_L100_N500
Then run breakdown_smallmut/save_simulations.py and breakdown_smallmut/save_simulations2.py (in parallel, this goes faster). This generates the simulations to fill up breakdown_smallmut/sims_L100_N500/. Then run breakdown_smallmut/plot_th_vs_sim.py 

To generate Figure S4 (illustrating the breakdown when the distribution of alpha has heavy tails), first run
        mkdir pareto/sims_L100
Then run pareto/save_simulations.py followed by pareto/plot_th_vs_sim.py

## Concerning the other files
The file simulate_population.py contains the code to simulate the individual-based model.

The file reference_alphaL.npy contains a vector of variables of size 100, generated with law exponential(1). Except for pareto/save_simulations, all simulations use this distribution, renormalized to have sum 1, as alpha.

The file solve_fixed_point.py solves the fixed point equation for $Delta^*$ numerically using a golden-section search algorithm.
