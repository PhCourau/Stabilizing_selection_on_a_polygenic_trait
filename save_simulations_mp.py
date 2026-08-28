import numpy as np
from simulate_population import generate_pop, Simulate
from time import time
import os
from multiprocessing import Pool, freeze_support, set_start_method

def main():
    #----- Fixed parameters
    eta = 1.2
    T= 200
    L=100 #Must be a multiple of 100 for technical reasons
    N=500

    outdir_root = 'contoursims'
    outdir = "sims_L"+str(L)+"_N"+str(N)
    outdir_target = os.path.join(outdir_root, outdir)

    if not os.path.exists(outdir_target):
        os.makedirs(outdir_target, exist_ok=True)

    theta = (.5,.5) # The rate of mutation from 0 to +1 is muN[0]/N per organism
                    # per generation per locus
    nbpoints = 4

    alphabar = 1/L #Mean effect of an allele (do not change here)

    list_alphaL = np.load("reference_alphaL.npy",allow_pickle=True)
    list_alpha = np.repeat(list_alphaL,L//100)/L
    np.random.shuffle(list_alpha)

    # list_gamma = np.logspace(0,-np.log10(L)-1,nbpoints)
    list_gamma = np.array([1e-10,5e-2,1e-1,1])
    list_omega = alphabar * np.sqrt(N/list_gamma)


    #The starting population must be close to the optimum otherwise the fitness gets
    #degenerate when selection is too strong
    pop0 = generate_pop((eta,2-eta),N,L)

    # Setup iterable of arguments for each worker process.
    args = [(theta,omega,eta,N,L,T,pop0,1,list_alpha) for omega in list_omega]

    # Setup pool of workers, one per simulation. For large number of simulations, consider chunking.
    with Pool(nbpoints) as pool:
        # Use starmap to calculate results.
        results = pool.starmap_async(Simulate, args)

        # Wait for results to come in from all workers.
        results.wait()

    # Iterate over results objects to retrieve Simulation() returns.
    for (k,result) in enumerate(results.get()):

        sim = [list_omega[k], 0, 0, 0, 0]
        sim[1:] = result
        sim = np.array(sim, dtype='object')

        # Writo to file.
        outfile = os.path.join(outdir_target, str(k)+".npy")
        np.save(outfile, sim)

if __name__ == '__main__':
    # Safety catch if program is "frozen" to produce an executable. See https://docs.python.org/3/library/multiprocessing.html#multiprocessing.freeze_support
    freeze_support()

    # Start the main routine.
    main()
