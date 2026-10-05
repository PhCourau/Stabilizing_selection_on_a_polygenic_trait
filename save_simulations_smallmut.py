#MUST BE RUN FROM THE PARENT DIRECTORY
import os

import numpy as np
from simulate_population import generate_pop, Simulate
from multiprocessing import Pool, freeze_support, set_start_method


def main():
    #----- Fixed parameters
    eta = 1.2
    T= 500
    N=500
    L=100 #Must be a multiple of 100 for technical reasons
    omem2 = 2*L**2/10

    nbpoints = 6

    list_alpha = np.load("alpha100.npy",allow_pickle=True)

    omega = np.sqrt(2*N/omem2)


    list_theta = np.logspace(-np.log10(10*L),0,nbpoints) # The rate of mutation from 0 to +1 is theta[0]/2N per organism
                 # per generation per locus
    #The starting population must be close to the optimum otherwise the fitness gets
    #degenerate when selection is too strong
    np.random.seed(0)
    pop0 = generate_pop((eta,2-eta),N,L)

    #Target file
    outdir_root = 'breakdown_smallmut'
    outdir = "sims_L"+str(L)+"_N"+str(N)
    outdir_target = os.path.join(outdir_root, outdir)

    if not os.path.exists(outdir_target):
        os.makedirs(outdir_target, exist_ok=True)

    # Setup iterable of arguments for each worker process.
    list_pairtheta = []
    for theta1 in list_theta:
        for theta2 in list_theta:
            list_pairtheta.append((theta1,theta2))
    args = [(theta,omega,eta,N,L,T,pop0,1,list_alpha,k+int(theta[0]//theta[1])) for (k,theta) in enumerate(list_pairtheta)]

    # Setup pool of workers, one per simulation. For large number of simulations, consider chunking.
    with Pool(nbpoints) as pool:
        # Use starmap to calculate results.
        results = pool.starmap_async(Simulate, args)

        # Wait for results to come in from all workers.
        results.wait()

    # Iterate over results objects to retrieve Simulation() returns.
    for (k,result) in enumerate(results.get()):

        sim = [omega, 0, 0, 0, 0]
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
