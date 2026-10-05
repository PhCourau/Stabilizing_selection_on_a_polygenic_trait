import numpy as np
from simulate_population import generate_pop, Simulate
import os
from multiprocessing import Pool, freeze_support, set_start_method

def main():
    #----- Fixed parameters
    eta = 1.2
    T= 10000
    list_N=(10,30,50)
    L=200 #Must be a multiple of 100 for technical reasons
    list_theta = [(.05,.1),(.06,.12),(.07,.14),(.08,.16),(.09,.18),(.1,.2)] # The rate of mutation from 0 to +1 is muN[0]/N per organism
                 # per generation per locus
    nbpoints = len(list_theta)

    list_alpha = np.load("alpha200.npy",allow_pickle=True)

    omem2 = 2*L #Weak selection

    np.random.seed(0)
    for N in list_N:
        omega = np.sqrt(2*N/omem2) #Weak selection

        #The starting population must be close to the optimum otherwise the fitness gets
        #degenerate when selection is too strong
        pop0 = generate_pop((eta,2-eta),N,L)
    
        outdir_root = 'hitch_hike'
        outdir = "sims_L"+str(L)+"_N"+str(N)
        outdir_target = os.path.join(outdir_root, outdir)
    
        if not os.path.exists(outdir_target):
            os.makedirs(outdir_target, exist_ok=True)
    
        # Setup iterable of arguments for each worker process.
        args = [(theta,omega,eta,N,L,T,pop0,1,list_alpha,k) for (k,theta) in enumerate(list_theta)]
    
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

