import numpy as np
from simulate_population import generate_pop, Simulate
import os
from multiprocessing import Pool, freeze_support, set_start_method, log_to_stderr

def main():
    #----- Fixed parameters
    eta = 1.2
    T= 1000
    N=50
    L=100
    theta = (.1,.2) # The rate of mutation from 0 to +1 is muN[0]/N per organism
                 # per generation per locus
    nbpoints = 20


    list_alpha = np.load("alpha100.npy",allow_pickle=True)

    list_omem2 = 2*L*np.logspace(np.log10(L),-1,nbpoints) #strengths of selection
    list_omega = np.sqrt(2*N/list_omem2)

    #Output directory
    outdir = "sims_L"+str(L)+"_N"+str(N)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    np.random.seed(0)

    #The starting population must be close to the optimum otherwise the fitness gets
    #degenerate when selection is too strong
    pop0 = generate_pop((eta,2-eta),N,L)

    # Setup iterable of arguments for each worker process.
    args = [(theta,omega,eta,N,L,T,pop0,1,list_alpha,k) for (k,omega) in enumerate(list_omega)]
    # Setup pool of workers, one per simulation. For large number of simulations, consider chunking.
    with Pool(nbpoints) as pool:
        # Use starmap to calculate results.
        results = pool.starmap_async(Simulate, args)

        # Wait for results to come in from all workers.
        results.wait()

    for (k,result) in enumerate(results.get()):
        sim = [list_omega[k], 0, 0, 0, 0]
        sim[1:] = result
        sim = np.array(sim, dtype='object')

        # Writo to file.
        outfile = os.path.join(outdir, str(k)+".npy")
        np.save(outfile, sim)



if __name__ == '__main__':
    # Safety catch if program is "frozen" to produce an executable. See https://docs.python.org/3/library/multiprocessing.html#multiprocessing.freeze_support
    freeze_support()

    # Start the main routine.
    main()
