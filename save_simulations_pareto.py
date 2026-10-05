# RUN THIS FROM ITS PARENT DIRECTORY
import numpy as np
from simulate_population import Simulate, generate_pop
import os
from multiprocessing import Pool, freeze_support, set_start_method

def main():
    #----- Fixed parameters
    eta = 1.2
    T= 500
    N=500
    L=100 #Must be a multiple of 100 for technical reasons
    theta = (.1,.2) # The rate of mutation from 0 to +1 is muN[0]/N per organism
                 # per generation per locus
    omem2 = L**2/5
    omega = np.sqrt(2*N/omem2)

    nbpoints = 11

    np.random.seed(0)
    pop0 = generate_pop((eta,2-eta),N,L)


    #Target directory
    outdir_root = 'pareto'
    outdir = "sims_L"+str(L)+"_N"+str(N)
    outdir_target = os.path.join(outdir_root, outdir)

    if not os.path.exists(outdir_target):
        os.makedirs(outdir_target, exist_ok=True)

    # Setup iterable of arguments for each worker process.
    list_list_alphas = np.zeros((nbpoints,L))
    for (k,param) in enumerate(np.linspace(1,4,nbpoints)):
        list_alpha = np.random.pareto(param,size=L)
        list_alpha = list_alpha/np.sum(list_alpha)
        list_list_alphas[k] = list_alpha
    args = [(theta,omega,eta,N,L,T,pop0,1,list_alpha,k) for (k,list_alpha) in enumerate(list_list_alphas)]

    # Setup pool of workers, one per simulation. For large number of simulations, consider chunking.
    with Pool(nbpoints) as pool:
        print("Starting pool execution...")
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
