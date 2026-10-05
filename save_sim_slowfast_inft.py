import numpy as np
from simulate_population import *
from time import time

# Parameters
eta = 1.2
T= 5
N=5000
L=1000 #Must be a multiple of 100 for technical reasons
theta = (.1,.2) # The rate of mutation from 0 to +1 is muN[0]/N per organism
                 # per generation per locus
omem2 = 2*L**2 #omega_e^{-2}
omega = np.sqrt(2*N/omem2)

list_alpha = np.load("alpha1000.npy",allow_pickle=True)

# Simulation
np.random.seed(0)

np.random.shuffle(list_alpha)
#The starting population must be close to the optimum otherwise the fitness gets
#degenerate when selection is too strong
pop0 = generate_pop((eta,2-eta),N,L)
pop = Population(theta,N,L,alpha=list_alpha,population=pop0)

allele_freq = np.zeros((T*N,10))
traitmeans = np.zeros(T*N)
for t in range(T*N):
    pop.selection_drift_sex(omega,eta,N,L)
    pop.mutation(theta,N,L)
    allele_freq[t] = pop.allele_frequencies()[::(L//10)]
    traitmeans[t] = (np.mean(pop.trait())-eta)
    if ((t*100) %(T*N)) == 0:
        print("Done "+str((t*100)//(T*N))+" per cent")

np.save("sim_slowfast",np.array([[L,N,theta,omega,eta,list_alpha],allele_freq[-2000:],traitmeans[-2000:],pop.population],dtype="object"))
