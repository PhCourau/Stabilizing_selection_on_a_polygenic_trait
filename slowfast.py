import numpy as np
import matplotlib.pyplot as plt
from simulate_population import generate_pop, Population
#----- Fixed parameters
eta = 1.2
T= 10
N=1000
L=1000 #Must be a multiple of 100 for technical reasons
theta = (.1,.2) # The rate of mutation from 0 to +1 is muN[0]/N per organism
                 # per generation per locus
alphabar = 1/L #Mean effect of an allele (do not change here)
gamma = 1
omega = alphabar * np.sqrt(N/gamma)
ome2 = omega**2/(2*N) #omega_e^{-2}

list_alphaL = np.load("reference_alphaL.npy",allow_pickle=True)
list_alpha = np.repeat(list_alphaL,L//100)/L
np.random.shuffle(list_alpha)


#The starting population must be close to the optimum otherwise the fitness gets
#degenerate when selection is too strong
pop0 = generate_pop((eta,2-eta),N,L)
pop = Population(theta,N,L,population=pop0,alpha=list_alpha)

allele_freq = np.zeros((T*N,10))
traitmeans = np.zeros(T*N)
for t in range(T*N):
  pop.selection_drift_sex(omega,eta,N,L)
  pop.mutation(theta,N,L)
  allele_freq[t] = pop.allele_frequencies()[::(L//10)]
  traitmeans[t] = (np.mean(pop.trait())-eta)
  if ((t*100) %(T*N)) == 0:
    print("Done "+str((t*100)//(T*N))+" per cent")

ax=plt.axes()
ax.plot(allele_freq[-1000:,0],label=r"$X_t^0$")
ax.plot(traitmeans[-1000:]/alphabar,label=r"$L\Delta_t$")
ax.legend()

plt.show()





fig, ax1 = plt.subplots()
t0,t1 = (-400,-300)
list_t = np.arange(t0,t1)/(2*N)
ax1.set_xlabel('t')
ax1.set_ylabel(r'$\Delta_t/(L\omega_e^2)$', color = 'blue')
ax1.plot(list_t,traitmeans[t0:t1]/(L*ome2), color = 'blue')
ax1.tick_params(axis ='y', labelcolor = 'blue')

# Adding Twin Axes

ax2 = ax1.twinx()

ax2.set_ylabel(r'$X_t^\ell$', color = 'red')
ax2.set_ylim((0,1))
ax2.plot(list_t,allele_freq[t0:t1,1], color = 'red')
ax2.tick_params(axis ='y', labelcolor = 'red')

plt.show()

np.save("sims_slowfast.npy",np.append(allele_freq,np.transpose([traitmeans]),axis=1))
