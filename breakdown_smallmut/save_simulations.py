#MUST BE RUN FROM THE PARENT DIRECTORY
import numpy as np
from simulate_population import generate_pop, Simulate
from time import time
#----- Fixed parameters
eta = 1.2
T= 500
N=500
L=100 #Must be a multiple of 100 for technical reasons
gamma = 1/10

nbpoints = 6

alphabar = 1/L #Mean effect of an allele (do not change here)

list_alphaL = np.load("reference_alphaL.npy",allow_pickle=True)
list_alpha = np.repeat(list_alphaL,L//100)/L
np.random.shuffle(list_alpha)

omega = alphabar * np.sqrt(N/gamma)


list_theta = np.logspace(-np.log10(10*L),0,nbpoints) # The rate of mutation from 0 to +1 is theta[0]/2N per organism
                 # per generation per locus


#The starting population must be close to the optimum otherwise the fitness gets
#degenerate when selection is too strong
pop0 = generate_pop((eta,2-eta),N,L)

for (k1,mu1_2N) in enumerate(list_theta[:3]):
  for (k2,mu2_2N) in enumerate(list_theta):
    if (k1==0)*(k2==0):
      time1=time()
    elif (k1==0)*(k2==1):
      time_interval = time()-time1
      print("Estimated total time: "+str(time_interval*nbpoints**2/3600)+"h")
    print("Simulation "+str(k1*nbpoints+k2+1)+" of "+str(nbpoints**2))
    sim = [omega,0,0,0,0]
    sim[1:5] = Simulate((mu1_2N,mu2_2N),omega,eta,N,L,T,initial_pop=pop0,alpha=list_alpha)
    sim = np.array(sim,dtype="object")

    np.save("breakdown_smallmut/sims_L"+str(L)+"_N"+str(N)+"/"+str(k1)+"_"+str(k2)+".npy",sim)
