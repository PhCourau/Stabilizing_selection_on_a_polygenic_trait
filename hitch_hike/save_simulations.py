import numpy as np
from simulate_population import generate_pop, Simulate
from time import time
#----- Fixed parameters
eta = 1.2
T= 10000
N=50
L=200 #Must be a multiple of 100 for technical reasons
list_theta = [(.05,.1),(.06,.12),(.07,.14),(.08,.16),(.09,.18),(.1,.2)] # The rate of mutation from 0 to +1 is muN[0]/N per organism
                 # per generation per locus

alphabar = 1/L #Mean effect of an allele (do not change here)

list_alphaL = np.load("reference_alphaL.npy",allow_pickle=True)
list_alpha = np.repeat(list_alphaL,L//100)/L
np.random.shuffle(list_alpha)

omega = alphabar * np.sqrt(N*L) #Weak selection


#The starting population must be close to the optimum otherwise the fitness gets
#degenerate when selection is too strong
pop0 = generate_pop((eta,2-eta),N,L)

for (k,theta) in enumerate(list_theta):
	if k==0:
		time1=time()
	elif k==1:
		time_interval = time()-time1
		print("Estimated total time: "+str(time_interval*len(list_theta)/3600)+"h")
	print("Simulation "+str(k+1)+" of "+str(len(list_theta)))
	sim = [theta,0,0,0,0]
	sim[1:5] = Simulate(theta,omega,eta,N,L,T,initial_pop=pop0,alpha=list_alpha)
	sim = np.array(sim,dtype="object")

	np.save("hitch_hike/sims/theta_"+str(k)+"_N"+str(N)+".npy",sim)
