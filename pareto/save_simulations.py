# RUN THIS FROM ITS PARENT DIRECTORY
import numpy as np
from simulate_population import Simulate, generate_pop
#----- Fixed parameters
eta = 1.2
T= 500
N=500
L=100 #Must be a multiple of 100 for technical reasons
theta = (.1,.2) # The rate of mutation from 0 to +1 is muN[0]/N per organism
                 # per generation per locus
alphabar = 1/L #Mean effect of an allele (do not change here)
gamma = .1
omega = alphabar * np.sqrt(N/gamma)


for (k,param) in enumerate(np.linspace(1,4,11)):
  list_alpha = np.random.pareto(param,size=L)
  list_alpha = list_alpha/np.mean(list_alpha) * 1/L


  #The starting population must be close to the optimum otherwise the fitness gets
  #degenerate when selection is too strong
  pop0 = generate_pop((eta,2-eta),N,L)

  allele_freq = np.zeros((T*N,L))
  sim = [omega,0,0,0,0]
  sim[1:5] = Simulate(theta,omega,eta,N,L,T,initial_pop=pop0,alpha=list_alpha)
  sim = np.array(sim,dtype="object")

  sim = np.array(sim,dtype="object")
  np.save("pareto/sims_L"+str(L)+"_N"+str(N)+"/"+str(k)+".npy",sim)
  print("Done step: "+str(k)+";")
