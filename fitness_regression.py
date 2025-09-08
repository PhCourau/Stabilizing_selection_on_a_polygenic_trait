import numpy as np
import matplotlib.pyplot as plt
from simulate_population import Population



eta = 1.2
T= 100
N=100
L=100 #Must be a multiple of 100 for technical reasons
mu2N = (.1,.2) # The rate of mutation from 0 to +1 is thetaN[0]/N per organism
                 # per generation per locus
alphabar = 1/L #Mean effect of an allele (do not change here)

list_alphaL = np.load("reference_alphaL.npy",allow_pickle=True)
list_alpha = np.repeat(list_alphaL,L//100)/L
np.random.shuffle(list_alpha)

gamma = .1
omega = alphabar * np.sqrt(N/gamma)

pop = Population(mu2N,N,L,alpha=list_alpha)
for t in range(T*N):
  pop.selection_drift_sex(omega,eta,N,L)
  pop.mutation(mu2N,N,L)


ell = 0
gene_content = np.sum(pop.population[:,ell,:],axis=-1) #gene content at locus ell
fitnesses = np.log(pop.fitness(omega,eta,L))
fitness_classes = [fitnesses[gene_content==0],fitnesses[gene_content==1],fitnesses[gene_content==2]]

regression_coefficient = np.cov(gene_content,fitnesses)[0,1]/np.var(gene_content,ddof=1)
intercept = np.mean(fitnesses) - regression_coefficient*np.mean(gene_content)

plt.boxplot(fitness_classes,labels=[0,1,2])
plt.plot(np.arange(1,4),intercept + regression_coefficient*np.arange(3),color="red")

plt.ylabel(r"$W(z)$")
plt.xlabel(r"$g_\ell$")
plt.show()
