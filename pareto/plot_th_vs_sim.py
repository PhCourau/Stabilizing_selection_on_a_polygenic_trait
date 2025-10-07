#TO BE RUN FROM THE PARENT DIRECTORY
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from solve_fixed_point import *
listcolors=plt.get_cmap("viridis")

L=100
N=500

nbpoints=11
list_param = np.linspace(1,4,nbpoints) #Parameters of the Pareto distribution

theta = (.1,.2)

eta=1.2
alphabar = 1/L
gamma = 1/10
omega = alphabar *np.sqrt(N/gamma)

klim = 81 #Degree of precision for the theoretical computations under strong selection
#If klim is too low then the genetic variance sigma2_th_ss will have a bump for strong selection

# What shall we plot ?
plotdelta = True
plotnu = True
plotsigma = True
plotrho = True


################################## LOADING ########################################
burn_in = 1/2 #We consider that the system reaches stationarity after a fraction burn_in of the time
########## LOADING N=500 #########
Delta = np.zeros(nbpoints)
nu2 = np.zeros(nbpoints)
genicvar = np.zeros(nbpoints) #L E[alpha**2 X(1-X)]
vargenicvar = np.zeros(nbpoints) #variance of the previous item
sigma2 = np.zeros(nbpoints) #trait variance with linkage
varsigma2 = np.zeros(nbpoints) #trait variance variance with linkage
rho = np.zeros(nbpoints) #Autocorrelation parameters

list_alpha = np.zeros((nbpoints,L))
list_proba_alpha = np.ones(L)/L

tmp = np.load("pareto/sims_L"+str(L)+"_N"+str(N)+"/0.npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))


for (k,param) in enumerate(list_param):
  tmp = np.load("pareto/sims_L"+str(L)+"_N"+str(N)+"/"+str(k)+".npy",allow_pickle=True)
  Delta[k] = np.mean(tmp[3][-postburnin:]) - eta
  nu2[k] = np.var(tmp[3][-postburnin:],ddof=1)
  genicvar[k] = np.mean(tmp[4][-postburnin:])
  vargenicvar[k] = np.var(tmp[4][-postburnin:],ddof=1)
  sigma2[k] = np.mean(tmp[5][-postburnin:])
  varsigma2[k] = np.var(tmp[5][-postburnin:],ddof=1)
  rho[k] = -np.log(np.cov(tmp[3][-postburnin-1:-1],tmp[3][-postburnin:])[0,1]/nu2[k])*2*N
  list_alpha[k] = tmp[2]


#Computations
s2Nth = np.zeros(nbpoints)
Deltath = np.zeros(nbpoints)
sigma2th = np.zeros(nbpoints)


for (k,param) in enumerate(list_param):
  s2Nth[k] = solve_fp(gamma,theta,eta,L,alphabar,list_alpha[k],list_proba_alpha)
  sigma2th[k] = genetic_variance_fp(s2Nth[k],
                                              gamma,
                                              theta,
                                              L,
                                              alphabar,
                                              list_alpha[k],
                                              list_proba_alpha,
                                              klim=klim)

Deltath = -omega**2 * s2Nth/(2*N)
nu2th = alphabar**2/(2*gamma) * np.ones(nbpoints)
rhoth = 2*gamma *sigma2th/alphabar**2 * np.ones(nbpoints)


#Plots
fig,ax = plt.subplots(2,2)

ax[0,0].plot(list_param,Delta,"v",color="purple",label="Simulation")
ax[0,0].plot(list_param,Deltath,"o",color="blue",label="Prediction")
ax[0,0].set_xlabel(r"$k$")
ax[0,0].set_ylabel(r"$\Delta$")
ax[0,0].legend()

ax[0,1].plot(list_param,nu2,"v",color="purple")
ax[0,1].plot(list_param,nu2th,"o",color="blue")
ax[0,1].set_xlabel(r"$k$")
ax[0,1].set_ylabel(r"$\nu^2$")
ax[0,1].set_yscale("log")

ax[1,0].plot(list_param,sigma2,"v",color="purple",label="Simulation")
ax[1,0].plot(list_param,2*genicvar,marker="1",ls="",color="purple",label="Simulation (no linkage)")
ax[1,0].plot(list_param,sigma2th,"o",color="blue",label="Prediction")
ax[1,0].set_xlabel(r"$k$")
ax[1,0].set_ylabel(r"$\sigma^2$")
ax[1,0].set_yscale("log")
ax[1,0].legend()

ax[1,1].plot(list_param,rho,"v",color="purple")
ax[1,1].plot(list_param,rhoth,"o",color="blue")
ax[1,1].set_xlabel(r"$k$")
ax[1,1].set_ylabel(r"$\rho$")
ax[1,1].set_yscale("log")
plt.show()
