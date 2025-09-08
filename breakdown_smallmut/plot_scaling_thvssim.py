#TO BE RUN FROM THE PARENT DIRECTORY
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from solve_fixed_point import *
listcolors=plt.get_cmap("viridis")

L=100
N=500

nbpoints=6
list_theta = np.logspace(-np.log10(10*L),0,nbpoints)

eta=1.2
# We know the distribution of the alphas
list_alpha = np.load("reference_alphaL.npy",allow_pickle=True)/L
list_proba_alpha = np.ones(L)/L
alphabar = np.sum(list_alpha*list_proba_alpha)
gamma = 1/10
omega = alphabar *np.sqrt(N/gamma)

klim = 81 #Degree of precision for the theoretical computations under strong selection
#If klim is too low then the genetic variance sigma2_th_ss will have a bump for strong selection


################################## LOADING ########################################
burn_in = 1/2 #We consider that the system reaches stationarity after a fraction burn_in of the time
########## LOADING N=500 #########
Delta = np.zeros((nbpoints,nbpoints))
nu2 = np.zeros((nbpoints,nbpoints))
genicvar = np.zeros((nbpoints,nbpoints)) #L E[alpha**2 X(1-X)]
vargenicvar = np.zeros((nbpoints,nbpoints)) #variance of the previous item
sigma2 = np.zeros((nbpoints,nbpoints)) #trait variance with linkage
varsigma2 = np.zeros((nbpoints,nbpoints)) #trait variance variance with linkage
rho = np.zeros((nbpoints,nbpoints)) #Autocorrelation parameters

tmp = np.load("breakdown_smallmut/sims_L"+str(L)+"_N"+str(N)+"/0_0.npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))


for (k1,mu1_2N) in enumerate(list_theta):
  for (k2,mu2_2N) in enumerate(list_theta):
    tmp = np.load("breakdown_smallmut/sims_L"+str(L)+"_N"+str(N)+"/"+str(k1)+"_"+str(k2)+".npy",allow_pickle=True)
    Delta[k1,k2] = np.mean(tmp[3][-postburnin:])-eta
    nu2[k1,k2] = np.var(tmp[3][-postburnin:],ddof=1)
    genicvar[k1,k2] = np.mean(tmp[4][-postburnin:])
    vargenicvar[k1,k2] = np.var(tmp[4][-postburnin:],ddof=1)
    sigma2[k1,k2] = np.mean(tmp[5][-postburnin:])
    varsigma2[k1,k2] = np.var(tmp[5][-postburnin:],ddof=1)
    rho[k1,k2] = -np.log(np.cov(tmp[3][-postburnin-1:-1],tmp[3][-postburnin:])[0,1]/nu2[k1,k2])



#Computations
s2Nth = np.zeros((nbpoints,nbpoints))
Deltath = np.zeros((nbpoints,nbpoints))
sigma2th = np.zeros((nbpoints,nbpoints))

nu2th = alphabar**2/(2*gamma)

for (k1,mu1_2N) in enumerate(list_theta):
  for (k2,mu2_2N) in enumerate(list_theta):
    theta = (mu1_2N,mu2_2N)
    s2Nth[k1,k2] = solve_fp(gamma,theta,eta,L,alphabar,list_alpha,list_proba_alpha)
    Deltath[k1,k2] = -omega**2 * s2Nth[k1,k2]/(2*N)
    sigma2th[k1,k2] = genetic_variance_fp(s2Nth[k1,k2],
                                              gamma,
                                              theta,
                                              L,
                                              alphabar,
                                              list_alpha,
                                              list_proba_alpha,
                                              klim=klim)
rhoNth = gamma *sigma2th/alphabar**2


#Heatmaps
str_theta = ["{:2.2}".format(np.log10(theta)) for theta in list_theta]

fig,ax = plt.subplots(2,3,figsize=(10*9/8,10),gridspec_kw={'width_ratios': [3,3,1/3]})

sb.heatmap(np.abs((Delta-Deltath)/Deltath),xticklabels=str_theta,yticklabels=str_theta,vmax=1,vmin=0,cbar=True,ax=ax[0,0],cbar_ax=ax[0,2])
ax[0,0].set_title(r"$|(\Delta_e-\Delta_{th})/\Delta_{th}|$")
ax[0,0].set_xlabel(r"$\log\theta^{+}$")
ax[0,0].set_ylabel(r"$\log\theta^{-}$")
ax[0,0].plot([0,6],[0,6],color="blue",ls="--")
for x in np.linspace(0,3,7):
  ax[0,0].plot([0,x],[x,0],color="grey",lw=3)

sb.heatmap(np.abs((sigma2-sigma2th)/sigma2th),xticklabels=str_theta,yticklabels=str_theta,vmax=1,vmin=0,cbar=False,ax=ax[1,0])
ax[1,0].set_title(r"$|(\sigma^2_e-\sigma^2_{th})/\sigma^2_{th}|$")
ax[1,0].set_xlabel(r"$\log\theta^{+}$")
ax[1,0].set_ylabel(r"$\log\theta^{-}$")
ax[1,0].plot([0,6],[0,6],color="blue",ls="--")
for x in np.linspace(0,3,7):
  ax[1,0].plot([0,x],[x,0],color="grey",lw=3)

sb.heatmap(np.abs((nu2-nu2th)/nu2th),xticklabels=str_theta,yticklabels=str_theta,vmax=1,vmin=0,cbar=False,ax=ax[0,1])
ax[0,1].set_title(r"$|(\nu^2_e-\nu^2_{th})/\nu^2_{th}|$")
ax[0,1].set_xlabel(r"$\log\theta^{+}$")
ax[0,1].set_ylabel(r"$\log\theta^{-}$")
ax[0,1].plot([0,6],[0,6],color="blue",ls="--")
for x in np.linspace(0,3,7):
  ax[0,1].plot([0,x],[x,0],color="grey",lw=3)

sb.heatmap(np.abs((rho*N-rhoNth)/rhoNth),xticklabels=str_theta,yticklabels=str_theta,mask=(rho<0),vmax=1,vmin=0,cbar=False,ax=ax[1,1])
ax[1,1].set_title(r"$|(\rho_e -\rho_{th})/\rho_{th}|$")
ax[1,1].set_xlabel(r"$\log\theta^{+}$")
ax[1,1].set_ylabel(r"$\log\theta^{-}$")
ax[1,1].plot([0,6],[0,6],color="blue",ls="--")
for x in np.linspace(0,3,7):
  ax[1,1].plot([0,x],[x,0],color="grey",lw=3)


ax[1,2].set_visible(False)


plt.show()
