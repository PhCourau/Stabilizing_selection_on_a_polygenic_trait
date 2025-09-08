#MUST BE RUN FROM PARENT DIRECTORY
import numpy as np
import scipy.special as special
import matplotlib as mpl
import matplotlib.pyplot as plt
from solve_fixed_point import *
listcolors=plt.get_cmap("viridis")


theta = (.5,.5)
eta=1.2
L=100
N=500
alphabar = 1/L
list_alpha = np.linspace(0,6/L,L) #numerical instabilities if we allow alpha too large
list_density_alpha = L*np.exp(-L*list_alpha)
list_proba_alpha = list_density_alpha/np.sum(list_density_alpha)


listx = np.linspace(1/(2*N),1-1/(2*N),2*N)

nlevels = 20 #How many levels for the contourplot ?

nbsims = 4

vmax = 100


omega = np.zeros(nbsims)


############### PREDICTIONS
inflation = 100
listx = np.linspace(1/(inflation*N),1-1/(inflation*N),inflation*N)

fig,ax=plt.subplots(1,nbsims+1,figsize=(20,5),gridspec_kw={'width_ratios': [3,3,3,3, 1/3]})
omega = np.zeros(nbsims)
for k in np.arange(nbsims):
  tmp = np.load("contoursims/sims_L"+str(L)+"_N"+str(N)+"/"+str(k)+".npy",allow_pickle=True)
  omega[k] = tmp[0]
  allele_freq = tmp[1]
  alphas = tmp[2]
  ax[k].plot(allele_freq,alphas,"o",color="red")

gamma = N*alphabar**2/omega**2
ome2 = omega**2 /(2*N)


#NEUTRAL DISTRIBUTION
#neutral_distrib = 1/special.beta(2*theta[0],2*theta[1]) * listx**(2*theta[0]-1) * (1-listx)**(2*theta[1]-1)
neutral_distrib = np.ones(len(listx)) #When theta= (.5,.5)


#WEAK SELECTION
Zw=np.zeros((L,inflation*N))
for (kalpha,alpha) in enumerate(list_alpha):
  Zw[kalpha] = neutral_distrib*list_density_alpha[kalpha]
ax[0].contourf(listx,list_alpha,Zw,levels=nlevels,vmin=0,vmax=vmax)

#WEAK SELECTION
s2N_weak = solve_fp(gamma[1],theta,eta,L,alphabar,list_alpha,list_proba_alpha,tolerance=1e-10)

Zm=np.zeros((L,inflation*N))
for (kalpha,alpha) in enumerate(list_alpha):
  Zm[kalpha] = neutral_distrib[k] * np.exp(2*alpha*s2N_weak*listx) * list_density_alpha[kalpha]
  normalization_constant = special.hyp1f1(2*theta[0],2*(theta[0]+theta[1]),2*alpha*s2N_weak)
  Zm[kalpha] /= normalization_constant
ax[1].contourf(listx,list_alpha,Zm,levels=nlevels,vmin=0,vmax=vmax)


#MODERATE SELECTION
s2N_moderate = solve_ms(theta,eta,L,list_alpha,list_proba_alpha)
Zs=np.zeros((L,inflation*N))
for (kalpha,alpha) in enumerate(list_alpha):
  s_ell2N = alpha*s2N_moderate*listx
  Zs[kalpha] = neutral_distrib[k] * np.exp(2*s_ell2N) * list_density_alpha[kalpha]
  normalization_constant = special.hyp1f1(2*theta[0],2*(theta[0]+theta[1]),2*alpha*s2N_moderate)
  Zs[kalpha] /= normalization_constant
ax[2].contourf(listx,list_alpha,Zs,levels=nlevels,vmin=0,vmax=vmax)

#STRONG SELECTION
Zs=np.zeros((L,inflation*N))
s2N_strong = solve_fp(gamma[3],theta,eta,L,alphabar,list_alpha,list_proba_alpha)
for (kalpha,alpha) in enumerate(list_alpha):
  s_ell2N = alpha*s2N_strong*listx - alpha**2 * 2*N * 1/omega[3]**2 *listx*(1-listx)
  Zs[kalpha] = neutral_distrib[k] * np.exp(2*s_ell2N) * list_density_alpha[kalpha]
  normalization_constant = special.hyp1f1(2*theta[0],2*(theta[0]+theta[1]),2*alpha*s2N_moderate)
  Zs[kalpha] /= normalization_constant
ax[3].contourf(listx,list_alpha,Zs,levels=nlevels,vmin=0,vmax=vmax)

########## Plot
ax[0].set_xlabel(r"$P_t$")
ax[0].set_ylabel(r"$\alpha$")

ax[0].set_title(r"$\omega_e^{-2}=0$")
for k in range(1,nbsims):
  ax[k].set_title(r"$\omega_e^{-2}=$"+"{0:.1E}".format(1/ome2[k]))


#Colorbar
# Normalizer
norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
# creating ScalarMappable
sm = plt.cm.ScalarMappable(cmap=listcolors, norm=norm)
sm.set_array([])
plt.colorbar(sm,cax=ax[4],ticks=(0,vmax),format=("%.1e"),label="")


plt.show()
