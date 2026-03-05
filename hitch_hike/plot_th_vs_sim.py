import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from solve_fixed_point import *
listcolors=plt.get_cmap("viridis")

eta=1.2
L=200
#Exponentially distributed genetic effects
#list_alpha = np.linspace(0,6/L,100) #numerical instabilities if we allow alpha too large
#list_proba_alpha = L*np.exp(-L*list_alpha)
#list_proba_alpha /= np.sum(list_proba_alpha)
# We know the distribution of the alphas
list_alpha = np.load("reference_alphaL.npy",allow_pickle=True)
list_alpha = np.repeat(list_alpha,L//100)/L
list_proba_alpha = np.ones(L)/L
alphabar = np.sum(list_alpha*list_proba_alpha)
gamma = 1/L * (L*alphabar)**2


klim = 81 #Degree of precision for the theoretical computations under strong selection
#If klim is too low then the genetic variance sigma2_th_fp will have a bump for strong selection


# What shall we plot ?
plotdelta = True
plotnu = True
plotsigma = True
plotrho = True
plotautocor = True
correction = False #Whether to plot the correction for low population size

################################## LOADING ########################################
burn_in = 1/2 #We consider that the system reaches stationarity after a fraction burn_in of the time
########## LOADING N=10 #########
N=10
nbpoints=6

theta = np.zeros((nbpoints,2))
list_means = np.zeros(nbpoints)
list_varmeans = np.zeros(nbpoints)
list_meanvarsX = np.zeros(nbpoints) #L E[alpha**2 X(1-X)]
list_varvarsX = np.zeros(nbpoints) #variance of the previous item
list_meanvars = np.zeros(nbpoints) #trait variance with linkage
list_varvars = np.zeros(nbpoints) #trait variance variance with linkage

tmp = np.load("hitch_hike/sims/theta_0_N"+str(N)+".npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))

for k in range(nbpoints):
  tmp = np.load("hitch_hike/sims/theta_"+str(k)+"_N"+str(N)+".npy",allow_pickle=True)
  theta[k] = tmp[0]
  list_means[k] = eta-np.mean(tmp[3][-postburnin:])
  list_varmeans[k] = np.var(tmp[3][-postburnin:],ddof=1)
  list_meanvarsX[k] = np.mean(tmp[4][-postburnin:])
  list_varvarsX[k] = np.var(tmp[4][-postburnin:],ddof=1)
  list_meanvars[k] = np.mean(tmp[5][-postburnin:])
  list_varvars[k] = np.var(tmp[5][-postburnin:],ddof=1)

omega = np.sqrt(N*alphabar**2/gamma)
omem2 = 2*N/omega**2

thetaplus = np.transpose(theta)[0]

########## LOADING N=30 #########
N3=30

list_means3 = np.zeros(nbpoints)
list_varmeans3 = np.zeros(nbpoints)
list_meanvarsX3 = np.zeros(nbpoints) #L E[alpha**2 X(1-X)]
list_varvarsX3 = np.zeros(nbpoints) #variance of the previous item
list_meanvars3 = np.zeros(nbpoints) #trait variance with linkage
list_varvars3 = np.zeros(nbpoints) #trait variance variance with linkage

tmp = np.load("hitch_hike/sims/theta_0_N"+str(N3)+".npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))

for k in range(nbpoints):
  tmp = np.load("hitch_hike/sims/theta_"+str(k)+"_N"+str(N3)+".npy",allow_pickle=True)
  list_means3[k] = eta-np.mean(tmp[3][-postburnin:])
  list_varmeans3[k] = np.var(tmp[3][-postburnin:],ddof=1)
  list_meanvarsX3[k] = np.mean(tmp[4][-postburnin:])
  list_varvarsX3[k] = np.var(tmp[4][-postburnin:],ddof=1)
  list_meanvars3[k] = np.mean(tmp[5][-postburnin:])
  list_varvars3[k] = np.var(tmp[5][-postburnin:],ddof=1)

########## LOADING N=50 #########
N5=50

list_means5 = np.zeros(nbpoints)
list_varmeans5 = np.zeros(nbpoints)
list_meanvarsX5 = np.zeros(nbpoints) #L E[alpha**2 X(1-X)]
list_varvarsX5 = np.zeros(nbpoints) #variance of the previous item
list_meanvars5 = np.zeros(nbpoints) #trait variance with linkage
list_varvars5 = np.zeros(nbpoints) #trait variance variance with linkage

tmp = np.load("hitch_hike/sims/theta_0_N"+str(N5)+".npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))

for k in range(nbpoints):
  tmp = np.load("hitch_hike/sims/theta_"+str(k)+"_N"+str(N5)+".npy",allow_pickle=True)
  list_means5[k] = eta-np.mean(tmp[3][-postburnin:])
  list_varmeans5[k] = np.var(tmp[3][-postburnin:],ddof=1)
  list_meanvarsX5[k] = np.mean(tmp[4][-postburnin:])
  list_varvarsX5[k] = np.var(tmp[4][-postburnin:],ddof=1)
  list_meanvars5[k] = np.mean(tmp[5][-postburnin:])
  list_varvars5[k] = np.var(tmp[5][-postburnin:],ddof=1)


##### PLOTTING DELTA #####
fig,ax = plt.subplots(2)

if plotdelta:
  #ax[0].set_xscale("log")
  line1 = ax[0].plot(thetaplus,list_means,"<",label="N="+str(N),color="gray")
  line3 = ax[0].plot(thetaplus,list_means3,"v",label="N="+str(N3),color="purple")
  line5 = ax[0].plot(thetaplus,list_means5,"o",label="N="+str(N5),color="red")

###Selection coefficients
#Fixed point
s2N_fp = np.array([solve_fp(gamma,
                                     theta[k],
                                     eta,
                                     L,
                                     alphabar,
                                     list_alpha,
                                     list_proba_alpha,
                                     klim=klim)
             for k in range(nbpoints)])

if plotdelta:
  list_delta_strong = -omega**2 * s2N_fp/(2*N)
  lineth = ax[0].plot(thetaplus,-list_delta_strong,label="Fixed point",color=listcolors(0.6),ls="-.")
  ax[0].set_xlabel(r"$\theta^+$")
  ax[0].set_ylabel(r"$-\Delta$")
  ax[0].legend()

##### PLOTTING SIGMA #####
if plotsigma:
  transparency=1
  ax[1].plot(thetaplus,list_meanvars,marker="<",label="N="+str(N),ls="",color="gray",alpha=transparency)
  #If we neglect linkage, then we expect Var[z] = 2 L E[alpha**2 X(1-X)] = 2 list_meanvarsX
  ax[1].plot(thetaplus,2*list_meanvarsX,marker="3",label="N="+str(N)+" (LE)",ls="",color="gray",alpha=transparency)
  ax[1].plot(thetaplus,list_meanvars3,marker="v",label="N="+str(N3),ls="",color="purple",alpha=transparency)
  ax[1].plot(thetaplus,2*list_meanvarsX3,marker="1",label="N="+str(N3)+" (LE)",ls="",color="purple",alpha=transparency)
  ax[1].plot(thetaplus,list_meanvars5,marker="o",label="N="+str(N5),ls="",color="red",alpha=transparency)
  ax[1].plot(thetaplus,2*list_meanvarsX5,marker="+",label="N="+str(N5)+" (LE)",ls="",color="red",alpha=transparency)

sigma2_th_fp = np.array([ #fixed_point
                      genetic_variance_fp(s2N_fp[k],
                                              gamma,
                                              theta[k],
                                              L,
                                              alphabar,
                                              list_alpha,
                                              list_proba_alpha,
                                              klim=klim)
                      for k in range(nbpoints)])



if plotsigma:
  ax[1].plot(thetaplus,sigma2_th_fp, label="Fixed point",color=listcolors(.6),alpha=transparency,ls="-.")
  ax[1].set_xlabel(r"$\theta^+$")
  ax[1].set_ylabel(r"$\sigma^2$")

ax[1].legend()
plt.show()



