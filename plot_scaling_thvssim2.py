import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from solve_fixed_point import *
listcolors=plt.get_cmap("viridis")

theta= (.1,.2)
eta=1.2
L=100
#Exponentially distributed genetic effects
#list_alpha = np.linspace(0,6/L,100) #numerical instabilities if we allow alpha too large
#list_proba_alpha = L*np.exp(-L*list_alpha)
#list_proba_alpha /= np.sum(list_proba_alpha)
# We know the distribution of the alphas
list_alpha = np.load("reference_alphaL.npy",allow_pickle=True)
list_alpha = np.repeat(list_alpha,L//100)/L
list_proba_alpha = np.ones(L)/L
alphabar = np.sum(list_alpha*list_proba_alpha)


klim = 81 #Degree of precision for the theoretical computations under strong selection
#If klim is too low then the genetic variance sigma2_th_fp will have a bump for strong selection

limit_autocor = 1000 #We will compute the correlation between Delta_t and Delta_{t+s} for s smaller
                     #than limit_autocor

# What shall we plot ?
plotdelta = True
plotnu = True
plotsigma = True
plotrho = True
plotautocor = True


################################## LOADING ########################################
burn_in = 1/2 #We consider that the system reaches stationarity after a fraction burn_in of the time
########## LOADING N=100 #########
N=50
nbpoints=20

omega = np.zeros(nbpoints)
list_means = np.zeros(nbpoints)
list_varmeans = np.zeros(nbpoints)
list_meanvarsX = np.zeros(nbpoints) #L E[alpha**2 X(1-X)]
list_varvarsX = np.zeros(nbpoints) #variance of the previous item
list_meanvars = np.zeros(nbpoints) #trait variance with linkage
list_varvars = np.zeros(nbpoints) #trait variance variance with linkage
list_rho = np.zeros(nbpoints) #Autocorrelation parameters

tmp = np.load("sims_L"+str(L)+"_N"+str(N)+"/"+str(0)+".npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))

for k in range(nbpoints):
  tmp = np.load("sims_L"+str(L)+"_N"+str(N)+"/"+str(k)+".npy",allow_pickle=True)
  omega[k] = tmp[0]
  list_means[k] = eta-np.mean(tmp[3][-postburnin:])
  list_varmeans[k] = np.var(tmp[3][-postburnin:],ddof=1)
  list_rho[k] = -np.log(np.cov(tmp[3][-postburnin-1:-1],tmp[3][-postburnin:])[0,1]/list_varmeans[k])
  list_meanvarsX[k] = np.mean(tmp[4][-postburnin:])
  list_varvarsX[k] = np.var(tmp[4][-postburnin:],ddof=1)
  list_meanvars[k] = np.mean(tmp[5][-postburnin:])
  list_varvars[k] = np.var(tmp[5][-postburnin:],ddof=1)

gamma = N*alphabar**2/omega**2
omem2 = 2*N/omega**2

########## LOADING N=500 #########
N2=500
nbpoints2=20

omega2 = np.zeros(nbpoints2)
list_means2 = np.zeros(nbpoints2)
list_varmeans2 = np.zeros(nbpoints2)
list_meanvarsX2 = np.zeros(nbpoints2) #L E[alpha**2 X(1-X)]
list_varvarsX2 = np.zeros(nbpoints2) #variance of the previous item
list_meanvars2 = np.zeros(nbpoints2) #trait variance with linkage
list_varvars2 = np.zeros(nbpoints2) #trait variance variance with linkage
list_rho2 = np.zeros(nbpoints2) #Autocorrelation parameters
list_autocor = np.zeros((nbpoints2,limit_autocor)) #Autocorrelations

tmp = np.load("sims_L"+str(L)+"_N"+str(N2)+"/"+str(0)+".npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))
for k in range(nbpoints2):
  tmp = np.load("sims_L"+str(L)+"_N"+str(N2)+"/"+str(k)+".npy",allow_pickle=True)
  omega2[k] = tmp[0]
  list_means2[k] = eta-np.mean(tmp[3][-postburnin:])
  list_varmeans2[k] = np.var(tmp[3][-postburnin:],ddof=1)
  list_rho2[k] = -np.log(np.cov(tmp[3][-postburnin-1:-1],tmp[3][-postburnin:])[0,1]/list_varmeans2[k])
  list_meanvarsX2[k] = np.mean(tmp[4][-postburnin:])
  list_varvarsX2[k] = np.var(tmp[4][-postburnin:],ddof=1)
  list_meanvars2[k] = np.mean(tmp[5][-postburnin:])
  list_varvars2[k] = np.var(tmp[5][-postburnin:],ddof=1)
  if plotautocor:
    list_autocor[k,0] = 1
    for j in range(1,limit_autocor):
      list_autocor[k,j] =np.cov(tmp[3][-postburnin:-j],tmp[3][-postburnin+j:])[0,1]/list_varmeans2[k]

tmp = None

gamma2 = N2*alphabar**2/omega2**2
omem22 = 2*N2/omega2**2 #This is the strength of selection 1/omega_e^{-2}

##### PLOTTING DELTA #####
fig,ax = plt.subplots(2,2)

if plotdelta:
  ax[0,0].set_xscale("log")
  ax[0,0].set_yscale("log")
  line1 = ax[0,0].plot(omem2,list_means,"v",label="N="+str(N),color="blue")
  line2 = ax[0,0].plot(omem22,list_means2,"^",label="N="+str(N2),color="purple")

###Selection coefficients
#Strong selection
s2N_fp = np.array([solve_fp(gamma[k],
                                     theta,
                                     eta,
                                     L,
                                     alphabar,
                                     list_alpha,
                                     list_proba_alpha,
                                     klim=klim)
             for k in range(nbpoints)])

#Moderate selection
s2N_ms = solve_ms(theta,eta,L,list_alpha,list_proba_alpha)


if plotdelta:
  list_delta_strong = -omega**2 * s2N_fp/(2*N)
  line4 = ax[0,0].plot(omem2,-list_delta_strong,label="Fixed point",color=listcolors(0.6),ls="-.")
  list_delta_ms = -omega**2 * s2N_ms/(2*N)
  line5 = ax[0,0].plot(omem2,-list_delta_ms,label="Moderate selection",color=listcolors(.75),ls="--")
  ax[0,0].set_xlabel(r"$\omega_e^{-2}$")
  ax[0,0].set_ylabel(r"$-\Delta$")
  ax[0,0].legend()

##### PLOTTING SIGMA #####
if plotsigma:
  ax[1,0].set_xscale("log")
  ax[1,0].set_yscale("log")
  transparency=1
  ax[1,0].plot(omem2,np.sqrt(list_meanvars),marker="v",label="N="+str(N),ls="",color="blue",alpha=transparency)
  #If we neglect linkage, then we expect Var[z] = 2 L E[alpha**2 X(1-X)] = 2 list_meanvarsX
  ax[1,0].plot(omem2,np.sqrt(2*list_meanvarsX),marker="1",label="N="+str(N)+" (no linkage)",ls="",color="blue",alpha=transparency)
  ax[1,0].plot(omem22,np.sqrt(list_meanvars2),marker="^",label="N="+str(N2),ls="",color="purple",alpha=transparency)
  ax[1,0].plot(omem22,np.sqrt(2*list_meanvarsX2),marker="2",label="N="+str(N2)+" (no linkage)",ls="",color="purple",alpha=transparency)

sigma2_th_fp = np.array([ #fixed_point
                      genetic_variance_fp(s2N_fp[k],
                                              gamma[k],
                                              theta,
                                              L,
                                              alphabar,
                                              list_alpha,
                                              list_proba_alpha,
                                              klim=klim)
                      for k in range(nbpoints)])


sigma2_th_ms = genetic_variance_ms(s2N_ms,theta,L,list_alpha,list_proba_alpha)*np.ones(nbpoints)

if plotsigma:
  ax[1,0].loglog(omem2,np.sqrt(sigma2_th_fp), label="Fixed point",color=listcolors(.6),alpha=transparency,ls="-.")
  ax[1,0].loglog(omem2,np.sqrt(sigma2_th_ms),label="Moderate selection",color=listcolors(.75),alpha=transparency,ls="--")
  ax[1,0].set_xlabel(r"$\omega_e^{-2}$")
  ax[1,0].set_ylabel(r"$\sigma$")
  ax[1,0].legend()

#####   PLOTTING NU #####
if plotnu:
  ax[0,1].loglog(omem2,np.sqrt(list_varmeans),"v",label="N="+str(N),color="blue")
  ax[0,1].loglog(omem22,np.sqrt(list_varmeans2),"^",label="N="+str(N2),color="purple")
  nu = alphabar/np.sqrt(2*gamma)
  neutral_prediction = sigma2_th_fp/(theta[0]+theta[1]) #Keeping in mind theta = mu * 2*N
  ax[0,1].loglog(omem2,np.sqrt(1/(1/neutral_prediction+1/nu**2)),label="Weak selection",color=listcolors(.9),ls="-")
  ax[0,1].loglog(omem2,nu,label="Moderate/strong selection",color=listcolors(.6),ls="--")


  ax[0,1].legend()
  ax[0,1].set_xlabel(r"$\omega_e^{-2}$")
  ax[0,1].set_ylabel(r"$\nu$")


##### PLOTTING N RHO #####
if plotrho:
  ax[1,1].loglog(omem2,2*N*list_rho,"v",label="N="+str(N),color="blue")
  ax[1,1].loglog(omem22,2*N2*list_rho2,"^",label="N="+str(N2),color="purple")

  rhoN_th_ms = gamma *sigma2_th_ms/alphabar**2
  rhoN_th_fp = gamma *sigma2_th_fp/alphabar**2
  correctedrhoN_th = gamma *sigma2_th_fp/alphabar**2 + (theta[0]+theta[1])/2
                                #recall |mu| = (theta[0]+theta[1])/(2*N)


  ax[1,1].loglog(omem2,2*correctedrhoN_th,label="Weak selection",color=listcolors(0.9),ls="-")
  ax[1,1].loglog(omem2,2*rhoN_th_fp,label="Fixed point",color=listcolors(0.6),ls="-.")
  ax[1,1].loglog(omem2,2*rhoN_th_ms,label="Moderate selection",color=listcolors(0.75),ls="--")
  ax[1,1].set_xlabel(r"$\omega_e^{-2}$")
  ax[1,1].set_ylabel(r"$\rho$")
  ax[1,1].legend()
  plt.show()


##### PLOTTING AUTOCORRELATION #####
list_rhoth = correctedrhoN_th/N2

rho_u = np.array([-np.log(list_autocor[k])/list_rho2[k] for k in np.arange(0,nbpoints2,2)])
u0 = 1/(2*N2)
list_u = np.repeat([np.arange(limit_autocor)],nbpoints2//2,axis=0)

list_omem2 = np.repeat([omem22[::2]],limit_autocor,axis=0)
list_omem2 = np.transpose(list_omem2)
list_omem2 = list_omem2[::-1,:]

if plotautocor:

  plt.scatter(list_u * u0,rho_u,c=list_omem2,alpha=0.4,norm=mpl.colors.LogNorm())

  plt.plot(np.arange(limit_autocor)/(2*N),np.arange(limit_autocor),color="black")
  color_bar=plt.colorbar(label=r"$\omega_e^{-2}$")
  color_bar.set_alpha(1)
  color_bar.draw_all()


  plt.xlabel(r"$u$")
  plt.ylabel(r"$\rho_u / \rho_{u_0}$")
  plt.legend()
  plt.show()


