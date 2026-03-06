import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as scp
from solve_fixed_point import *

############A-C################
theta = (0.1,0.1)
eta = 1.5
L=700
list_alpha = np.load("reference_alphaL.npy",allow_pickle=True)
list_alpha = np.repeat(list_alpha,L//100)/L
#list_alpha = np.random.exponential(L,size=L)
list_proba_alpha = np.ones(L)/L
alphabar = np.sum(list_alpha*list_proba_alpha)


sigma2_0 = genetic_variance_ms(0,theta,L,list_alpha,list_proba_alpha)

s2N_ms = solve_ms(theta,eta,L,list_alpha,list_proba_alpha)
sigma2_ms = genetic_variance_ms(s2N_ms,theta,L,list_alpha,list_proba_alpha)

fig,ax = plt.subplots()
ax.plot([0,1],[L*sigma2_0]*2,color="red")
ax.plot([1,2],[L*sigma2_ms]*2,color="red")
ax.plot([2,3],[0]*2,color="red")
ax.set_xticks([0,1,2,3])
ax.set_yticks([0,0.1,0.2])
ax.set_xlabel(r"b")
ax.set_ylabel(r"$L\sigma^2$")
plt.show()

#Weak selection
list_omem2 = np.logspace(-1,2,20)*L #list of omega_e^{-2}
s2N_ws = np.zeros(20)
list_sig2 = np.zeros(20)

for (k,omem2) in enumerate(list_omem2):
  gamma = omem2/(2*L**2)
  s2N_ws[k] = solve_ws(gamma,
                   theta,
                   eta,
                   L,
                   alphabar,
                   list_alpha,
                   list_proba_alpha,
                   tolerance=1e-5,
                   a0=-1000,
                   b0=1000)
  list_sig2[k] = genetic_variance_ms(s2N_ws[k],theta,L,list_alpha,list_proba_alpha)

fig,ax = plt.subplots()
ax.plot(np.linspace(-1,2,20),L*list_sig2,color="red")
ax.set_xticks([0,1],labels=[r"$10^0$",r"$10^1$"])
ax.set_yticks([0.2,0.3])
ax.set_xlabel(r"$\omega_e^{-2}/L$")
plt.show()


#Strong selection
list_omem2 = np.logspace(-1.3,1.3,20)*L**2 #list of omega_e^{-2}
list_sig2 = np.zeros(20)
for (k,omem2) in enumerate(list_omem2):
  gamma = omem2/(2*L**2)
  list_sig2[k] = genetic_variance_fp_sm(s2N_ms,gamma,theta,L,alphabar,list_alpha,list_proba_alpha)

fig,ax = plt.subplots()
ax.plot(np.linspace(-1.3,1.3,20),L*list_sig2,color="red")
ax.set_xticks([-1,1],labels=[r"$10^{-1}$",r"$10^1$"])
ax.set_yticks([0,0.2])
ax.set_xlabel(r"$\omega_e^{-2}/L^2$")
plt.show()


############D-F################
theta = (0.01,0.1)
sigma2_0 = genetic_variance_ms(0,theta,L,list_alpha,list_proba_alpha)

s2N_ms = solve_ms(theta,eta,L,list_alpha,list_proba_alpha)
sigma2_ms = genetic_variance_ms(s2N_ms,theta,L,list_alpha,list_proba_alpha)

fig,ax = plt.subplots()
ax.plot([0,1],[L*sigma2_0]*2,color="red")
ax.plot([1,2],[L*sigma2_ms]*2,color="red")
ax.plot([2,3],[0]*2,color="red")
ax.set_xticks([0,1,2,3])
ax.set_yticks([0,0.05,0.10],labels=["0.00","0.05","0.10"])
ax.set_xlabel(r"b")
ax.set_ylabel(r"$L\sigma^2$")
plt.show()

#Weak selection
list_omem2 = np.logspace(-1,2,20)*L #list of omega_e^{-2}
s2N_ws = np.zeros(20)
list_sig2 = np.zeros(20)

for (k,omem2) in enumerate(list_omem2):
  gamma = omem2/(2*L**2)
  s2N_ws[k] = solve_ws(gamma,
                   theta,
                   eta,
                   L,
                   alphabar,
                   list_alpha,
                   list_proba_alpha,
                   tolerance=1e-5,
                   a0=-1000,
                   b0=1000)
  list_sig2[k] = genetic_variance_ms(s2N_ws[k],theta,L,list_alpha,list_proba_alpha)

fig,ax = plt.subplots()
ax.plot(np.linspace(-1,2,20),L*list_sig2,color="red")
ax.set_xticks([0,1],labels=[r"$10^0$",r"$10^1$"])
ax.set_yticks([0.05,0.1])
ax.set_xlabel(r"$\omega_e^{-2}/L$")
plt.show()


#Strong selection
list_omem2 = np.logspace(-1.5,1.5,20)*L**2 #list of omega_e^{-2}
list_sig2 = np.zeros(20)
for (k,omem2) in enumerate(list_omem2):
  gamma = omem2/(2*L**2)
  list_sig2[k] = genetic_variance_fp_sm(s2N_ms,gamma,theta,L,alphabar,list_alpha,list_proba_alpha)

fig,ax = plt.subplots()
ax.plot(np.linspace(-1.5,1.5,20),L*list_sig2,color="red")
ax.set_xticks([-1,1],labels=[r"$10^{-1}$",r"$10^1$"])
ax.set_yticks([0,0.1])
ax.set_xlabel(r"$\omega_e^{-2}/L^2$")
plt.show()




