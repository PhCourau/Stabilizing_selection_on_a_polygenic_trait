import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

eta=1.2
L=100
N=100
alphabar = 1/L
listx = np.linspace(0.8,1.3,100)

postburnin=5000 #Only look at the 1000 last positions



fig,ax=plt.subplots(1,3,figsize=(20,5))
ax2 = [0]*3
sim_numbers = [13,8,1] #We will take simulation number 13 as a reference of weak selection, 6 for moderate etc.
for k in range(3):
  tmp = np.load("sims_L"+str(L)+"_N"+str(N)+"/"+str(sim_numbers[k])+".npy",allow_pickle=True)
  omega = tmp[0]
  list_means = tmp[3][-postburnin:]
  #plt.subplot(131+k)
  ax2[k] = ax[k].twinx()
  ax2[k].hist(list_means,density=True,label=r"Distribution of $\bar z_t$",color="orange")
  ax[k].plot(listx,np.exp(-(listx-eta)**2/(2*omega**2)),color="red",label=r"$F(z)$")
  ax[k].plot([eta],[1],marker="o",color="red")
  ax[k].set_ylim(0,1.1)
  ax[k].yaxis.set_visible(False)
  ax2[k].yaxis.set_visible(False)
  ax[k].set_xlim(listx[0],listx[-1])
  ax[k].set_title(r"$\omega_e^{-2}=$"+"{0:.1E}".format(2*N/omega**2))

ax[2].plot([1],color="orange",label=r"Distribution of $\bar z_t$")
ax[2].legend()
ax[1].set_xlabel(r"$z$")
plt.show()

