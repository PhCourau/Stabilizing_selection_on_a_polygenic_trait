# Run this from parent directory !
from simulate_population import Simulate, generate_pop
from simulate_population_dir import Simulate as Simulate_dir
import numpy as np
import matplotlib.pyplot as plt

T=1000
burnin = 5000
N=10
L=200
eta=1.2
theta = (.1,.2)

list_alphaL = np.load("reference_alphaL.npy",allow_pickle=True)
list_alpha = np.repeat(list_alphaL,L//100)/L
np.random.shuffle(list_alpha)
alphabar = np.sum(list_alpha)/L

gamma = 1/L
omega = alphabar * np.sqrt(N/gamma)

pop0 = generate_pop((eta,2-eta),N,L)

sim = Simulate(theta,omega,eta,N,L,T,initial_pop=pop0,alpha=list_alpha)

sstar = -(np.mean(sim[2][-burnin:])-eta)/omega**2

sim_dir = Simulate_dir(theta,sstar,N,L,T,initial_pop=pop0,alpha=list_alpha)

fig,ax = plt.subplots(2)

line1=ax[0].plot(2*sim[3][-500:],label=r"$\sigma_{t}^2$ (LE)")
line2=ax[0].plot(sim[4][-500:],label=r"$\sigma_t^2$")
ax[0].legend()
ax[0].set_title("Stabilizing selection")

line3=ax[1].plot(2*sim_dir[3][-500:],label=r"$\sigma_{t}^2$ (LE)")
line4=ax[1].plot(sim_dir[4][-500:],label=r"$\sigma_t^2$")
ax[1].legend()
ax[1].set_xlabel("Time (generations)")
ax[1].set_title("Directional selection")

plt.show()
