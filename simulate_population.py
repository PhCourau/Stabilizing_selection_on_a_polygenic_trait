import numpy as np
import logging
import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)

#This file contains the function Simulate which
# simulates a population under selection, mutation, genetic drift.


# Generating the initial population
def generate_pop(theta,N,L):
    """Generates a population with allele density
       x**(2theta[0]-1)(1-x)**(2theta[1] - 1)
    Parameters
    ----------
    N (int): the population size
    L (int): the number of loci
    theta (positive float array of size 2): the mutation rate. The probability of mutation from 0 to +1 
    (resp. +1 to 0) is theta[0]/(2N) (resp. theta[1]/(2N))
    
    Returns:
    --------
    population: a (N,L,2) boolean array representing the population
    """
    x = np.linspace(1/N,1-1/N,N)
    equilibrium_distribution = x**(2*theta[0]-1) * (1-x)**(2*theta[1]-1)
    equilibrium_distribution /= np.sum(equilibrium_distribution)
    frequencies = np.random.choice(x,p=equilibrium_distribution,size=L)
    population = np.array([[[False]*2]*L]*N,dtype="bool")
    for l in range(L):
        proba = (frequencies[l],1-frequencies[l])
        population[:,l,:] = np.random.choice([True,False],p=proba,size=(N,2))
    return population

# Generating an array of mutations
def generate_mutated(size,theta):
    """
    Generates an array of size size containing mutations. 
    Parameters
    ----------
    size (int) : the size
    theta (positive float array of size 2): the mutation rate. 
            The probability of mutation from -1 to +1 is theta[0]/(2N)
            
    Returns:
    --------
    an array of size size containing independent Bernoulli(theta[0]/(theta[0]+theta[1])) variables.
    """
    optimum_mutation = theta[0]/(theta[0]+theta[1])
    return np.random.choice([True,False],
                            size=size,
                            p=(optimum_mutation,1-optimum_mutation)).astype("bool")



class Population():
    """A population has
    population : a (N,L,2) array of bools representing the L genes of each of the N diploid organisms
    meantrait: a float for the mean trait value in the population
    vartrait: a float for the variance in trait value in the population
    alpha : a string of L floats representing the additive effects at each locus (picked at
                the beginning and then fixed)
    """
    def __init__(self,theta,N,L,alpha,population=None):
        """Population is initialized from
            Parameters
        ----------
        N (int): the population size
        L (int): the number of loci
        theta (positive float array of size 2): denotes the relative balance of 1 and 0 in the population if 
                                                it is generated using generate_pop
        """
        self.alpha=alpha
        if population is None:
            population = generate_pop(theta,N,L)
        self.population = population
        self.meantraits = None
        self.vartraits = None

    def trait(self):
        """Returns an array of size N of trait values within the population"""
        return (self.population@np.ones(2)) @ self.alpha

    def fitness(self,omega,eta,L):
        """Returns an array of size N of fitness values within the population 
        Parameters
        ----------
        omega (positive float): the inverse  strength of selection
        eta (float, between 0 and 2): the selection optimum
        L (int): the number of loci
        """
        return np.exp(-1/(2*omega**2) * (self.trait() - eta)**2)

    def allele_frequencies(self):
        """Returns an array of size L of frequencies of trait-increasing alleles within the population"""
        return np.mean(self.population,axis=(0,2))

    def mutation(self,theta,N,L):
        """Mutation works by sampling a Poisson number of genes and
        mutating them.
        The mutation probability from 0 to +1 (resp. +1 to 0) is theta[0]/(2N) (resp. theta[1]/(2N)) per
        generation per locus per haploid organism.
        Parameters
        ----------
        theta (positive float array of size 2): the mutation rate. The probability of mutation from 0 to +1 
                (resp. +1 to 0) is theta[0]/(2N) (resp. theta[1]/(2N))
        N (int): the population size
        L (int): the number of loci
        """
        number_of_mutations=np.random.poisson(L*(theta[0]+theta[1]))
        mutated_indexes= (np.random.randint(N,size=number_of_mutations),
                        np.random.randint(L,size=number_of_mutations),
                        np.random.randint(2,size=number_of_mutations))
        self.population[mutated_indexes]=generate_mutated(number_of_mutations,theta)

    def selection_drift_sex(self,omega,eta,N,L):
        """
        Generates a new population by sampling N offspring from the current population.
        For each offspring we sample two parents proportionnally
        to fitness, and recombine their genomes with a single uniform crossover.
        Parameters
        ----------
        omega (positive float): the inverse  strength of selection
        eta (float, between 0 and 2): the selection optimum
        N (int): the population size
        L (int): the number of loci
        """
        population_fitnesses = self.fitness(omega,eta,L)
        sum_fitnesses = np.sum(population_fitnesses)
        population_fitnesses = population_fitnesses/sum_fitnesses

        list_parents = np.random.choice(N,p=population_fitnesses,size=2*N)
        list_crossing_overs = np.random.randint(L,size=2*N)
        list_chromosome_shuffling = np.random.randint(2,size=2*N)
        new_population = np.array([[[False]*2]*L]*N,dtype="bool")
        for n in range(N):
            parents = (list_parents[2*n],list_parents[2*n+1])

            new_genome = np.zeros((L,2),dtype="bool")

            crossing_over = list_crossing_overs[2*n:2*n+2]
            new_population[n,:,0] = np.append(
                                        self.population[parents[0],
                                                        :crossing_over[0],
                                                        list_chromosome_shuffling[2*n]],
                                        self.population[parents[0],
                                                        crossing_over[0]:,
                                                        1-list_chromosome_shuffling[2*n]]
                                    )
            new_population[n,:,1] = np.append(
                                        self.population[parents[1],
                                                        :crossing_over[1],
                                                        list_chromosome_shuffling[2*n+1]],
                                        self.population[parents[1],
                                                        crossing_over[1]:,
                                                        1-list_chromosome_shuffling[2*n+1]]
                                    )
        self.population = new_population
        self.meantraits = np.mean(self.trait())
        self.vartraits = np.var(self.trait(),ddof=1)

def Simulate(theta,omega,eta,N,L,T,initial_pop=None,record_every = 1,alpha=None,seed=0):
    """Simulates population evolution over time T. Returns vector of allele frequencies.
    Parameters:
    ----------
    theta: mutation rates. The probability of mutation from -1 to +1 is theta[0]/(2N) per locus
            per generation per haploid genome
    omega: inverse selection strength
    N: diploid population size
    L: number of loci
    T: how long the simulation should last (in units of N)
    recond_every: optional:how often should the population be recorded ?

    Returns:
    list_allele_frequencies,list_varfitnesses,parameters
    """
    np.random.seed(seed)
    
    nb_timesteps = int(T*N)+1
    list_meantraits = np.zeros(nb_timesteps//record_every)
    list_vartraits = np.zeros(nb_timesteps//record_every)
    list_varX = np.zeros(nb_timesteps//record_every)

    pop = Population(theta,N,L,alpha,population=initial_pop)

    for tN,t in enumerate(np.linspace(0,T,nb_timesteps)):
        pop.selection_drift_sex(omega,eta,N,L)
        pop.mutation(theta,N,L)
        if (tN*100)%(nb_timesteps-1)==0:
            logger.info("Done: "+str((tN*100)//nb_timesteps+1)+" per cent")
        if tN%record_every==0:
            list_meantraits[tN//record_every] = pop.meantraits
            list_vartraits[tN//record_every] = pop.vartraits
            list_varX[tN//record_every] = np.sum(pop.alpha**2
                                            *pop.allele_frequencies()*(1-pop.allele_frequencies()))
    return pop.allele_frequencies(), pop.alpha, list_meantraits, list_varX, list_vartraits

