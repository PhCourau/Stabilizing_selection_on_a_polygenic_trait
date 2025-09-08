import numpy as np
import scipy.special as special
import scipy.integrate as integ
import warnings

def gss(f, a0, b0, tolerance=1e-5):
  """
  Golden-section search
  to find the minimum of f on [a,b]
  * f: a strictly unimodal function on [a,b]

  Example:
  >>> def f(x): return (x - 2) ** 2
  >>> x = gss(f, 1, 5)
  >>> print(f"{x:.5f}")
  2.00000

  Credit to Wikipedia
  """
  a=a0
  b=b0
  invphi = (np.sqrt(5) - 1) / 2  # 1 / phi (inverse golden ratio)
  while b - a > tolerance:
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi
    if f(c) < f(d):
      b = d
    else:  # f(c) > f(d) to find the maximum
      a = c
  if ((a-a0 < 2*tolerance) or (b0-b< 2*tolerance)):
    warnings.warn("The golden-section search found an answer close to the boundaries, maybe widen up the domain of the search")
  return (b + a) / 2

def mean_X(s2N,theta):
  """Finds the mean X associated with a modified beta solution
  to a WF with selection s2N, mutation theta"""
  return (theta[0]/(theta[0]+theta[1])
          *special.hyp1f1(2*theta[0]+1,2*(theta[0]+theta[1])+1,2*s2N)
          /special.hyp1f1(2*theta[0],2*(theta[0]+theta[1]),2*s2N))

def mean_sigma2(s2N,theta):
  """Finds the mean X(1-X) associated with a modified beta solution
  to a WF with selection s2N, mutation theta"""
  return (theta[0]*theta[1]/((theta[0]+theta[1])*(theta[0]+theta[1]+1/2))
          *special.hyp1f1(2*theta[0]+1,2*(theta[0]+theta[1])+2,2*s2N)
          /special.hyp1f1(2*theta[0],2*(theta[0]+theta[1]),2*s2N))

def mean_phenotype(s2N,theta,L,list_alpha,list_proba_alpha):
  """Finds the mean phenotype.
  Arguments:
  ----------
  theta: mutation rates (tuple of two positive floats)

  s2N: effective selection coefficient s^* (multiplied by 2N)

  list_alpha: list of classes of values for gene effect. Typically np.linspace(a,b)

  list_proba_alpha: the list of probabilities of each class in list_alpha
  """
  return L*np.sum(2*list_alpha*mean_X(s2N*list_alpha,theta)*list_proba_alpha)

def genetic_variance(s2N,theta,L,list_alpha,list_proba_alpha):
  """Finds the mean LE[2alpha**2 X(1-X)].
  Arguments:
  ----------
  theta: mutation rates (tuple of two positive floats)

  s2N: effective selection coefficient s^* (multiplied by 2N)

  list_alpha: list of classes of values for gene effect. Typically np.linspace(a,b)

  list_proba_alpha: the list of probabilities of each class in list_alpha
  """
  return L*np.sum(2*list_alpha**2*mean_sigma2(s2N*list_alpha,theta)*list_proba_alpha)

############################ WEAK SELECTION ##################################
def match_equation_weak(s2N,theta,eta,L,gamma,alphabar,list_alpha,list_proba_alpha):
  """Finds how well the equation for weak selection is satisfied
  L: the number of loci
  gamma: the strength of stabilizing selection
  alphabar: the mean effect of a locus
  """
  return np.abs( (mean_phenotype(s2N,theta,L,list_alpha,list_proba_alpha)-eta)
                 *2*gamma/alphabar**2 + s2N)

def find_s2N_weak(gamma,
                 theta,
                 eta,
                 L,
                 alphabar,
                 list_alpha,
                 list_proba_alpha,
                 tolerance=1e-5,
                 a0=-1000,
                 b0=1000):
  """Finds the value of s^* given by the equation for weak selection. The search
     is made on the interval (a0,b0)"""
  return gss(lambda s2N: match_equation_weak(s2N,
                                           theta,
                                           eta,
                                           L,
                                           gamma,
                                           alphabar,
                                           list_alpha,
                                           list_proba_alpha),
             a0,b0, #Boundaries of the golden-section search
             tolerance=tolerance
  )


############################ MODERATE SELECTION ##################################
def match_equation_moderate(s2N,theta,eta,L,list_alpha,list_proba_alpha):
  """Finds the absolute value of the distance between the mean phenotype and the
  optimum eta"""
  return np.abs(mean_phenotype(s2N,theta,L,list_alpha,list_proba_alpha)-eta)


def find_s2N_moderate(theta,
                     eta,
                     L,
                     list_alpha,
                     list_proba_alpha,
                     tolerance=1e-5,
                     a0=-1000,
                     b0=1000):
  """Finds the value of s^* given by the equation for moderate selection, searches
     on the interval (a0,b0)"""
  return gss(lambda s: match_equation_moderate(s,
                                               theta,
                                               eta,
                                               L,
                                               list_alpha,
                                               list_proba_alpha),
             a0,b0, #Boundaries of the golden-section search
             tolerance=tolerance
  )


############################ STRONG SELECTION ##################################
# First we need to compute the integral of x^{a-1} (1-x)^{b - 1} e^{cx + dx(1-x)}
def integral_WF_FD(a,b,c,d,klim):
  """
  Computes
  sum_{k} c^k / (k!)   a(a+1)...(a+k-1)/((a+b)(a+b+1)...(a+b+k-1)) 1F1(a+k,b;a+b+k;d)
  One can check that this is proportionnal to the integral of
  x^{a-1} (1-x)^{b-1} e^{cx + dx(1-x)}
  with proportionnality coefficient independent of c and d.
  klim is the precision of the approximation

  WARNING: this method fails if d>65 for klim=81
  """
  return np.sum([d**j/special.factorial(j) * special.poch(a,j) * special.poch(b,j)
                 /special.poch(a+b,2*j) * special.hyp1f1(a+j,a+b+2*j,c)
                for j in range(klim)],axis=0)


def mean_X_FD(s2N,d2N,theta,klim=21):
  """
  Computes the mean of a WF diffusion with frequency-dependent (FD) selection
  xi(x) = s + d(1-2x)
  and mutation rates given by mu.

  Parameters
  ----------
  theta: tuple of two positive floats: mutation rates
  s2N: float: directional selection
  d2N: float: dominance coefficient
  klim: int: the precision of the approximation
  """
  return (theta[0]/(theta[0]+theta[1])
         *integral_WF_FD(2*theta[0]+1,2*theta[1],2*s2N,2*d2N,klim)
         /integral_WF_FD(2*theta[0],2*theta[1],2*s2N,2*d2N,klim))

def var_WF_FD(s2N,d2N,theta,klim=21):
  """
  Computes E[X(1-X)] where X is a WF diffusion with frequency-dependent (FD) selection
  xi(x) = s + d(1-2x)
  and mutation rates given by mu.

  Parameters
  ----------
  theta: tuple of two positive floats: mutation rates
  s2N: float: directional selection
  d2N: float: dominance coefficient
  klim: int: the precision of the approximation (should be odd)
  """
  return np.max([np.min([theta[0]*theta[1]/((theta[0]+theta[1])*(theta[0]+theta[1]+1/2))
         *integral_WF_FD(2*theta[0]+1,2*theta[1]+1,2*s2N,2*d2N,klim)
         /integral_WF_FD(2*theta[0],2*theta[1],2*s2N,2*d2N,klim),
		         np.ones(np.shape(s2N))/4],axis=0),
                 np.zeros(np.shape(s2N))],axis=0)



def mean_phenotype_strong(s2N,d2N,theta,L,list_alpha,list_proba_alpha,klim=21):
  """Finds the mean phenotype.
  Arguments:
  ----------
  theta: mutation rates (tuple of two positive floats)

  s2N: effective selection coefficient s^*

  list_alpha: list of classes of values for gene effect. Typically np.linspace(a,b)

  list_proba_alpha: the list of probabilities of each class in list_alpha
  """
  return L*np.sum(2*list_alpha*mean_X_FD(s2N*list_alpha,
                                         d2N*list_alpha**2,
                                         theta,
                                         klim=klim)
                  *list_proba_alpha)

def genetic_variance_strong(s2N,gamma,theta,L,alphabar,list_alpha,list_proba_alpha,klim=21):
  """Finds the mean E[2 alpha**2 X(1-X)].
  Arguments:
  ----------
  theta: mutation rates (tuple of two positive floats)

  s2N: effective selection coefficient s^*

  alphabar: mean effect of an allele

  list_alpha: list of classes of values for gene effect. Typically np.linspace(a,b)

  list_proba_alpha: the list of probabilities of each class in list_alpha
  """
  return L*np.sum(2*list_alpha**2*var_WF_FD(s2N*list_alpha,
                                            -gamma*(list_alpha/alphabar)**2,
                                            theta,
                                            klim=klim)
                  *list_proba_alpha)

def match_equation_strong(s2N,d2N,theta,eta,L,list_alpha,list_proba_alpha,klim=21):
  """Finds the absolute value of the distance between the mean phenotype and the
  optimum eta for strong selection"""
  return np.abs(mean_phenotype_strong(s2N,d2N,theta,L,list_alpha,list_proba_alpha,klim)
                -eta)


def find_s2N_strong(gamma,
                   theta,
                   eta,
                   L,
                   alphabar,
                   list_alpha,
                   list_proba_alpha,
                   tolerance=1e-5,
                   a0=-1000,
                   b0=1000,
                   klim=21):
  """Finds the value of s^* given by the equation for strong selection for a WF
  diffusion with frequency-dependent selection
  xi(x,alpha) = s^* alpha +  N alpha**2/(2 om2)  (1-2x)

  Parameters:
  ----------
  om2: inverse strength of selection
  theta: mutation rates (the probability of mutation at one locus from 0 to +1 is
          theta[0] per organism per generation)
  eta: optimum
  N: population size (number of diploid organisms)
  L: number of loci
  list_alpha: list of values of additive effects
  list_proba_alpha: list of probabilities of the additive effects
  tolerance: precision
  a0,b0: domain in which the search is made
  """
  return gss(lambda s2N: match_equation_strong(s2N,
                                              -gamma/alphabar**2,
                                              theta,
                                              eta,
                                              L,
                                              list_alpha,
                                              list_proba_alpha,
                                              klim=klim),
             a0,b0, #Boundaries of the golden-section search
             tolerance=tolerance
  )


############################ STRONG SELECTION (small mutation rates) ##################################
# If the mutation rate is small, then all computations for strong selection are the same as for moderate selection
# except for the genetic variance. We have
def mean_X1_X_SM(s2N,d2N,theta):
  """
  Computes the integral of e**(cx + dx(1-x))
  which is approximately equal to
  E[X(1-X)]
  if theta is small and X has distribution
  x**(theta[0]-1) * x**(theta[1]-1) * np.exp(s2N*x + d2N*x*(1-x))

  Parameters
  ----------
  theta: tuple of two positive floats: mutation rates
  s2N: array of float: directional selection coefficients to be evaluated
  d2N: float: dominance coefficients to be evaluated
  """
  return np.array([integ.quad(lambda x: np.exp(s2N[k]*x+d2N[k]*x*(1-x)),0,1)[0]
          /special.hyp1f1(2*theta[0],2*(theta[0]+theta[1]),s2N[k])
          * theta[0]*theta[1]/(theta[0]+theta[1]) #normalization constant
                   for k in range(np.shape(s2N)[0])
                  ])

def genetic_variance_strong_SM(s2N,gamma,theta,L,alphabar,list_alpha,list_proba_alpha):
  """Finds the mean E[2 alpha**2 X(1-X)] under the assumption of small mutation rates.
  Arguments:
  ----------
  theta: mutation rates (tuple of two positive floats)

  s2N: effective selection coefficient s^*

  alphabar: mean effect of an allele

  list_alpha: list of classes of values for gene effect. Typically np.linspace(a,b)

  list_proba_alpha: the list of probabilities of each class in list_alpha
  """
  return L*np.sum(2*list_alpha**2*mean_X1_X_SM(s2N*list_alpha,
                                               -gamma*(list_alpha/alphabar)**2,
                                            theta)
                  *list_proba_alpha)
