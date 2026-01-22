"""Cython-based helper functions for HMM implementation for ghost admixture."""
from libc.math cimport erf, exp, lgamma, log
from libcpp cimport bool
import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma
from libc.stdio cimport printf

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef exp_focal_coal_interval_cython(int n, double t_admix, double t_interval):
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] Q = create_Q_cython(n)
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] P1 = expm(t_admix * Q)
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] P2 = expm(t_interval * Q)
  cdef double tot_expectation
  cdef double [:, ::1] P1_view = P1
  cdef double [:, ::1] P2_view = P2
  cdef double ps = 0.0
  cdef Py_ssize_t nt0, nt1
  cdef int j

  for nt0 in range(1, n):
    for nt1 in range(1, nt0):
      ps = 0.0
      for j in range(nt0, nt1, -1):
        ps += 2.0 / j
      tot_expectation += ps * P1_view[0, n - nt0] * P2_view[n - nt0, n - nt1]

  return tot_expectation

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef double exp_focal_coal_alt_cython1(int n, double t_admix, double t_interval, double padmix):
  cdef double tot_expectation_alt = 0.0

  for nn in range(2, n + 1):
    tot_expectation_alt += exp_focal_coal_interval_cython(nn, t_admix, t_interval) * binom.pmf(nn, n, padmix)
  return tot_expectation_alt

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef (double, double) exp_focal_coal_cython(int n, double t_admix, double t_interval, double padmix):
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] Q = create_Q_cython(n)
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] P1 = expm(t_admix * Q)
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] P2 = expm(t_interval * Q)
  cdef double tot_expectation
  cdef double tot_expectation_alt = 0.0
  cdef double tot_expectation_null = 0.0
  cdef double [:, ::1] P1_view = P1
  cdef double [:, ::1] P2_view = P2
  cdef double ps = 0.0
  cdef Py_ssize_t nt0, nt1
  cdef int j

  for nn in range(2, n + 1):
    tot_expectation = 0.0
    for nt0 in range(1, nn):
      for nt1 in range(1, nt0):
        ps = 0.0
        for j in range(nt0, nt1, -1):
          ps += 2.0 / j
        tot_expectation += ps * P1_view[n - nn, n - nt0] * P2_view[n - nt0, n - nt1]
    tot_expectation_alt += tot_expectation * binom.pmf(nn, n, padmix)
    if nn == n:
      tot_expectation_null = tot_expectation
  return tot_expectation_null, tot_expectation_alt

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] exp_ncoal_intro(long [::1] nt1, double ne, double t_interval, double padmix, double ne_archaic = 10000.0):
  """Compute the expected number of coalescent events under the introgression model."""
  cdef int n = <int> np.max(nt1)
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] Q = create_Q_cython(n)
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] P1 = expm(t_interval * Q /  (2 * ne_archaic))
  cdef double tot_expectation = 0.0
  cdef double [:, ::1] P1_view = P1
  cdef double ps = 0.0  
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] output = np.zeros(nt1.shape[0])
  cdef double [:] output_view = output
  cdef Py_ssize_t i, nt0
  cdef int j, k
  cdef double tot
  for i in range(nt1.shape[0]):
    for j in range(2, nt1[i] + 1):
      tot = 0.0
      for nt0 in range(1, j):
        ps = 0.0
        for k in range(j, nt0, -1):
          ps += 2.0 / k
        tot += ps * P1_view[n - j, n - nt0]
      p = hypergeom.pmf(j, 2 * ne, int(2 * ne * padmix), nt1[i])
      output_view[i] += tot * p
  return output

cpdef double exp_coal_event(int nt1, int nt2):
  """Compute the expected number of coalescent events
  between two timepoints under the null.

  NOTE: this is assuming there nt1 and nt2 available at the two timepoints
  """
  cdef double y = 1e-9;
  for j in range(nt2 + 1, nt1+1):
    y += 2.0 / j
  return y

cpdef double no_coal_event(int nt1, int nt2):
  """Fast calculation of the probability of no-coalescent events for a given tree."""
  cdef double x = 0.0;
  for j in range(nt2 + 1, nt1+1):
    x += log(1 - 2/j + 1e-32)
  return exp(x)

cdef double logsumexp(double[:] x):
  """Custom definition of the logsumexp function for optimization."""
  cdef int i,n;
  cdef double m = -1e32;
  cdef double c = 0.0;
  n = x.size
  for i in range(n):
      m = max(m,x[i])
  for i in range(n):
      c += exp(x[i] - m)
  return m + log(c)

cpdef double logaddexp(double a, double b):
  """Simple logaddexp function for just two numbers."""
  cdef double m = -1e32;
  cdef double c = 0.0;
  m = max(a,b)
  c = exp(a - m) + exp(b - m)
  return m + log(c)

cdef double xlogy(double x, double y):
  """Implementation of x*log(y)"""
  if x == 0.0:
    return 0.0
  else:
    return x * log(y)


cdef double poisson_logpmf(double x, double mu):
  """Return the logpmf of the Poisson distribution."""
  return xlogy(x, mu) - mu - lgamma(x + 1)


cpdef double emission_coal_event(double x, int z, double pi0, double lamb_n, double lamb0):
  """Emission model for number of coalescent events.

  Arguments:
    x: number of coalescent events on branch in time interval
    z: binary indicator of null or ghost admixture state
    pi0: proportion of 'zero-inflation' that takes place
    lamb_null: expected number of coalescent events under the null
    lamb0: expected wiggle room for reduced portion of lambda

  Returns:
    emission probability of a specific outcome

  """
  if z == 0:
    # The null state (note this still uses the poisson PMF here)
    return poisson_logpmf(x, lamb_n)
  else:
    # The ghost admixture state
    if x == 0:
      return log(pi0 + exp(log(1.0 - pi0) + poisson_logpmf(x, lamb0)))
    else:
      return log(1.0 - pi0) + poisson_logpmf(x, lamb0)


def forward_algo(double[:] x, double[:] lamb_null, double[:] gen_pos,  double p=1e-2, double q=1e-2, double pi0=0.5, double lamb0=1.0):
  """Cython implementation of a helper function for the forward algorithm."""
  cdef int i,j,n,m;
  cdef float p_i, q_i;
  n = x.size
  m = 2
  alphas = np.zeros(shape=(m, n))
  alphas[0, 0] = log(1.0 / m) + emission_coal_event(x=x[0], z=0, pi0=pi0, lamb_n=lamb_null[0], lamb0=lamb0)
  alphas[1, 0] = log(1.0 / m) + emission_coal_event(x=x[0], z=1, pi0=pi0, lamb_n=lamb_null[0], lamb0=lamb0)
  scaler = np.zeros(n)
  scaler[0] = logsumexp(alphas[:, 0])
  alphas[:, 0] -= scaler[0]
  for i in range(1, n):
    # Mark the distance for the forward transition and the probability of staying ...
    di = gen_pos[i] - gen_pos[(i-1)]
    p_i = exp(-p*di)
    q_i = exp(-q*di)
    # This is in log-space ...
    cur_emission0 = emission_coal_event(x=x[i],z=0,pi0=pi0,lamb_n=lamb_null[i],lamb0=lamb0)
    cur_emission1 = emission_coal_event(x=x[i],z=1,pi0=pi0,lamb_n=lamb_null[i],lamb0=lamb0)
    alphas[0, i] = cur_emission0 + logaddexp(log(p_i) + alphas[0, (i - 1)], log(1 - q_i) + alphas[1, (i - 1)])
    alphas[1, i] = cur_emission1 + logaddexp(log(1 - p_i) + alphas[0, (i - 1)], log(q_i) + alphas[1, (i - 1)])
    scaler[i] = logsumexp(alphas[:, i])
    alphas[:, i] -= scaler[i]
  # Returns the alphas, scaler, and sum of the scaler (or loglik)
  return alphas, scaler, sum(scaler)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef forward_algo_product(double[:, :, ::1] es , double p=1e-2, double q=1e-2, double pi0=0.5):
  """Cython implementation of a helper function for the forward algorithm.
  
  Arguments:
    es: emission probabilities (log space)
    p: transition probability from state 0 to state 0
    q: transition probability from state 1 to state 1
    pi0: initial probability of being in state 0
  """
  cdef int i, j, k, n, m, t
  cdef float p_i, q_i, cur_emission0, cur_emission1
  t = es.shape[0]
  n = es.shape[2]
  m = 2
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=3, mode="c"] alphas = np.zeros(shape=(t, m, n))
  cdef double [:, :, ::1] alphas_view = alphas
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] scaler = np.zeros(shape=(t, n))
  cdef double [:, ::1] scaler_view = scaler
  for k in range(t):
    alphas_view[k, 0, 0] = log(pi0) + es[k, 0, 0]
    alphas_view[k, 1, 0] = log(1 - pi0) + es[k, 1, 0]
    scaler_view[k, 0] = logsumexp(alphas_view[k, :, 0])
    for i in range(m):
      alphas_view[k, i, 0] = alphas_view[k, i, 0] - scaler_view[k, 0]
    for i in range(1, n):
      p_i = p
      q_i = q
      # This is in log-space ...
      cur_emission0 = es[k, 0, i]
      cur_emission1 = es[k, 1, i]
      alphas_view[k, 0, i] = cur_emission0 + logaddexp(log(1 - p_i) + alphas_view[k, 0, (i - 1)], log(q_i) + alphas_view[k, 1, (i - 1)])
      alphas_view[k, 1, i] = cur_emission1 + logaddexp(log(p_i) + alphas_view[k, 0, (i - 1)], log(1 - q_i) + alphas_view[k, 1, (i - 1)])
      scaler_view[k, i] = logsumexp(alphas_view[k, :, i])
      for j in range(m):
        alphas_view[k, j, i] = alphas_view[k, j, i] - scaler_view[k, i]
  return alphas, scaler, np.sum(scaler)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef backward_algo_product(double[:, :, ::1] es, double p=1e-2, double q=1e-2):
  """Cython implementation of backward algorithm.
  
  Arguments:
    es: emission probabilities (log space)
    p: transition probability from state 0 to state 0
    q: transition probability from state 1 to state 1
  """
  cdef int i, j, k, n, m, t
  cdef float p_i, q_i, cur_emission0, cur_emission1
  n = es.shape[2]
  t = es.shape[0]
  m = 2
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=3, mode="c"] betas = np.zeros(shape=(t, m, n))
  cdef double [:, :, ::1] betas_view = betas
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] scaler = np.zeros(shape=(t, n))
  cdef double [:, ::1] scaler_view = scaler
  for k in range(t):
    betas_view[k, 0, n - 1] = 0
    betas_view[k, 1, n - 1] = 0
    scaler_view[k, n - 1] = logsumexp(betas_view[k, :, n - 1])
    for i in range(m):
      betas_view[k, i, n - 1] = betas_view[k, i, n - 1] - scaler_view[k, n - 1]
    for i in range(n - 2, -1, -1):
      p_i = p
      q_i = q
      # Calculate the full set of emissions
      cur_emission0 = es[k, 0, i + 1]
      cur_emission1 = es[k, 1, i + 1]
      # betas(z_i) = sum_{j}(beta(z_{i+1}) * p(x_{i+1} | z_{i+1}) * p(z_{i+1} | z_i))
      betas_view[k, 0, i] = logaddexp(betas_view[k, 0, (i + 1)] + cur_emission0 + log(1 - p_i), betas_view[k, 1, (i + 1)] + cur_emission1 + log(p_i))
      betas_view[k, 1, i] = logaddexp(betas_view[k, 0, (i + 1)] + cur_emission0 + log(q_i), betas_view[k, 1, (i + 1)] + cur_emission1 + log(1 - q_i))
      # Do the rescaling here ...
      scaler_view[k, i] = logsumexp(betas_view[k, :, i])
      for j in range(m):
        betas_view[k, j, i] = betas_view[k, j, i] - scaler_view[k, i]
  return betas, scaler, np.sum(scaler)


def backward_algo(double[:] x, double[:] lamb_null, double[:] gen_pos, double p=1e-2, double q=1e-2, double pi0=0.5, double lamb0=1.0):
  """Cython implementation of backward algorithm."""
  cdef int i,j,n,m;
  cdef float p_i, q_i;
  n = x.size
  m = 2
  betas = np.zeros(shape=(m, n))
  betas[0,-1] = 0.0
  betas[1,-1] = 0.0
  scaler = np.zeros(n)
  scaler[-1] = logsumexp(betas[:, -1])
  betas[:, -1] -= scaler[-1]
  for i in range(n - 2, -1, -1):
    di = gen_pos[(i+1)] - gen_pos[i]
    p_i = exp(-p*di)
    q_i = exp(-q*di)
    # Calculate the full set of emissions
    cur_emission0 = emission_coal_event(
        x=x[i+1],
        z=0,
        pi0=pi0,
        lamb_n=lamb_null[i+1],
        lamb0=lamb0
      )
    cur_emission1 = emission_coal_event(
        x=x[i+1],
        z=1,
        pi0=pi0,
        lamb_n=lamb_null[i+1],
        lamb0=lamb0
      )
    # I don't think that this is entirely correct ...
    # betas(z_i) = sum_{j}(beta(z_{i+1}) * p(x_{i+1} | z_{i+1}) * p(z_{i+1} | z_i))
    betas[0,i] = logaddexp(betas[0, (i + 1)] + cur_emission0 + log(p_i), betas[1, (i + 1)] + cur_emission1 + log(1 - p_i))
    betas[1,i] = logaddexp(betas[0, (i + 1)] + cur_emission0 + log(1 - q_i), betas[1, (i + 1)] + cur_emission1 + log(q_i))
    if i == 0:
      betas[0,i] += log(1/m) + emission_coal_event(
        x=x[i],
        z=0,
        pi0=pi0,
        lamb_n=lamb_null[i],
        lamb0=lamb0
      )
      betas[1,i] += log(1/m) + emission_coal_event(
        x=x[i],
        z=1,
        pi0=pi0,
        lamb_n=lamb_null[i],
        lamb0=lamb0
      )
    # Do the rescaling here ...
    scaler[i] = logsumexp(betas[:, i])
    betas[:, i] -= scaler[i]
  return betas, scaler, sum(scaler)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef update_oneind_cython(double [:, :, ::1] alphas, double [:, :, ::1] betas, double [:, :, ::1] emissions, double p, double q):
  """Update the transition probabilities for the one-individual case.
  Arguments:
    m: number of trees
  """
  cdef int m = alphas.shape[2]
  cdef int l = alphas.shape[0]
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] n = np.zeros(shape = (l, m - 1))
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] eta01 = np.zeros(shape = (l, m - 1))
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] eta10 = np.zeros(shape = (l, m - 1))
  cdef double[4] norm_factor
  cdef double [:, ::1] n_view = n
  cdef double [:, ::1] eta01_view = eta01
  cdef double [:, ::1] eta10_view = eta10
  cdef int i, j, t

  for t in range(l):
    for i in range(m - 1):
        j = i + 1
        norm_factor = [0.0, 0.0, 0.0, 0.0]
        norm_factor[0] = (
            alphas[t, 0, i]
            + log(1.0 - p)
            + betas[t, 0, j]
            + emissions[t, 0, j]
        )
        norm_factor[1] = (
            alphas[t, 0, i]
            + log(p)
            + betas[t, 1, j]
            + emissions[t, 1, j]
        )
        norm_factor[2] = (
            alphas[t, 1, i]
            + log(q)
            + betas[t, 0, j]
            + emissions[t, 0, j]
        )
        norm_factor[3] = (
            alphas[t, 1, i]
            + log(1 - q)
            + betas[t, 1, j]
            + emissions[t, 1, j]
        )
        n_view[t, i] = logsumexp(norm_factor)
        eta01_view[t, i] = (
            alphas[t, 0, i]
            + log(p)
            + betas[t, 1, j]
            + emissions[t, 1, j]
        ) - n_view[t, i]
        eta10_view[t, i] = (
            alphas[t, 1, i]
            + log(q)
            + betas[t, 0, j]
            + emissions[t, 0, j]
        ) - n_view[t, i]
  return eta01, eta10

def marginal_loglik(double[:] xs, double[:] lamb_null, double[:] gen_pos,  double p=1e-2, double q=1e-2, double pi0=0.5, double lamb0=1.0):
  """Cython implementation of a helper function for the forward algorithm."""
  cdef int i,j,m,n;
  cdef float p_i, q_i;
  n = xs.shape[0]
  m = 2
  alphas = np.zeros(shape=(m, n))
  alphas[0, 0] = np.log(0.5) + emission_coal_event(x=xs[0], z=0, pi0=pi0, lamb_n=lamb_null[0], lamb0=lamb0)
  alphas[1, 0] = np.log(0.5) + emission_coal_event(x=xs[0], z=1, pi0=pi0, lamb_n=lamb_null[0], lamb0=lamb0)
  scaler = np.zeros(n)
  scaler[0] = logsumexp(alphas[:, 0])
  alphas[:, 0] -= scaler[0]
  for i in range(1, n):
    # Mark the distance for the forward transition and the probability of staying ...
    di = gen_pos[i] - gen_pos[(i-1)]
    p_i = exp(-p*di)
    q_i = exp(-q*di)
    # This is in log-space ...
    cur_emission0 = emission_coal_event(
        x=xs[i],
        z=0,
        pi0=pi0,
        lamb_n=lamb_null[i],
        lamb0=lamb0
      )
    cur_emission1 = emission_coal_event(
        x=xs[i],
        z=1,
        pi0=pi0,
        lamb_n=lamb_null[i],
        lamb0=lamb0
      )
    alphas[0, i] = cur_emission0 + logaddexp(log(p_i) + alphas[0, (i - 1)], log(1 - q_i) + alphas[1, (i - 1)])
    alphas[1, i] = cur_emission1 + logaddexp(log(1 - p_i) + alphas[0, (i - 1)], log(q_i) + alphas[1, (i - 1)])
    scaler[i] = logsumexp(alphas[:, i])
    alphas[:, i] -= scaler[i]
  # Returns the alphas, scaler, and sum of the scaler (or loglik)
  return sum(scaler)

# ------ Section for Product HMM helper functions ------- #



'''
def ecm_full_update(xss, gammas, alpha, a1, a2, b1, b2, pi0):
  """Compute the two ECM updates in lockstep.
  Arguments:
    - xss (`np.array`): array of the product across the
    - gammas (`np.array`): array of posterior probability of being in the non-null category
    - alpha (`float`): the proportion in the first tmrca cutoff
    - a1 (`float`): the shape parameter for t_admix
    - a2 (`float`): the shape parameter for t_archaic
    - b1 (`float`): the scale parameter for t_admix
    - b2 (`float`): the scale parameter for t_archaic
    - pi0 (`float`): the proportion of the null category
    - cp (`float`): the first crossing point for the two distributions
  """
  if gammas.ndim > 2:
      p_gammas = np.exp(gammas[:, 1,:])
  else:
      p_gammas = np.exp(gammas[1,:])
  # NOTE: this is kind of arbitrary right now ...
  xs_flat = xss[p_gammas >= 0.8,:].flatten()
  if xs_flat.size > 0:
    zs0 = np.array([posterior_assignment(x, alpha, a1, a2, b1, b2) for x in xs_flat])
    a1_hat, a2_hat, alpha_hat = ecm_update_shape(xs_flat, zs0, 1 - zs0, b1, b2)
    b1_hat, b2_hat = ecm_update_scales(xs_flat, zs0, 1 - zs0, a1_hat, a2_hat)
    if gammas.ndim > 2:
        pi0_hat = np.exp(gammas[:, 0, 0]).mean()
    else:
        pi0_hat = np.exp(gammas[0, 0])
    return alpha_hat, a1_hat, a2_hat, b1_hat, b2_hat, pi0_hat
  else:
    return alpha, a1, a2, b1, b2, pi0
'''
