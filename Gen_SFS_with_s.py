# Code to generate predicted SFS for a given mutation rate distribution, NFE demography and selection coefficient, using the formulas in Schraiber et al. 2025
import pandas as pd
import numpy as np
import scipy as sc
from scipy.integrate import simpson
from scipy import stats
import joblib
from joblib import Parallel, delayed, parallel_backend
import multiprocessing


def A(u, tmax, g, a, r, ts):
    """
    Vectorized version of A(u) for multiple values of `u`. A(u) in Schraiber et al. is calculated for a peicewise exponential function

    Parameters:
    - u: Scalar or array of time values at which A is calculated.
    - tmax: Max time for which the demography goes. This is the time at which the SFS is observed
    - g: Population size scaled selection coefficient i.e., 4*N0*s
    - a, r, ts: Piecewise exponential a*exp(r(t-ts))

    Returns:
    - A(u) for each value of u.
    """

    if len(r) != len(a) or len(ts) != len(a):
        raise ValueError("a, r, and t should be the same length")

    u = np.asarray(u)  # Ensure u is a NumPy array
    I = np.searchsorted(ts, u, side='right') - 1  # Find appropriate epoch index

    # Compute integral_bits for all valid indices
    i = np.arange(np.max(I))  # Only compute up to max(I)

    denominator = a[i] * (g + r[i])

    integral_bits = np.where(
        (r[i] == 0) & (g == 0),
        0.5 * (ts[i+1] - ts[i]) / a[i],
        (np.exp(-g/2 * (ts[i] - tmax)) - np.exp(-g/2 * (ts[i+1] - ts[i]) - r[i]/2 * (ts[i+1] - ts[i]) - g/2 * (ts[i] - tmax))) / np.where(denominator == 0, np.inf, denominator)
    )

    last_integral = np.where(
        (r[I] == 0) & (g == 0),
        0.5 * (u - ts[I]) / a[I],  # Safe when g + r[I] = 0
        (np.exp(-g/2 * (ts[I] - tmax)) - np.exp(-g/2 * (u - ts[I]) - r[I]/2 * (u - ts[I]) - g/2 * (ts[I] - tmax))) / np.where((a[I] * (g + r[I])) == 0, np.inf, a[I] * (g + r[I]))
    )

    # Compute cumulative sum and prepend 0
    cumsum_integral_bits = np.insert(np.cumsum(integral_bits), 0, 0)
    
    # Directly index into cumulative sum
    integral_values = cumsum_integral_bits[I] + last_integral

    return integral_values



def d_integrand(u, t, i, n, g, A_diff):
    """
    Computes the integrand function for di_tilde calculation in Schraiber et al.
    Ensures proper handling of potential numerical issues.
    - u: Time at which integrand is evaluated
    - t: maximum time in the pop gen simulation. Time at which sample is taken
    - i: To calculate the xi_i
    - n: Sample size
    - g: Population size scaled selection coefficient i.e., 4*N0*s
    - A_diff: A(t)-A(u) - Compute A(u) for all values in `u` (vectorized)
    
    Returns:
    - Vector of exp(res) values for integration.
    """
    
    # Handle case where A_diff == 0 (avoid log(0))
    A_diff = np.where(A_diff == 0, np.finfo(float).eps, A_diff)

    # Compute the vectorized integrand
    res = (g / 2) * (t - u) + (i - 1) * np.log(A_diff) - (i + 1) * np.log(1 / n + A_diff)

    return np.exp(res)




def d_tilde(u_values, i, t, theta, n, g, A_diff): # num_points=10**4
    """
    Computes d_tilde using the trapezoidal rule (`trapz`).
    
    Parameters:
    - i, t, theta, n, g, A_u, A_t: Input parameters for d_integrand.
    - u_values: A_u is calculated at u_values
    
    Returns:
    - Computed d_tilde value.
    """

    # Compute d_integrand at these points
    integrand_values = d_integrand(u_values, t, i, n, g, A_diff)

    # Perform numerical integration using trapz
    integral = simpson(integrand_values, x=u_values)

    return (theta / (2 * n)) * integral



def bell_polynomial_tilde(n, d):
    """
    Computes the Bell polynomial using dynamic programming.
    
    Parameters:
    - `n`: Order of the Bell polynomial. Maximum n which can be input is the sample size
    - `d`: Input array, must be of length 'n`.
    
    Returns:
    - An array of length `n+1` with Bell polynomial values.
    """

    if len(d) != n:
        raise ValueError("Need the same number of d_tilde(i) as the order you want to compute to")

    b = np.zeros(n + 1)  # Initialize array
    b[0] = 1  # Base case

    for k in range(1, n + 1):
        i = np.arange(0, k)  # Generate index array (0 to k-1)
        b[k] = 1/ (k) * np.sum((i+1) * b[k-1 - i] * d[i])  # Recurrence formula

    return b


def bell_gamma_polynomial(n, d, alpha, beta, beta_p_d0):
    """
    Computes the Bell polynomial using dynamic programming.
    
    Parameters:
    - `n`: Order of the Bell polynomial. Maximum n which can be input is the sample size
    - `d`: Input array, must be of length 'n`.
    
    Returns:
    - An array of length `n+1` with Bell polynomial values.
    """

    if len(d) != n:
        raise ValueError("Need the same number of d_tilde(i) as the order you want to compute to")

    f = np.zeros(n + 1)  # Initialize array
    f[0] = np.exp(alpha*(np.log(beta)-np.log(beta_p_d0)))  # Base case

    for k in range(1, n + 1):
        i = np.arange(1, k+1)  # Generate index array (1 to k)
        f[k] = 1/ (beta_p_d0) * np.sum(((alpha-1)*i+k)/k * f[k - i] * d[i-1])  # Recurrence formula

    return f


def partial_bell_polynomial_tilde(n, d):
    """
    Computes the partial bell polynomial using dynamic programming.
    
    Parameters:
    - `n`: Order of the Bell polynomial. Maximum n which can be input is the sample size
    - `d`: Input array, must be of length 'n`.
    
    Returns:
    - An array of length `n+1 X n+1` with Bell polynomial values.
    """

    if len(d) != n:
        raise ValueError("Need the same number of d_tilde(i) as the order you want to compute to")

    b_part = np.zeros((n+1, n+1))  # Initialize array
    # Base cases
    b_part[0,0] = 1  

    for k in range(1, n + 1):
        for m in range(1, k + 1):
            i = np.arange(0, k-m+1)  # Generate index array (0 to k-m)
            b_part[m, k] = 1/k * np.sum((i+1) * b_part[m-1, k-1-i] * d[i])  # Recurrence formula

    return b_part


def partial_bell_gamma_polynomial(n, d, alpha, beta, beta_p_d0):
    """
    Computes the Bell polynomial using dynamic programming.
    
    Parameters:
    - `n`: Order of the Bell polynomial. Maximum n which can be input is the sample size
    - `d`: Input array, must be of length 'n`.
    
    Returns:
    - An array of length `n+1` with Bell polynomial values.
    """

    if len(d) != n:
        raise ValueError("Need the same number of d_tilde(i) as the order you want to compute to")

    f_part = np.zeros((n + 1, n+1))  # Initialize array
    f_part[0, 0] = np.exp(alpha*(np.log(beta)-np.log(beta_p_d0)))  # Base case

    for k in range(1, n + 1):
        for m in range(1, k + 1):
            i = np.arange(1, k-m+1+1)  # Generate index array (1 to k-m+1)
            f_part[m, k] = 1/ (beta_p_d0) * np.sum(((alpha-1)*i+k)/k * f_part[m-1, k - i] * d[i-1])  # Recurrence formula

    return f_part


def exp_integral(u_values, t, theta, n, g, A_diff): #, num_points = 10**4):
    """
    Computes the exponential integral using numerical integration (`quad`).
    """

    # Compute d_integrand at these points
    integrand_values = np.exp((g / 2) * (t - u_values)) / (1 / n + A_diff)

    # Perform numerical integration using trapz
    integral = simpson(integrand_values, x=u_values)

    return (theta / 2) * integral


# Get p[k] (SFS)

def calc_SFS_k_only(kmax, N, a, ts, r, t, theta, g, n):

    k = np.arange(1, kmax + 1)

    # Generate `num_points` evenly spaced u values from 0 to t
    # Define the intervals
    low_res = np.linspace(0, 68000, num=1000, endpoint=False)  # Exclude 69000
    low_res_1 = np.arange(68000, 69000)  # Exclude 69000
    mid_res = np.arange(69000, 69700, 0.5)  # Medium resolution, step 0.5
    high_res = np.linspace(69700, 70000, num=int((70000-69700)/0.05) + 1)  # Ensures 70000 is included
    
    # Concatenate all intervals into a single array
    u_values = np.concatenate([low_res, low_res_1, mid_res, high_res])/(2*N)

    # Precompute A_t to avoid redundant calculations
    A_t = A(t, t, g, a, r, ts)

    # Compute A(u) for all values in `u` (vectorized)
    A_u = A(u_values, t, g, a, r, ts)

    # Compute A_t - A_u safely
    A_diff = A_t - A_u

    # Ensure A_diff is always positive to avoid log errors
    if np.any(A_diff < 0):
        raise ValueError(f"A_diff is negative! A_t={A_t}, A_u={A_u}")
               
    # Output
    e = np.zeros(kmax) # e_i from i = 1 to kmax

    # Run    
    e_0 = exp_integral(u_values, t, theta, n, g, A_diff)
    
    for k_curr in k:
        e[k_curr-1] = d_tilde(u_values, k_curr, t, theta, n, g, A_diff)
        
    bell_tilde = bell_polynomial_tilde(kmax, e)
    
    # Final SFS
    p = np.exp(-e_0)*bell_tilde

    return p


# Get p[k] (SFS) but for mut rate being Gamma distributed

def calc_SFS_k_only_gamma_mu(kmax, N, a, ts, r, t, alpha, beta, g, n):

    k = np.arange(1, kmax + 1)

    # Generate `num_points` evenly spaced u values from 0 to t
    # Define the intervals
    low_res = np.linspace(0, 68000, num=1000, endpoint=False)  # Exclude 69000
    low_res_1 = np.arange(68000, 69000)  # Exclude 69000
    mid_res = np.arange(69000, 69700, 0.5)  # Medium resolution, step 0.5
    high_res = np.linspace(69700, 70000, num=int((70000-69700)/0.05) + 1)  # Ensures 70000 is included
    
    # Concatenate all intervals into a single array
    u_values = np.concatenate([low_res, low_res_1, mid_res, high_res])/(2*N)

    # Precompute A_t to avoid redundant calculations
    A_t = A(t, t, g, a, r, ts)

    # Compute A(u) for all values in `u` (vectorized)
    A_u = A(u_values, t, g, a, r, ts)

    # Compute A_t - A_u safely
    A_diff = A_t - A_u

    # Ensure A_diff is always positive to avoid log errors
    if np.any(A_diff < 0):
        raise ValueError(f"A_diff is negative! A_t={A_t}, A_u={A_u}")
               
    # Output
    e = np.zeros(kmax) # e_i from i = 1 to kmax

    # Run    
    e_0 = exp_integral(u_values, t, 1, n, g, A_diff)
    
    for k_curr in k:
        e[k_curr-1] = d_tilde(u_values, k_curr, t, 1, n, g, A_diff)
        
    p = bell_gamma_polynomial(kmax, e, alpha, beta, beta+e_0)

    return p


# Get p[k, mut_orig] : Joint dist of mut origin and allele counts

def calc_SFS_k(kmax, N, a, ts, r, t, theta, g, n):

    k = np.arange(1, kmax + 1)

    # Generate `num_points` evenly spaced u values from 0 to t
    # Define the intervals
    low_res = np.linspace(0, 68000, num=1000, endpoint=False)  # Exclude 69000
    low_res_1 = np.arange(68000, 69000)  # Exclude 69000
    mid_res = np.arange(69000, 69700, 0.5)  # Medium resolution, step 0.5
    high_res = np.linspace(69700, 70000, num=int((70000-69700)/0.05) + 1)  # Ensures 70000 is included
    
    # Concatenate all intervals into a single array
    u_values = np.concatenate([low_res, low_res_1, mid_res, high_res])/(2*N)

    # Precompute A_t to avoid redundant calculations
    A_t = A(t, t, g, a, r, ts)

    # Compute A(u) for all values in `u` (vectorized)
    A_u = A(u_values, t, g, a, r, ts)

    # Compute A_t - A_u safely
    A_diff = A_t - A_u

    # Ensure A_diff is always positive to avoid log errors
    if np.any(A_diff < 0):
        raise ValueError(f"A_diff is negative! A_t={A_t}, A_u={A_u}")
               
    # Output
    e = np.zeros(kmax) # e_i from i = 1 to kmax
    
    # Run    
    e_0 = exp_integral(u_values, t, theta, n, g, A_diff)
    
    for k_curr in k:
        e[k_curr-1] = d_tilde(u_values, k_curr, t, theta, n, g, A_diff)
        
    partial_bell_tilde = partial_bell_polynomial_tilde(kmax, e)
    
    # Final SFS
    p_m_k = np.exp(-e_0)*partial_bell_tilde

    return p_m_k


# Get p[k, mut_orig] : Joint dist of mut origin and allele counts. Mut rate is gamma distributed

def calc_SFS_k_gamma_mu(kmax, N, a, ts, r, t, alpha, beta, g, n):

    k = np.arange(1, kmax + 1)

    # Generate `num_points` evenly spaced u values from 0 to t
    # Define the intervals
    low_res = np.linspace(0, 68000, num=1000, endpoint=False)  # Exclude 69000
    low_res_1 = np.arange(68000, 69000)  # Exclude 69000
    mid_res = np.arange(69000, 69700, 0.5)  # Medium resolution, step 0.5
    high_res = np.linspace(69700, 70000, num=int((70000-69700)/0.05) + 1)  # Ensures 70000 is included
    
    # Concatenate all intervals into a single array
    u_values = np.concatenate([low_res, low_res_1, mid_res, high_res])/(2*N)

    # Precompute A_t to avoid redundant calculations
    A_t = A(t, t, g, a, r, ts)

    # Compute A(u) for all values in `u` (vectorized)
    A_u = A(u_values, t, g, a, r, ts)

    # Compute A_t - A_u safely
    A_diff = A_t - A_u

    # Ensure A_diff is always positive to avoid log errors
    if np.any(A_diff < 0):
        raise ValueError(f"A_diff is negative! A_t={A_t}, A_u={A_u}")
               
    # Output
    e = np.zeros(kmax) # e_i from i = 1 to kmax

    # Run    
    e_0 = exp_integral(u_values, t, 1, n, g, A_diff)
    
    for k_curr in k:
        e[k_curr-1] = d_tilde(u_values, k_curr, t, 1, n, g, A_diff)
        
    p = partial_bell_gamma_polynomial(kmax, e, alpha, beta, beta+e_0)

    return p


# Function to compute SFS for a given mutation rate and selection coefficient
def compute_SFS(mu, s, n = 1461892, kmax = 5000, demography = "schraiber_et_al"):
    """
    Computes the SFS using `calc_SFS_k_only` for a given mutation rate (`mu`) and selection coefficient and NFE demography
    """

    # Define demographic parameters
    if demography == "schraiber_et_al":
    
        Ne = np.array([
            14448,   14068,   14068,   14464,   14464,   15208,   15208,   16256,   16256,   17618,
            17618,   19347,   19347,   21534,   21534,   24236,   24236,   27367,   27367,   30416,
            30416,   32060,   32060,   31284,   29404,   26686,   23261,   18990,   16490,   16490,
            12958,   12958,    9827,    9827,    7477,    7477,    5791,    5791,    4670,    4670,
            3841 ,    3841,    3372,    3372,    3287,    3359,    3570,    4095,    4713,    5661,
            7540 ,   11375,   14310,   15887,   79693,  290702,  721018, 1368726, 2166404, 4110830,
            5096386,23368593,31498556  
        ])
    
        Te = np.array([
            70000,   55940,   51395,   47457,   43984,   40877,   38067,   35501,   33141,   30956,
            28922,   27018,   25231,   23545,   21951,   20439,   19000,   17628,   16318,   15063,
            13859,   12702,   11590,   10517,    9482,    8483,    7516,    6580,    5672,    5520,
            5156 ,    4817,    4500,    4203,    3922,    3656,    3404,    3165,    2936,    2718,
            2509 ,    2308,    2116,    1930,    1752,    1579,    1413,    1252,    1096,     945,
            798  ,     656,     517,     448,     283,     179,     113,      71,      45,      28,
            18   ,      11,       7
        ])
    
        # Define R (growth rates)
        r = np.zeros(len(Ne))
        
    elif demography == "GeneBayes":

        Ne = np.array([
            14448,   14068,   14068,   14464,   14464,   15208,   15208,   16256,   16256,   17618,
            17618,   19347,   19347,   21534,   21534,   24236,   24236,   27367,   27367,   30416,
            30416,   32060,   32060,   31284,   29404,   26686,   23261,   18990,   16490,   16490,
            12958,   12958,    9827,    9827,    7477,    7477,    5791,    5791,    4670,    4670,
            3841 ,    3841,    3372,    3372,    3287,    3359,    3570,    4095,    4713,    5661,
            7540 ,   11375,   14310,   13292,   14522,  613285, 5000000  
        ])
    
        Te = np.array([
            70000,   55940,   51395,   47457,   43984,   40877,   38067,   35501,   33141,   30956,
            28922,   27018,   25231,   23545,   21951,   20439,   19000,   17628,   16318,   15063,
            13859,   12702,   11590,   10517,    9482,    8483,    7516,    6580,    5672,    5520,
            5156 ,    4817,    4500,    4203,    3922,    3656,    3404,    3165,    2936,    2718,
            2509 ,    2308,    2116,    1930,    1752,    1579,    1413,    1252,    1096,     945,
            798  ,     656,     517,     383,     252,     124,      50
        ])
    
        # Define R (growth rates)
        r = np.zeros(len(Ne))

    else:

        print("Incorrect demography")
        return
    
    # Transform time values
    ts = (Te[0] - Te) / (2 * Ne[0])
    t = Te[0] / (2 * Ne[0])  # t in coalescent time units
    a = Ne / Ne[0]  # Scaling factor

    N = Ne[0]
    # n = 1461892  # Fixed sample size

    # Compute theta (mutation rate population genetic parameter)
    theta = mu * 4 * Ne[0]
    g = 4 * Ne[0] * s  # Selection parameter

    # Compute SFS
    SFS = calc_SFS_k_only(kmax, N, a, ts, r, t, theta, g, n)

    return mu, s, n, SFS
    


def compute_SFS_gamma_mu_var(mu, var, s, n = 1461892, kmax = 5000, demography = "schraiber_et_al"):
    """
    Computes the SFS for a given Gamma distributed mutation rate with mean `mu` and variance 'var'; selection 's'; NFE demography
    """

    beta = mu/var
    alpha = mu**2/var

    # Define demographic parameters
    if demography == "schraiber_et_al":
    
        Ne = np.array([
            14448,   14068,   14068,   14464,   14464,   15208,   15208,   16256,   16256,   17618,
            17618,   19347,   19347,   21534,   21534,   24236,   24236,   27367,   27367,   30416,
            30416,   32060,   32060,   31284,   29404,   26686,   23261,   18990,   16490,   16490,
            12958,   12958,    9827,    9827,    7477,    7477,    5791,    5791,    4670,    4670,
            3841 ,    3841,    3372,    3372,    3287,    3359,    3570,    4095,    4713,    5661,
            7540 ,   11375,   14310,   15887,   79693,  290702,  721018, 1368726, 2166404, 4110830,
            5096386,23368593,31498556  
        ])
    
        Te = np.array([
            70000,   55940,   51395,   47457,   43984,   40877,   38067,   35501,   33141,   30956,
            28922,   27018,   25231,   23545,   21951,   20439,   19000,   17628,   16318,   15063,
            13859,   12702,   11590,   10517,    9482,    8483,    7516,    6580,    5672,    5520,
            5156 ,    4817,    4500,    4203,    3922,    3656,    3404,    3165,    2936,    2718,
            2509 ,    2308,    2116,    1930,    1752,    1579,    1413,    1252,    1096,     945,
            798  ,     656,     517,     448,     283,     179,     113,      71,      45,      28,
            18   ,      11,       7
        ])
    
        # Define R (growth rates)
        r = np.zeros(len(Ne))
        
    elif demography == "GeneBayes":

        Ne = np.array([
            14448,   14068,   14068,   14464,   14464,   15208,   15208,   16256,   16256,   17618,
            17618,   19347,   19347,   21534,   21534,   24236,   24236,   27367,   27367,   30416,
            30416,   32060,   32060,   31284,   29404,   26686,   23261,   18990,   16490,   16490,
            12958,   12958,    9827,    9827,    7477,    7477,    5791,    5791,    4670,    4670,
            3841 ,    3841,    3372,    3372,    3287,    3359,    3570,    4095,    4713,    5661,
            7540 ,   11375,   14310,   13292,   14522,  613285, 5000000  
        ])
    
        Te = np.array([
            70000,   55940,   51395,   47457,   43984,   40877,   38067,   35501,   33141,   30956,
            28922,   27018,   25231,   23545,   21951,   20439,   19000,   17628,   16318,   15063,
            13859,   12702,   11590,   10517,    9482,    8483,    7516,    6580,    5672,    5520,
            5156 ,    4817,    4500,    4203,    3922,    3656,    3404,    3165,    2936,    2718,
            2509 ,    2308,    2116,    1930,    1752,    1579,    1413,    1252,    1096,     945,
            798  ,     656,     517,     383,     252,     124,      50
        ])
    
        # Define R (growth rates)
        r = np.zeros(len(Ne))

    else:

        print("Incorrect demography")
        return
    
    # Transform time values
    ts = (Te[0] - Te) / (2 * Ne[0])
    t = Te[0] / (2 * Ne[0])  # t in coalescent time units
    a = Ne / Ne[0]  # Scaling factor

    N = Ne[0]
    # n = 1461892  # Fixed sample size

    # Compute theta (mutation rate population genetic parameter)
    g = 4 * Ne[0] * s  # Selection parameter

    # Compute SFS
    SFS = calc_SFS_k_only_gamma_mu(kmax, N, a, ts, r, t, alpha, beta, g, n)

    return mu, var, s, SFS



def compute_SFS_m(mu, s, n = 1461892, kmax = 600, demography = "schraiber_et_al"):
    """
    Computes the joint dist of mutational origin and allele counts for a given mutation rate (`mu`) and selection (s) and NFE demography
    """

    # Define demographic parameters
    if demography == "schraiber_et_al":
    
        Ne = np.array([
            14448,   14068,   14068,   14464,   14464,   15208,   15208,   16256,   16256,   17618,
            17618,   19347,   19347,   21534,   21534,   24236,   24236,   27367,   27367,   30416,
            30416,   32060,   32060,   31284,   29404,   26686,   23261,   18990,   16490,   16490,
            12958,   12958,    9827,    9827,    7477,    7477,    5791,    5791,    4670,    4670,
            3841 ,    3841,    3372,    3372,    3287,    3359,    3570,    4095,    4713,    5661,
            7540 ,   11375,   14310,   15887,   79693,  290702,  721018, 1368726, 2166404, 4110830,
            5096386,23368593,31498556  
        ])
    
        Te = np.array([
            70000,   55940,   51395,   47457,   43984,   40877,   38067,   35501,   33141,   30956,
            28922,   27018,   25231,   23545,   21951,   20439,   19000,   17628,   16318,   15063,
            13859,   12702,   11590,   10517,    9482,    8483,    7516,    6580,    5672,    5520,
            5156 ,    4817,    4500,    4203,    3922,    3656,    3404,    3165,    2936,    2718,
            2509 ,    2308,    2116,    1930,    1752,    1579,    1413,    1252,    1096,     945,
            798  ,     656,     517,     448,     283,     179,     113,      71,      45,      28,
            18   ,      11,       7
        ])
    
        # Define R (growth rates)
        r = np.zeros(len(Ne))
        
    elif demography == "GeneBayes":

        Ne = np.array([
            14448,   14068,   14068,   14464,   14464,   15208,   15208,   16256,   16256,   17618,
            17618,   19347,   19347,   21534,   21534,   24236,   24236,   27367,   27367,   30416,
            30416,   32060,   32060,   31284,   29404,   26686,   23261,   18990,   16490,   16490,
            12958,   12958,    9827,    9827,    7477,    7477,    5791,    5791,    4670,    4670,
            3841 ,    3841,    3372,    3372,    3287,    3359,    3570,    4095,    4713,    5661,
            7540 ,   11375,   14310,   13292,   14522,  613285, 5000000  
        ])
    
        Te = np.array([
            70000,   55940,   51395,   47457,   43984,   40877,   38067,   35501,   33141,   30956,
            28922,   27018,   25231,   23545,   21951,   20439,   19000,   17628,   16318,   15063,
            13859,   12702,   11590,   10517,    9482,    8483,    7516,    6580,    5672,    5520,
            5156 ,    4817,    4500,    4203,    3922,    3656,    3404,    3165,    2936,    2718,
            2509 ,    2308,    2116,    1930,    1752,    1579,    1413,    1252,    1096,     945,
            798  ,     656,     517,     383,     252,     124,      50
        ])
    
        # Define R (growth rates)
        r = np.zeros(len(Ne))

    else:

        print("Incorrect demography")
        return
    
    # Transform time values
    ts = (Te[0] - Te) / (2 * Ne[0])
    t = Te[0] / (2 * Ne[0])  # t in coalescent time units
    a = Ne / Ne[0]  # Scaling factor

    N = Ne[0]
    # n = 1461892  # Fixed sample size

    # Compute theta (mutation rate population genetic parameter)
    theta = mu * 4 * Ne[0]
    g = 4 * Ne[0] * s  # Selection parameter

    # Compute SFS
    SFS = calc_SFS_k(kmax, N, a, ts, r, t, theta, g, n)

    return mu, s, n, SFS



def compute_SFS_m_gamma_mu_var(mu, var, s, n = 1461892, kmax = 600, demography = "schraiber_et_al"):
    """
    Computes the joint dist of mutational origin and allele counts for a Gamma distributed mutation rate with mean `mu` and variance 'var', selection 's'; NFE demography
    """

    beta = mu/var
    alpha = mu**2/var

    # Define demographic parameters
    if demography == "schraiber_et_al":
    
        Ne = np.array([
            14448,   14068,   14068,   14464,   14464,   15208,   15208,   16256,   16256,   17618,
            17618,   19347,   19347,   21534,   21534,   24236,   24236,   27367,   27367,   30416,
            30416,   32060,   32060,   31284,   29404,   26686,   23261,   18990,   16490,   16490,
            12958,   12958,    9827,    9827,    7477,    7477,    5791,    5791,    4670,    4670,
            3841 ,    3841,    3372,    3372,    3287,    3359,    3570,    4095,    4713,    5661,
            7540 ,   11375,   14310,   15887,   79693,  290702,  721018, 1368726, 2166404, 4110830,
            5096386,23368593,31498556  
        ])
    
        Te = np.array([
            70000,   55940,   51395,   47457,   43984,   40877,   38067,   35501,   33141,   30956,
            28922,   27018,   25231,   23545,   21951,   20439,   19000,   17628,   16318,   15063,
            13859,   12702,   11590,   10517,    9482,    8483,    7516,    6580,    5672,    5520,
            5156 ,    4817,    4500,    4203,    3922,    3656,    3404,    3165,    2936,    2718,
            2509 ,    2308,    2116,    1930,    1752,    1579,    1413,    1252,    1096,     945,
            798  ,     656,     517,     448,     283,     179,     113,      71,      45,      28,
            18   ,      11,       7
        ])
    
        # Define R (growth rates)
        r = np.zeros(len(Ne))
        
    elif demography == "GeneBayes":

        Ne = np.array([
            14448,   14068,   14068,   14464,   14464,   15208,   15208,   16256,   16256,   17618,
            17618,   19347,   19347,   21534,   21534,   24236,   24236,   27367,   27367,   30416,
            30416,   32060,   32060,   31284,   29404,   26686,   23261,   18990,   16490,   16490,
            12958,   12958,    9827,    9827,    7477,    7477,    5791,    5791,    4670,    4670,
            3841 ,    3841,    3372,    3372,    3287,    3359,    3570,    4095,    4713,    5661,
            7540 ,   11375,   14310,   13292,   14522,  613285, 5000000  
        ])
    
        Te = np.array([
            70000,   55940,   51395,   47457,   43984,   40877,   38067,   35501,   33141,   30956,
            28922,   27018,   25231,   23545,   21951,   20439,   19000,   17628,   16318,   15063,
            13859,   12702,   11590,   10517,    9482,    8483,    7516,    6580,    5672,    5520,
            5156 ,    4817,    4500,    4203,    3922,    3656,    3404,    3165,    2936,    2718,
            2509 ,    2308,    2116,    1930,    1752,    1579,    1413,    1252,    1096,     945,
            798  ,     656,     517,     383,     252,     124,      50
        ])
    
        # Define R (growth rates)
        r = np.zeros(len(Ne))

    else:

        print("Incorrect demography")
        return
    
    # Transform time values
    ts = (Te[0] - Te) / (2 * Ne[0])
    t = Te[0] / (2 * Ne[0])  # t in coalescent time units
    a = Ne / Ne[0]  # Scaling factor

    N = Ne[0]
    # n = 1461892  # Fixed sample size

    # Compute theta (mutation rate population genetic parameter)
    g = 4 * Ne[0] * s  # Selection parameter

    # Compute SFS
    SFS = calc_SFS_k_gamma_mu(kmax, N, a, ts, r, t, alpha, beta, g, n)

    return mu, var, s, SFS


def compute_SFS_parallel(mu_values, s_values, n_values = [1461892], kmax = 5000, demography = "schraiber_et_al"):
    """
    Computes SFS for multiple mutation rates (`mu_values`), selection coeff ('s_values') and haploid sample size ('n_values') and  in parallel.

    - Returns a SFS (4D matrix - (len(mu_values), len(s_values), len(n_values), kmax+1)) and input mu_values, s_values, n_values
    """

    # Parallel computation
    results = Parallel(n_jobs=-1)(
        delayed(compute_SFS)(mu, s, n, kmax, demography) for mu in mu_values for s in s_values for n in n_values
    )

    # Reshape results
    mu_count = len(mu_values)
    s_count = len(s_values)
    n_count = len(n_values)
    SFS_values = np.array([sfs for _, _, _, sfs in results])

    SFS_matrix = SFS_values.reshape(mu_count, s_count, n_count, -1)

    return SFS_matrix, mu_values, s_values, n_values


def compute_SFS_gamma_mu_var_s_parallel(mu_values, var_values, s_values, n = 1461892, kmax = 5000, demography = "schraiber_et_al"):
    """
    Computes SFS for multiple mutation rates with different mean and variance and different selection (s_values) in parallel.

    - Returns a SFS (4D matrix - (len(mu_values), len(var_values), len(s_values), kmax+1)) and input mu_values, var_values, s_values
    """

    # Parallel computation
    results = Parallel(n_jobs=-1)(
        delayed(compute_SFS_gamma_mu_var)(mu, var, s, n, kmax, demography) for mu in mu_values for var in var_values for s in s_values
    )

    # Convert list to NumPy array (len(mu_values) x 5001)
    SFS_values = np.array([sfs for _, _, _, sfs in results])

    # Reshape results into (mu_count, var_count, 5001)
    mu_count = len(mu_values)
    var_count = len(var_values)
    s_count = len(s_values)

    SFS_matrix = SFS_values.reshape(mu_count, var_count, s_count, -1)

    return SFS_matrix, mu_values, var_values, s_values


def compute_SFS_gamma_mu_var_pairs(mu_var_pairs, s_values, n = 1461892, kmax = 5000, demography = "schraiber_et_al"):
    """
    Computes SFS for multiple mutation rates with given mean and variance of Gamma distribution and different selection 's_values' in parallel.

    - Returns a SFS (3D matrix - (len(mu_var_pairs), len(s_values), kmax+1))
    """

    # Parallel computation
    results = Parallel(n_jobs=-1)(
        delayed(compute_SFS_gamma_mu_var)(mu, var, s, n, kmax, demography) for mu, var in mu_var_pairs for s in s_values
    )

    # Convert list to NumPy array (len(mu_values) x 5001)
    SFS_values = np.array([sfs for _, _, _, sfs in results])

    mu_var_count = len(mu_var_pairs)
    s_count = len(s_values)
    
    SFS_matrix = SFS_values.reshape(mu_var_count, s_count, -1)

    return SFS_matrix


def compute_SFS_m_gamma_mu_var_pairs(mu_var_pairs, s_values, n = 1461892, kmax = 600, demography = "schraiber_et_al"):
    """
    Computes SFS for multiple mutation rates with different mean and variance in parallel.

    - Returns a joint mutational origin (m) and allele count distribution (k) p[m, k]: (4D matrix - (len(mu_var_pairs), len(s_values), kmax+1, kmax+1))
    """

    # Parallel computation
    results = Parallel(n_jobs=-1)(
        delayed(compute_SFS_m_gamma_mu_var)(mu, var, s, n, kmax, demography) for mu, var in mu_var_pairs for s in s_values
    )

    # Convert list to NumPy array (len(mu_values) x 5001)
    SFS_values = np.array([sfs for _, _, _, sfs in results])

    mu_var_count = len(mu_var_pairs)
    s_count = len(s_values)
    
    SFS_matrix = SFS_values.reshape(mu_var_count, s_count, *SFS_values.shape[1:])

    return SFS_matrix


