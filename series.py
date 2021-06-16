import numpy as np
import conservation_of_energy

def get_series(conds, N, raw_conds = False):
    '''
    Returns the coefficients of the Taylors series solution to the pendulum 
    problem
    '''
    #Initialize the solution, and its sine and cosine to be filled with zeros. 
    a = np.zeros(N + 1)
    C = np.zeros(N + 1) 
    S = np.zeros(N + 1)
    #set the initial conditions 
    if raw_conds:
        a[0], a[1] = conds.theta0, conds.omega0
    else:
        (a[0], a[1]) = conservation_of_energy.get_expansion_conditions(conds)
    kappa = conds.g / conds.l;
    C[0] = np.cos(a[0]) 
    S[0] = np.sin(a[0]);
    #now apply the key equation (#)
    for n in range(N - 1):
        C[n+1] = -sum((k+1)*a[k+1]*S[n-k] for k in range(n+1))/(n+1)
        S[n+1] = sum((k+1)*a[k+1]*C[n-k] for k in range(n+1))/(n+1)
        a[n+2] = -kappa*S[n]/((n+1)*(n+2)) 
    #and the a is our result
    return a    

def get_series_function(conds, N):
    '''
    Returns a function that will compute the series solution value of theta
    for a given time t
    '''
    coeffs = get_series(conds, N)
    def theta(t):
        if t == 0:
            return coeffs[0]
        return sum(a*(t**i) for (i, a) in enumerate(coeffs))
    return theta

def get_series_deriv_funct(conds, N):
    coeffs = get_series(conds, N)
    def omega(t):
        if t == 0:
            return coeffs[1]
        return sum((k+1)*coeffs[k+1]*(t**k) for k in range(len(coeffs)-1))
    return omega
    