import numpy as np
import conservation_of_energy
import series
import exact_critical_solution

def get_approx_coeffs(conds, N):
    '''
    Returns the coefficients of the approximant. 
    '''
    if conservation_of_energy.get_behavior(conds) == 'critical':
        return None
    #get the series solution
    a = series.get_series(conds, N)
    #modify the early terms to subtract jet at T star
    T_star = conservation_of_energy.get_T_star(conds)
    wb = -conservation_of_energy.get_bottom_velocity(conds)
    a[0] += wb * T_star
    a[1] -= wb
    #next, we actually compute the caychy's product rule
    a_hat = np.zeros(N+1)
    for n in range(len(a_hat)):
        a_hat[n] = sum(a[n-k] * (k+1) * (T_star ** (-k-2)) for k in range(n+1))
    return a_hat

    
def get_approximant(conds, N):
    '''
    Returns a function that computes the approximant for a given time
    '''
    if conservation_of_energy.get_behavior(conds) == 'critical':
        return exact_critical_solution.get_exact_critical_solution(conds)
    T_star = conservation_of_energy.get_T_star(conds)
    wb = -conservation_of_energy.get_bottom_velocity(conds)
    coeffs = get_approx_coeffs(conds, N)
    def theta(t):
        if t == 0:
            return -wb*T_star + coeffs[0]*(T_star ** 2)
        return wb * (t - T_star) + (t - T_star)**2 * sum(a * (t ** i) for (i, a) in enumerate(coeffs))
    return theta