import numpy as np
import conditions
from matplotlib import pyplot as plt
import numerical_solution
import conservation_of_energy
import solution

def plot_all(conds, N):
    PLOT_X = 15
    print('Behavior: ' + conservation_of_energy.get_behavior(m_conds))
    times, n_thetas = numerical_solution.numerical_solve(conds, TT = 10)
    series_funct = solution.get_series_solution(conds, N)
    approx_funct = solution.get_approximant(conds, N)
    s_thetas = np.array([series_funct(t) for t in times])
    a_thetas = np.array([approx_funct(t) for t in times])
    plt.plot(times, s_thetas, 'c')
    plt.plot(times, a_thetas, 'm')
    plt.plot(times[::len(times)//PLOT_X-1], n_thetas[::len(times)//PLOT_X-1], 'k.')
    
m_conds = conditions.Conditions(theta0 = np.deg2rad(450), omega0 = -.9)
m_N = 10

plot_all(m_conds, m_N)