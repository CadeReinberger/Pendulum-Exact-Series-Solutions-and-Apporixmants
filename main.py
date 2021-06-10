import numpy as np
import series
import conditions
import approximant
from matplotlib import pyplot as plt
import numerical_solution
import conservation_of_energy

#makes the assumption you go from 0 to pi. Fine for now. 
clamp = lambda x : max(-.1, min(3.2, x))

def plot_all(conds, N):
    print('Behavior: ' + conservation_of_energy.get_behavior(m_conds))
    times, n_thetas = numerical_solution.numerical_solve(conds)
    series_funct = series.get_series_function(conds, N)
    approx_funct = approximant.get_approximant(conds, N)
    s_thetas = np.array([clamp(series_funct(t)) for t in times])
    a_thetas = np.array([clamp(approx_funct(t)) for t in times])
    print(times)
    plt.plot(times, n_thetas, 'k')
    plt.plot(times, s_thetas, 'c')
    plt.plot(times, a_thetas, 'm')
    
m_conds = conditions.Conditions(theta0 = np.deg2rad(0), omega0 = np.sqrt(2)+.01)
m_N = 4

plot_all(m_conds, m_N)