import numpy as np
from scipy.integrate import odeint
import conservation_of_energy

def numerical_solve(conds, TT = None, dt = None):
    TT = TT if not TT is None else conservation_of_energy.get_T_star(conds)
    #handle critical case
    if TT is None:
        w0 = np.sqrt(conds.g/conds.l)
        TT = (1/w0) * np.arctanh(np.sin(.49*np.pi))
    dt = dt if not dt is None else .0001
    #uses a dyanmic RK to integrate the ODE numerically. 
    times = np.arange(0, TT, dt)
    kappa = conds.g / conds.l
    deriv = lambda x, t : (x[1], -kappa * np.sin(x[0]))
    init_cond = conservation_of_energy.get_expansion_conditions(conds)
    ps = odeint(deriv, init_cond, times)
    thetas = ps[:,0]
    return times, thetas
