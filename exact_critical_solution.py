import numpy as np

def get_exact_critical_solution(conds):
    '''
    Returns the exact solution in the crticical case, as a function
    '''
    w0 = np.sqrt(conds.g/conds.l)
    def theta(t):
        return 2*np.arcsin(np.tanh(w0*t))
    return theta