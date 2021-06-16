import numpy as np
import conservation_of_energy

def subcritical_repeat(approximant, T):
    #This is a a paramateric decorator, is all
    def get_repetition(t):
        mod = t % T
        mod = mod if mod >= 0 else mod + T
        if mod <= .25 * T:
            return approximant(mod)
        elif mod <= .5 * T:
            return -approximant(.5*T-mod)
        elif mod <= .75 * T:
            return -approximant(mod - .5*T)
        else:
            return approximant(T-mod)
    return get_repetition
    
def supercritical_repeat(approximant, T, flip = False):
    #similarly a decorator
    sig = -1 if flip else 1
    def get_repetition(t):
        mod = t % T
        mod = mod if mod >= 0 else mod + T
        return sig * (approximant(mod) if mod < .5*T else -approximant(T-mod))
    return get_repetition

def critical_repeat(approximant):
    #just makes sure to clamp the series for convenience
    clamp = lambda x : max(-np.pi, min(np.pi, x))
    return lambda x : clamp(approximant(x))

def repeat(conds, approximant):
    behavior = conservation_of_energy.get_behavior(conds)
    T_star = conservation_of_energy.get_T_star(conds)
    if behavior == 'subcritical':
        return subcritical_repeat(approximant, 4*T_star)
    elif behavior == 'supercritical':
        return supercritical_repeat(approximant, 2*T_star, flip = (np.sign(conds.omega0) == 1))
    elif behavior == 'critical':
        return critical_repeat(approximant)
    return approximant