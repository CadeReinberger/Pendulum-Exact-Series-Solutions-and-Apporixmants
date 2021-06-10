import numpy as np
from scipy.special import ellipk

def get_epsilon(conds):
    '''
    Returns the reduced energy of the system
    '''
    return .5 * (conds.l * conds.omega0)**2 + conds.g * conds.l * (1 - np.cos(conds.theta0))

def get_behavior(conds):
    '''
    Returns a string giving the case of the trajectory
    '''
    epsilon = get_epsilon(conds)
    critical_epsilon = 2 * conds.g * conds.l
    if np.isclose(epsilon, critical_epsilon):
        return 'critical'
    elif epsilon < critical_epsilon:
        return 'subcritical'
    elif epsilon > critical_epsilon:
        return 'supercritical'
    else:
        return None
    
def get_bottom_velocity(conds):
    '''
    Returns the angular velocity at the bottom
    '''
    epsilon = get_epsilon(conds)
    return np.sqrt(2*epsilon)/conds.l

def get_top_angle(conds):
    '''
    Returns the top angle of the trajectory
    '''
    if get_behavior(conds) in ('critical', 'supercritical'):
        return np.pi
    return np.arccos(1 - get_epsilon(conds) / (conds.g * conds.l))

def get_top_velocity(conds):
    '''
    Returns the angular velocity of the trajectory at the top
    '''
    if get_behavior(conds) in ('critical', 'subcritical'):
        return 0
    return np.sqrt((2*get_epsilon(conds)/(conds.l**2)) - (4*conds.g/conds.l))
    
def get_expansion_conditions(conds):
    '''
    Returns the 0-th and first derivative at the best point of expansion
    '''
    if get_behavior(conds) == 'critical':
        return (0, get_bottom_velocity(conds))
    return (get_top_angle(conds), -get_top_velocity(conds))

def get_T_star(conds):
    '''
    Gives the value of T-star, the time from the top angle to the bottom angle
    '''
    behavior = get_behavior(conds)
    if behavior == 'subcritical':
        t0 = get_top_angle(conds)
        return np.sqrt(conds.l/conds.g) * ellipk(np.sin(t0/2)**2)
    elif behavior == 'supercritical':
        w0 = get_bottom_velocity(conds)
        return (2/w0) * ellipk(4*conds.g/(conds.l * w0**2))
    return None
    