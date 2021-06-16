import conservation_of_energy
import numpy as np

BISECT = 100
NEWTON = 10

def standardize(angle):
    '''
    brings into [-pi, pi]. From stackoverflow
    '''
    return -((np.pi - angle) % (2*np.pi)) + np.pi

def get_subcrit_target(conds):
    return np.abs(standardize(conds.theta0))

def get_supercrit_target(conds):
    return -np.sign(conds.omega0) * np.abs(standardize(conds.theta0))

def get_shift(conds, approx, deriv):
    behavior = conservation_of_energy.get_behavior(conds)
    #handle the critical case. As a low-order approximation, far and away the
    #easiest thing is just to use the exact shift even for the series
    if behavior == 'critical':
        w0 = np.sqrt(conds.g/conds.l)
        return np.arctanh(np.sin(.5*conds.theta0*np.sign(conds.omega0)))/w0
    #make sure it need solving. 
    if behavior == 'supercritical' and np.isclose(conds.theta0 % np.pi, 0):
        return 0
    if behavior == 'subcritical' and np.isclose(conds.omega0, 0):
        return 0
    #handle the other cases
    theta_targ = get_subcrit_target(conds) if behavior == 'subcritical' else get_supercrit_target(conds)
    print('theta_targ: ' + str(theta_targ))
    Ts = conservation_of_energy.get_T_star(conds)
    #write the function to find the zero of
    f = lambda x : approx(x) - theta_targ
    #Handle the case so the Newton's method doesn't approach a critical point
    flag = False
    if np.abs(conds.omega0) < .05:
        bisect = BISECT + 4 * NEWTON
        newton = 0
    else: 
        bisect = BISECT
        newton = NEWTON
    #use the bisection method to get a starting guess
    decreasing = behavior == 'subcritical' or conds.omega0 < 0
    interv = (0, Ts)
    mid = Ts/2
    for _ in range(bisect):
        mid = .5*(interv[0]+interv[1])
        if _ % 10 == 0: print(approx(mid))
        res = f(mid)
        if res == 0:
            flag = True
            break
        elif res > 0 and decreasing or res < 0 and not decreasing:
            interv = (mid, interv[1])
        else:
            interv = (interv[0], mid)
    #Neewton's method to get the solution
    xc = mid
    print(xc)
    for _ in range(newton):
        if flag:
            break
        #because we're so careful bisecting, we don't really need to error check
        xc -= f(xc) / deriv(xc)    
    #now we use some logic to turn this into an actual shift
    if behavior == 'supercritical':
        if np.sign(standardize(conds.theta0)) == np.sign(standardize(approx(xc))):
            return xc
        else:
            return (2*Ts - xc)
    if behavior == 'subcritical':
        if np.sign(standardize(conds.theta0)) == np.sign(standardize(approx(xc))):
            if np.sign(conds.omega0) == np.sign(deriv(xc)):
                return xc
            else:
                return 4*Ts - xc
        else:
            if np.sign(conds.omega0) == np.sign(deriv(xc)):
                return 2*Ts - xc
            else:
                return 2*Ts + xc
    return None

def shift(conds, approx, deriv):
    shift = get_shift(conds, approx, deriv)
    return lambda x : approx(x + shift)    