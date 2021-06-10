from dataclasses import dataclass
import numpy as np

@dataclass
class Conditions:
    '''
    A dataclass to house the conditions of a pendulum problem
    '''
    theta0: float = np.deg2rad(170) #intial angle
    omega0: float = 0 #intial angular velocity
    l: float = 20 #length of the pendulum string
    g: float = 10 #gravitational field
    