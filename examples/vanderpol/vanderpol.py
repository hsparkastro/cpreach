import numpy as np


def dynamics(t, x, mu):
    """First dimension is the batch, second is the dimension of the values"""

    y, ydot = x.T
    mu = mu.squeeze()

    yddot = mu * (1 - y**2) * ydot - y

    if mu.shape == ():
        xddot = np.array([ydot, yddot])
    else:
        xddot = np.stack((ydot, yddot), axis=-1)
    return xddot
