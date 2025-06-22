import torch
import numpy as np


def dynamics(t, x, param):
    """First dimension is the batch, second is the dimension of the values"""

    delta, alpha, beta = param.T  # damping, linear, nonlinear
    y, ydot = x.T

    yddot = -delta * ydot - alpha * (y + beta * y**3)

    if isinstance(x, np.ndarray):
        xddot = np.vstack((ydot, yddot)).T
    else:
        xddot = torch.vstack((ydot, yddot)).T

    return xddot
