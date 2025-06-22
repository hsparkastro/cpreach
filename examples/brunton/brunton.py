import numpy as np


def dynamics(t, x, u, p):

    x0 = x[..., 0]
    x1 = x[..., 1]
    u = u[..., 0]
    ld = p[..., 0]
    mu = p[..., 1]

    xdot = np.stack(
        [-mu * x0, -ld * (x1 - x0**2 + u)], axis=-1)

    return xdot
