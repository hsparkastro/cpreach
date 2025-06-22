import numpy as np
from tqdm import trange
from reach.zonotope import Zonotope
# from numba import njit
from scipy.spatial.distance import cdist
from typing import Union, Callable

# @njit
# def compute_gradients_numba(xhist_next, bandwidth):
#     N, D = xhist_next.shape
#     gradients = np.zeros((N, D), dtype=np.float32)
#     weight_sums = np.zeros(N, dtype=np.float32)

#     for i in range(N):
#         for j in range(i+1, N):
#             dist_sq = 0.0
#             for d in range(D):
#                 diff = xhist_next[j, d] - xhist_next[i, d]
#                 dist_sq += diff * diff

#             weight = np.exp(-dist_sq / (2 * bandwidth**2))

#             for d in range(D):
#                 diff = xhist_next[j, d] - xhist_next[i, d]
#                 gradients[i, d] += weight * diff
#                 gradients[j, d] -= weight * diff

#             weight_sums[i] += weight
#             weight_sums[j] += weight

#     for i in range(N):
#         for d in range(D):
#             gradients[i, d] /= weight_sums[i]

#     return gradients


def compute_gradients(xhist_next, bandwidth=0.5):

    dists = cdist(xhist_next, xhist_next)  # (N, N)

    # Compute weights
    weights = np.exp(-dists**2 / (2 * bandwidth**2))
    weights /= np.sum(weights, axis=1, keepdims=True)

    # Compute gradients directly without storing full err
    gradients = (weights @ xhist_next) - xhist_next
    return gradients


def approximate_RFRS(
        initset: Zonotope, inputset: Zonotope, distset: np.ndarray,
        sys_dt, T=100, nsamples=50):

    raise NotImplementedError(
        "approximate_RFRS is not implemented yet."
    )

    distset = distset.astype(np.float32)
    u = inputset.rand.farthest(nsamples).astype(np.float32)
    x0 = initset.rand.farthest(nsamples).astype(np.float32)
    u[0] = inputset.center  # first is the mean trajectory.
    x0[0] = initset.center

    x = np.zeros((T+1,) + x0.shape, dtype=np.float32)
    x[0] = x0

    for i in trange(T, ncols=80, desc='Calculating FRS'):

        xhist_next = sys_dt(x[i], u)

        gradients = compute_gradients(xhist_next, bandwidth=0.5)
        # gradients = compute_gradients_numba(xhist_next, bandwidth=0.5)

        # normalize such that one element is +-1
        denom = np.max(np.abs(gradients), axis=1, keepdims=True)
        gradients /= np.maximum(denom, 1e-8)

        xhist_next += gradients * distset
        x[i+1] = xhist_next.copy()

    return x


def approximate_FRS(
        initset: Zonotope, inputset: Union[Zonotope, Callable],
        sys_dt, T=100, nsamples=50, constant_input=False):

    x0 = initset.rand.uniform(nsamples)

    if constant_input:
        if isinstance(inputset, Zonotope):
            u = inputset.rand.uniform(nsamples)
        elif callable(inputset):
            u = []
            for j in range(nsamples):
                inputset_j = inputset(x0[j])
                u.append(inputset_j.rand.uniform(1))
            u = np.array(u).squeeze()

    x = np.zeros((T+1,) + x0.shape)
    x[0] = x0

    for i in trange(T, ncols=80, desc='Calculating FRS'):
        if not constant_input:
            if isinstance(inputset, Zonotope):
                u = inputset.rand.uniform(nsamples)
            elif callable(inputset):
                u = []
                for j in range(nsamples):
                    inputset_j = inputset(x[i, j])
                    u.append(inputset_j.rand.uniform(1))
                u = np.array(u).squeeze()

        x[i+1] = sys_dt(x[i], u)

    return x
