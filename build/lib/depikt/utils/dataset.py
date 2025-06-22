#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
from tqdm import trange
import torch
from copy import deepcopy


def load_dataset(dataset, device=torch.device("cpu"), numpy=False):
    if isinstance(dataset, str):
        with open(dataset, "rb") as file:
            dataset = pickle.load(file)
    else:
        dataset = deepcopy(dataset)
    if numpy:
        if isinstance(dataset["x"], torch.Tensor):
            dataset = {
                key: dataset[key].cpu().numpy()
                for key in dataset.keys()}
    else:
        if isinstance(dataset["x"], np.ndarray):
            dataset = {
                key: torch.tensor(
                    dataset[key], device=device, dtype=torch.float32)
                for key in dataset.keys()}
        else:
            dataset = {
                key: dataset[key].to(device)
                for key in dataset.keys()}

    return dataset


def gen_traj(
        dynamics, x_range, dt, nsample, nstep,
        u_range=None, p_range=None, param=None,
        filepath=None, seed=None, constant_input=False, distribution="uniform",
        controller=None, vectorized=False, rksteps=1):
    """
    Generate trajectories from a dynamical system. 

    Key Parameters
    ----------
    dynamics : callable
        The dynamics function of the system. It should take the form
        dynamics(t, x, u, p) or dynamics(t, x, p), where t is the time, x is 
        the state, u is the input (optional), and p is the parameter (required)
        of the system.
        x_range : array-like
        The range of the state variables. It should be a 2D array with shape
        (nstate, 2), where nstate is the number of state variables. The first
        column is the lower bound and the second column is the upper bound.
    u_range : array-like, optional
        Not required if the system has no input.
    p_range : array-like, optional
        Range of the parameters. Not required if 'param' is provided.
    param : array-like, optional
        Parameters of the system. Not required if 'p_range' is provided.
    """

    rng = np.random.default_rng(seed)

    def random_uniform(range, dims):
        var_dim = range.shape[0]
        samples = rng.random(dims + (var_dim,))
        samples = samples * (range[:, 1] - range[:, 0]) + range[:, 0]
        return samples

    def random_normal(range, dims):

        var_dim = range.shape[0]

        # Gaussian mixture with 2 modes, small and large variance
        # half_dims = (int(dims[0]/2),) + dims[1:]
        # rest = (dims[0] - int(dims[0]/2),) + dims[1:]
        # samples0 = rng.normal(size=half_dims + (var_dim,)).reshape(-1)/10.0
        # samples1 = rng.normal(size=rest + (var_dim,)).reshape(-1)
        # samples = np.concatenate((samples0, samples1), axis=0)

        samples = rng.normal(size=dims + (var_dim,)).reshape(-1)

        # Ensure samples are within the range [-1, 1]
        indices = np.where(np.abs(samples) > 1)[0]
        while indices.size > 0:
            samples[indices] = rng.normal(size=indices.size)
            indices = np.where(np.abs(samples) > 1)[0]

        samples = samples.reshape(dims + (var_dim,))
        samples = samples * (range[:, 1] - range[:, 0])/2
        samples += (range[:, 1] + range[:, 0])/2
        return samples

    random_functions = {
        "uniform": random_uniform,
        "normal": random_normal}
    random = random_functions[distribution]

    assert (p_range is not None and param is None) or \
        (p_range is None and param is not None), \
        "Either p_range or param must be provided, not both."

    if p_range is not None:
        p_range = np.array(p_range)
    if param is not None:
        param = np.array(param)

    x_range = np.array(x_range)
    if u_range is not None:
        u_range = np.array(u_range)

    x = np.empty((nsample, nstep + 1, x_range.shape[0]))
    x[:, 0] = random(x_range, (nsample,))

    if p_range is not None:
        p = random(p_range, (nsample,))
    else:
        p = np.repeat(param[None, :], nsample, axis=0)

    if controller is None:
        if u_range is not None:
            if not constant_input:
                u = random(u_range, (nsample, nstep))
            else:
                u = np.repeat(random(u_range, (nsample, 1)), nstep, axis=1)
    else:
        u = np.empty((nsample, nstep, u_range.shape[0]))

    if vectorized:
        pbar = trange(nstep, desc="Generating..., Timestep", ncols=80)
        for j in pbar:
            if u_range is not None:
                if controller is not None:
                    u[:, j] = controller(x[:, j], p, rng)
                x[:, j+1] = RK45(dynamics, 0, dt, x[:, j],
                                 args=(u[:, j], p), steps=rksteps)
            else:
                x[:, j+1] = RK45(dynamics, 0, dt, x[:, j],
                                 args=(p,), steps=rksteps)
    else:
        pbar = trange(
            nsample, desc="Generating..., Trajectory", ncols=rksteps)
        for i in pbar:
            for j in range(nstep):
                if u_range is not None:
                    if controller is not None:
                        u[i, j] = controller(x[i, j], p[i], rng)
                    x[i, j+1] = RK45(dynamics, 0, dt, x[i, j],
                                     args=(u[i, j], p[i]), steps=rksteps)
                else:
                    x[i, j+1] = RK45(dynamics, 0, dt, x[i, j],
                                     args=(p[i],), steps=rksteps)

    if u_range is not None:
        trainset = {"x": x, "u": u, "p": p}
    else:
        trainset = {"x": x, "p": p}

    if filepath is not None:
        with open(filepath, "wb") as file:
            pickle.dump(trainset, file)
        print(f"Trajectory data saved to {filepath}")

    return trainset


def RK45(dynamics, t0, dt, x0, args, steps=10):
    x_next = RK45_step(dynamics, t0, dt/steps, x0, args)
    t0 += dt/steps
    for _ in range(1, steps):
        x_next = RK45_step(dynamics, t0, dt/steps, x_next, args)
        t0 += dt/steps

    return x_next


def RK45_step(dynamics, t0, dt, x0, args):
    k1 = dynamics(t0, x0, *args)
    k2 = dynamics(t0 + 0.5 * dt, x0 + 0.5 * dt * k1, *args)
    k3 = dynamics(t0 + 0.5 * dt, x0 + 0.5 * dt * k2, *args)
    k4 = dynamics(t0 + dt, x0 + dt * k3, *args)
    x_next = x0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next
