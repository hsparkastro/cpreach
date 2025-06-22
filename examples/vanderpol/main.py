# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os
import torch
import numpy as np
from depikt import Koopman
from depikt.utils import gen_traj
import matplotlib.pyplot as plt
try:
    from vanderpol import dynamics
except ImportError:
    from examples.vanderpol.vanderpol import dynamics


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

np.set_printoptions(precision=3, suppress=True)

cwd = os.getcwd()
if "examples/vanderpol" not in cwd:
    os.chdir("./examples/vanderpol")


def sample(config_path):

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    for key in config.keys():
        trajset = gen_traj(dynamics=dynamics, **config[key], vectorized=True)

        # plt.plot(trajset["x"][:100, :, 0].T, trajset["x"][:100, :, 1].T)
        # plt.show()

    print("="*50)
    print()


def fit(config_path):

    # Load NN configuration and training options
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_args = config["model_args"]
    fit_args = config["fit_args"]

    # Initialize model
    dkr = Koopman.from_args(**model_args, device=device)

    # Fit model
    val_loss = dkr.fit(**fit_args, plotext=True, validate=True)
    # NOTE: calling dkr.fit twice with plotext=True twice will cause an error
    # This is a bug in the plotext library.

    # # Example of using a learning rate schedule
    # from pkoopman.utils.lr_schedule import exp_decay
    # num_epochs = fit_args["num_epochs"]
    # lr_lambda = exp_decay(num_epochs, decay_rate=3.0)
    # val_loss = dkr.fit(
    #     **fit_args, lr_lambda=lr_lambda, plotext=False, validate=True
    # )

    print("="*50)
    print()

    return val_loss


def finetune(config_path):

    # Load NN configuration and training options
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_args = config["model_args"]
    fit_args = config["fit_args"]
    finetune_args = config["finetune_args"][0]
    filepath = fit_args["savepath"]

    """Initialize model"""
    dkr = Koopman.from_args(**model_args, filepath=filepath, device=device)

    """Finetune model"""

    val_loss = dkr.finetune(**finetune_args)

    print("="*50)
    print()

    return val_loss


def test(dkr, valset):

    from depikt.utils import load_dataset

    valset = load_dataset(valset, device=device, numpy=True)
    x = valset[0][:, 0]
    u = valset[1][:, :1]
    y = valset[0][:, 1]
    x_pred = dkr.predict(x, u)[:, 0]

    err = (y - x_pred)

    mean = np.mean(err, axis=0)
    var = np.diag(np.cov(err.T))
    rmse = (mean**2 + var) ** 0.5
    maxerr = np.max(np.abs(err), axis=0)

    print("Mean Err. = ", mean)
    print("Max  Err. = ", maxerr)
    print("RMS  Err. = ", rmse)
    print()

    print("="*50)
    print()


def control(config_path):

    from control import dlqr
    import matplotlib.pyplot as plt
    from tqdm import trange

    from vanderpol import dynamics
    from depikt.utils import RK45_step

    # Load NN configuration and training options
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_args = config["model_args"]

    if model_args["model_type"] == "standard":
        filepath = config["fit_args"]["savepath"]
    else:
        filepath = config["finetune_args"][0]["savepath"]

    """Initialize model"""
    dkr = Koopman.from_args(**model_args, device=device,
                            filepath=filepath)

    A0 = dkr.get("A")
    B0 = dkr.get("B")
    Cx0 = dkr.get("Cx")
    Cu0 = dkr.get("Cu")

    """ setup controller"""
    Q0 = np.diag([1.0, 1.0])
    R = np.array([[0.01]])

    K_koopman = dlqr(A0, B0, Cx0.T@Q0@Cx0, Cu0.T@R@Cu0)[0]

    T = 5.0
    dt = 0.01
    K = int(T / dt)
    Tpred = 10

    time = np.arange(K + 1) * dt
    state0 = np.array([-2.0, 2.0])
    state_est = np.zeros((K + 1, 2))
    state = np.zeros((K + 1, 2))
    control = np.zeros((K + 1, 1))
    state[0] = state0
    state_est[0] = state0

    for t in trange(K, ncols=80):

        control[t] = -K_koopman @ dkr.lift(state[t])
        # control[t] = np.clip(control[t], -1.0, 1.0)

        # option1: solve_ivp
        state[t + 1] = RK45_step(
            dynamics, 0, dt, state[t], args=(control[t], np.array([1.0]))
        )

        state_est[t + 1] = dkr.predict(state_est[t], control[t])

        # option2: model prediction
        # state[t + 1] = predict(state[t], control[t])

        # option3: linearized system prediction
        # state[t + 1] = Alin_discrete @ state[t] + Blin_discrete @ control[t]

        tstart = max(0, t - Tpred)
        est = state[tstart].copy()
        for k in range(tstart, t + 1):
            est = dkr.predict(est, control[k])
        state_est[t + 1] = est

        # state_est[t + 1] = dkr.predict(state_est[t], control[t])

    control[-1] = -K_koopman @ dkr.lift(state[-1])
    control[-1] = np.clip(control[-1], -10.0, 10.0)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time, state[:, 0], label='True State 1')
    axs[0].axhline(y=0, color='dimgray', linewidth=0.8)
    axs[0].set_ylabel('State 1')
    axs[0].legend()

    axs[1].plot(time, state[:, 1], label='True State 2')
    axs[1].axhline(y=0, color='dimgray', linewidth=0.8)
    axs[1].set_ylabel('State 2')
    axs[1].legend()

    axs[2].plot(time, control, label='Control Input')
    axs[2].axhline(y=0, color='dimgray', linewidth=0.8)
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Control')
    axs[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    sample("config/system.yaml")
    # fit("config/standard.yaml")
