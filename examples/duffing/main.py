# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os
import torch
import numpy as np
from depikt import Koopman
from depikt.utils.lr_schedule import ExponentialDecay
from depikt.utils import gen_traj

try:
    from duffing import dynamics
except ImportError:
    from examples.duffing.duffing import dynamics

# import optuna
# import logging
# import pickle

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

np.set_printoptions(precision=3, suppress=True)

cwd = os.getcwd()
if "examples/duffing" not in cwd:
    os.chdir("./examples/duffing")


def sample(config_path):

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    for key in config.keys():
        gen_traj(dynamics=dynamics, **config[key])

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
    exp_decay = ExponentialDecay(fit_args['num_epochs'], decay_rate=2.3)
    def step_decay(epoch): return 1.0 if epoch < 30000 else 0.2

    val_loss = dkr.fit(**fit_args, plotext=True,
                       validate=True, lr_lambda=step_decay)

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


if __name__ == "__main__":

    sample("config/system.yaml")
    # fit("config/standard.yaml")
