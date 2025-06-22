# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os
import torch
import numpy as np
from depikt import Koopman
from depikt.utils.lr_schedule import ExponentialDecay
from depikt.utils import gen_traj, load_dataset

try:
    from brunton import dynamics
except ImportError:
    from examples.brunton.brunton import dynamics

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

cwd = os.getcwd()
if "examples/brunton" not in cwd:
    os.chdir("./examples/brunton")

np.set_printoptions(formatter={"float": "{: .3E}".format})


def sample(config_path):

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    for key in config.keys():
        gen_traj(dynamics=dynamics, **config[key], vectorized=True)

    print()


def fit(config_path):

    # Load NN configuration and training options
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_args = config["model_args"]
    fit_args = config["fit_args"]

    dkr = Koopman.from_args(**model_args, device=device)

    # Fit model
    lr_lambda = ExponentialDecay(fit_args['num_epochs'], decay_rate=4.0)
    dkr.fit(**fit_args, plotext=True, validate=True, lr_lambda=lr_lambda)


def test(config_path):

    # Load NN configuration and training options
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_args = config["model_args"]
    fit_args = config["fit_args"]
    valset = fit_args["valset"]

    dkr = Koopman.from_args(**model_args, device=device)
    dkr.load(fit_args["savepath"])

    valset = load_dataset(valset, device=device, numpy=True)
    x0 = valset['x'][:, 0]
    x = valset['x'][:, :-1]
    u = valset['u']
    y = valset['x'][:, 1:]
    y_pred = dkr.predict(x0, u)

    change = np.mean(np.abs(y - x), axis=(0, 1))
    err = (y - y_pred)/change
    err = err.reshape(-1, err.shape[-1])

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
    # sample("config/system.yaml")
    fit("config/standard.yaml")
    test("config/standard.yaml")
