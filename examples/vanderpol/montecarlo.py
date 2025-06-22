# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os
import torch
import numpy as np
from depikt import Koopman
from depikt.utils import load_dataset
import random

# import optuna
# import logging
# import pickle

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

np.set_printoptions(precision=3, suppress=True)

cwd = os.getcwd()
if "examples/vanderpol" not in cwd:
    os.chdir("./examples/vanderpol")


def mc_standard():

    nsamples_list = [10, 100, 300, 1000, 2000, 5000, 10000]

    def mc_sim(nsamples):
        config_path = "config/standard.yaml"
        # Load NN configuration and training options
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        model_args = config["model_args"]
        fit_args = config["fit_args"]
        fit_args.pop("trainset")
        fit_args.pop("valset")
        fit_args.pop("savepath")

        # Initialize model
        dkr = Koopman.from_args(**model_args, device=device)

        trainset = "data/randomtraj_p0.pkl.train"
        trainset = load_dataset(trainset, device=device)
        nsamples_total = trainset[0].shape[0]
        valset = "data/randomtraj_p0.pkl.val"
        valset = load_dataset(valset, device=device, numpy=True)

        rmse = []
        for k in range(10):
            ind = random.sample(range(nsamples_total), nsamples)
            # select BS number of samples randomly at each epoch

            trainset_k = [data[ind] for data in trainset]
            save_path_k = f"save/mc/standard_{nsamples}_{k}.pt"
            dkr.fit(trainset=trainset_k, savepath=save_path_k,
                    **fit_args, verbose=1)

            x_pred = dkr.predict(valset[0][:, 0], valset[1][:, :1])[:, 0]
            err = (valset[0][:, 1] - x_pred)
            rmse_k = (np.sum(err**2) / nsamples) ** 0.5
            print("RMS  Err. = ", rmse_k)
            rmse.append(rmse_k)

        return rmse_k, dkr

    rmse_list = []
    for nsamples in nsamples_list:

        rmse, dkr = mc_sim(nsamples)
        mean_rmse = rmse.mean()
        rmse_list.append(rmse)
        print(f"{nsamples} samples mean RMSE: {mean_rmse}")


def mc_affine():
    pass


def mc_results_standard(nsamples_list):
    config_path = "config/standard.yaml"
    # Load NN configuration and training options
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_args = config["model_args"]

    valset = "data/randomtraj_p0.pkl.val"
    valset = load_dataset(valset, device=device, numpy=True)

    # Initialize model
    dkr = Koopman.from_args(**model_args, device=device)

    rmse_mean = []
    rmse_std = []
    for nsamples in nsamples_list:

        rmse_nsamples = []
        for k in range(10):
            save_path_k = f"save/mc/standard_{nsamples}_{k}.pt"
            dkr.load(save_path_k)
            x_pred = dkr.predict(valset[0][:, 0], valset[1][:, :1])[:, 0]
            err = (valset[0][:, 1] - x_pred)
            rmse_k = (np.sum(err**2) / nsamples) ** 0.5
            rmse_nsamples.append(rmse_k)
        rmse_mean.append(np.mean(rmse_nsamples))
        rmse_std.append(np.std(rmse_nsamples))

    return rmse_mean, rmse_std


def mc_results_affine(nsamples_list):

    config_path = "config/affine.yaml"
    # Load NN configuration and training options
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_args = config["model_args"]
    filepath = config["fit_args"]["savepath"]

    dkr = Koopman.from_args(**model_args, device=device, filepath=filepath)

    trainset = "data/randomtraj_p0.pkl.train"
    trainset = load_dataset(trainset, device=device)
    nsamples_total = trainset[0].shape[0]

    valset = "data/randomtraj_p0.pkl.val"
    valset = load_dataset(valset, device=device, numpy=True)

    rmse_mean = []
    rmse_std = []
    for nsamples in nsamples_list:
        rmse_nsamples = []

        for k in range(10):
            ind = random.sample(range(nsamples_total), nsamples)
            # select BS number of samples randomly at each epoch

            trainset_k = [data[ind] for data in trainset]
            dkr.finetune(trainset=trainset_k)

            x_pred = dkr.predict(valset[0][:, 0], valset[1][:, :1])[:, 0]
            err = (valset[0][:, 1] - x_pred)
            rmse_k = (np.sum(err**2) / nsamples) ** 0.5
            rmse_nsamples.append(rmse_k)

        rmse_mean.append(np.mean(rmse_nsamples))
        rmse_std.append(np.std(rmse_nsamples))

    return rmse_mean, rmse_std


def plot_results():

    import matplotlib.pyplot as plt

    nsamples_list = [10, 100, 300, 1000, 2000, 5000, 10000]
    rmse_mean, rmse_std = mc_results_standard(nsamples_list)
    rmse_mean_affine, rmse_std_affine = mc_results_affine(nsamples_list)

    fig, ax = plt.subplots()
    ax.loglog(nsamples_list, rmse_mean, marker="o", label="Mean RMSE")
    ax.fill_between(nsamples_list, np.array(rmse_mean) - np.array(rmse_std),
                    np.array(rmse_mean) + np.array(rmse_std), alpha=0.2)

    ax.loglog(nsamples_list, rmse_mean_affine,
              marker="o", label="Mean RMSE Affine")
    ax.fill_between(nsamples_list,
                    np.array(rmse_mean_affine) - np.array(rmse_std_affine),
                    np.array(rmse_mean_affine) + np.array(rmse_std_affine),
                    alpha=0.2)

    ax.set_xlabel("Number of samples")
    ax.set_ylabel("RMSE")
    plt.savefig("figures/vanderpol_mc.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # mc_standard()
    plot_results()
