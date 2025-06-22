#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library Name: conformal_prediction.py
Description: Perform conformal prediction and produce bounds
Author: Hyunsang Park (email: park1375@purdue.edu)
"""

import numpy as np


def conformal_prediction(err, alpha):

    nsample = err.shape[0]
    l = min(int(np.ceil((nsample+1)*(1-alpha))), nsample) - 1

    err = np.abs(err)
    bound = np.zeros(err.shape[1])
    for k in range(err.shape[1]):
        sorted = np.sort(err[:, k])
        bound[k] = sorted[l]

    return bound


if __name__ == "__main__":
    nsample = 1000
    y = np.random.randn(nsample, 3)
    yhat = y + np.random.randn(nsample, 3)*0.1
    alpha = 0.01

    bound = conformal_prediction(y, yhat, alpha)
    print(bound)
