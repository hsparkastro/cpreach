#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from collections import OrderedDict


class SNMLP(nn.Module):
    """
    A feedforward neural network with spectral normalization applied to
    all linear layers. Spectral normalization ensures that the Lipschitz
    constant of the network is bounded by 1. During training, after each
    backward pass, the weight matrix is divided by the spectral norm (largest
    singular value) of the weight matrix.
    """

    def __init__(self, nndims, normalize_hidden=True, normalize_output=False,
                 activation="relu", skip=False):
        """
        Parameters
        ----------
        nndims: list of int
            The dimensions of the network. The first element is the input
            dimension and the last element is the output dimension.
        normalize_hidden: bool, optional
            Whether to apply spectral normalization to the hidden layers.
        normalize_output: bool, optional
            Whether to apply spectral normalization to the output layer.
        activation: str, optional
            The activation function to use. Can be "relu", "tanh", or "elu".
            Default is "relu".
        """

        super(SNMLP, self).__init__()

        layers = []
        for i in range(len(nndims) - 2):
            if normalize_hidden:
                layers.append((
                    f"linear_{i}",
                    spectral_norm(nn.Linear(nndims[i], nndims[i + 1])),
                ))
            else:
                layers.append((
                    f"linear_{i}",
                    nn.Linear(nndims[i], nndims[i + 1])
                ))
            if activation == "relu":
                layers.append((f"relu_{i}", nn.ReLU(inplace=True)))
            elif activation == "tanh":
                layers.append((f"tanh_{i}", nn.Tanh()))
            elif activation == "elu":
                layers.append((f"elu_{i}", nn.ELU()))

        last_dim = nndims[-1] - nndims[0] if skip else nndims[-1]
        if normalize_output:
            layers.append((
                f"linear_{(len(nndims) - 2)}",
                spectral_norm(nn.Linear(nndims[-2], last_dim)),
            ))
        else:
            layers.append((
                f"linear_{(len(nndims) - 2)}",
                nn.Linear(nndims[-2], last_dim)
            ))
        self.model = nn.Sequential(OrderedDict(layers))
        self.nndims = nndims
        self.skip = skip

    def forward(self, x):
        if not self.skip:
            # if skip is False, just return the output of the model
            return self.model(x)
        else:
            return torch.cat((x, self.model(x)), dim=-1)

    def reset_parameters(self):
        for layer in self.model[::2]:
            layer.reset_parameters()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def __str__(self):
        return "SNMLP(" + str(self.nndims) + ")"
