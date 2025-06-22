#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from .koopman import Koopman


class StandardKoopman(Koopman):
    """
    Standard Koopman model (autonomous and linear input)
    """

    def __init__(self, state, ninput=None, normalization=dict(),
                 filepath=None, device=torch.device("cpu"), custom_basis=None):
        """
        Initialize the Koopman class.
        Parameters:
        -----------
        state : dict of
            type: choice of basis functions "snmlp", "monomial", "custom", or
                "none". If "custom", the custom_basis function should be 
                provided. If "none", state is not lifted (DMD-like).
            args: arguments to pass to the basis function.
        ninput : int or None
            Number of inputs. 0 or None means no input.
        normalization : dict or None, optional
            Normalization constants for the parameters, states, inputs, and/or
            time step. A dictionary containing 'x_range', 'u_range', and/or
            'dt'.
            normalization['x_range']: list or None, optional
                list of tuples containing the min and max values for each state
                ex) For 3 states, normalization['x_range'] =
                    [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
                If not provided, the default is (-1.0, 1.0).
            normalization['u_range']: list or None, optional
                list of tuples containing the min and max values for each input
                ex) For 2 inputs, normalization['u_range'] =
                    [(-1.0, 1.0), (-1.0, 1.0)]
                If not provided, the default is (-1.0, 1.0).
            normalization['dt']: float or None, optional
                the time step. Used to scale the loss function to be in terms
                of xdot.
                ex) normalization['dt'] = 0.01
                If not provided, the default is 1.0.
        filepath : str, optional
            If the filepath is provided, the parameters are loaded from the
            file.
        device : torch.device, optional
            The device to run the model on. If None, device is set to 'cpu'.
        """

        super(StandardKoopman, self).__init__(
            state, ninput, device, custom_basis)

        # set normalization
        self.normalization = self.set_normalization(normalization)

        # set matrices
        self.A = torch.nn.Parameter(
            torch.empty((self.NLIFT, self.NLIFT)), requires_grad=True)

        if self.NINPUT is not None:
            self.B = torch.nn.Parameter(
                torch.empty((self.NLIFT, self.NINPUT)), requires_grad=True)

        # recovers the original state from normalized state.
        # x = Cx * x_normalized + dx
        self.Cx = torch.nn.Parameter(
            torch.hstack((
                torch.diag(self.normalization["x_range"]["width"]),
                torch.zeros((self.NSTATE, self.NLIFT - self.NSTATE)),
            )), requires_grad=False,
        )
        self.dx = self.normalization["x_range"]["center"]

        # recovers the original input from normalized input.
        # u = Cu * u_normalized + du
        if self.NINPUT is not None:
            self.Cu = torch.diag(self.normalization["u_range"]["width"])
            self.du = self.normalization["u_range"]["center"]

        if filepath is not None:
            self.load(filepath)

    def _predict(self, x0, u):

        # x0.shape = (batch_size, nstate)
        # u.shape = (batch_size, nstep, ninput). ninput can be zero.

        batch_size, nstep = u.shape[:2]
        xlift = self.basis(x0)

        # # u.shape = (batch_size, nstep, nlift)
        xlift_pred_traj = torch.empty(
            (batch_size, nstep, self.NLIFT), device=self.device)

        for step in range(nstep):

            if self.NINPUT is None:
                xlift_next = xlift + xlift @ self.A.T
            else:
                xlift_next = xlift + \
                    xlift @ self.A.T + u[:, step] @ self.B.T

            xlift_pred_traj[:, step] = xlift_next
            xlift = xlift_next

        return xlift_pred_traj

    def _fit_lstsq(self, trainset, valset=None, savepath=None, verbose=True,
                   **kwargs):

        trainset, _ = self.load_dataset(trainset, 1, unfold=True)

        xset = trainset['x'][:, 0]
        yset = trainset['x'][:, 1]
        with torch.no_grad():
            xlift = self.basis(xset)
            ylift = self.basis(yset)
        nsample = xset.shape[0]

        if self.NINPUT is not None:
            uset = trainset['u'][:, 0]
            W = torch.cat((xlift, uset), axis=1)/nsample
        else:
            W = xlift/nsample

        Wn = (ylift - xlift)/nsample
        M = torch.linalg.inv(W.T @ W) @ (W.T @ Wn)

        self.A = torch.nn.Parameter(
            M[: self.NLIFT, :].T, requires_grad=False)
        if self.NINPUT is not None:
            self.B = torch.nn.Parameter(
                M[self.NLIFT:, :].T, requires_grad=False)

        with torch.no_grad():
            loss = self.loss(
                trainset, state_only=True,
                invariance=False, l2=False)[0].item()

        if verbose:
            output = (
                f"TRAINED [{self.__class__.__name__}]:\n"
                + f"num. samples: {nsample}, t-loss = {loss:.3E}")
            if valset is not None:
                valset, _ = self.load_dataset(valset, 1, unfold=True)
                with torch.no_grad():
                    loss_val = self.loss(
                        valset, state_only=True,
                        invariance=False, l2=False)[0].item()

                output += f", v-loss = {loss_val:.3E}"

            print(output)

        self.save(savepath, verbose)
