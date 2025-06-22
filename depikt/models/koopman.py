#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from tqdm import tqdm
import os
from depikt import SNMLP, Monomials, Identity
from depikt.utils import load_dataset

try:
    import plotext as pltx
    plotext_available = True
except ImportError:
    plotext_available = False


class Koopman(nn.Module):
    def __init__(self, state, ninput=None, device=torch.device("cpu"),
                 custom_basis=None):

        super(Koopman, self).__init__()

        if 'args' in state:
            args = state['args']
        if state['lifting'] == 'snmlp':
            nstate, nlift = args['nndims'][0], args['nndims'][-1]
            self.basis = SNMLP(**args, skip=True)
            self.fit = self._fit
        elif state['lifting'] == 'monomial':
            self.basis = Monomials(**args)
            nstate = self.basis.nstate
            nlift = self.basis.nlift
            if self.basis.reduce:
                self.fit = self._fit
            else:
                self.fit = self._fit_lstsq
        elif state['lifting'] == 'custom':
            if 'args' in state:
                self.basis = custom_basis(**args)
            else:
                self.basis = custom_basis()
            nstate = self.basis.nstate
            nlift = self.basis.nlift
            self.fit = self._fit_lstsq
        elif state['lifting'] == 'none':
            self.basis = Identity(**args)
            nstate = self.basis.nstate
            nlift = nstate
            self.fit = self._fit_lstsq
        else:
            raise ValueError("Invalid basis.")

        assert nlift >= nstate, \
            "nlift must be larger than or equal to nstate."
        self.NSTATE = nstate
        self.NLIFT = nlift
        self.NINPUT = ninput

        # lifting function

        self.lossfn = nn.MSELoss()
        self.device = device
        self.to(device)

    @classmethod
    def from_args(cls, modeltype, **kwargs):
        """ Factory method to create a Koopman model of the given type."""

        if modeltype == "standard":
            from .standard import StandardKoopman
            return StandardKoopman(**kwargs)
        elif modeltype == "affine":
            from .affine import ParameterAffineKoopman
            return ParameterAffineKoopman(**kwargs)
        elif modeltype == "nonaffine":
            from .nonaffine import ParameterNonAffineKoopman
            return ParameterNonAffineKoopman(**kwargs)
        elif modeltype == "bilinear":
            from .bilinear import BilinearKoopman
            return BilinearKoopman(**kwargs)
        elif modeltype == "bilinearaffine":
            from .bilinearaffine import BilinearParameterAffineKoopman
            return BilinearParameterAffineKoopman(**kwargs)
        elif modeltype == "bilinearnonaffine":
            from .bilinearnonaffine import BilinearParameterNonAffineKoopman
            return BilinearParameterNonAffineKoopman(**kwargs)
        else:
            raise ValueError("Invalid modeltype.")

    def set_normalization(self, normalization):
        """
        Sets the normalization constants. Must be called after
        self.NSTATE, self.NINPUT, and self.NPARAM (when applicable) are set.
        """

        # set default configuration
        if "x_range" not in normalization:
            normalization["x_range"] = [(-1.0, 1.0)] * self.NSTATE
        if self.NINPUT is not None and "u_range" not in normalization:
            normalization["u_range"] = [(-1.0, 1.0)] * self.NINPUT
        if hasattr(self, "NPARAM") and "p_range" not in normalization:
            normalization["p_range"] = [(-1.0, 1.0)] * self.NPARAM
        if "dt" not in normalization:
            normalization["dt"] = 1.0

        def range2centerwidth(range):
            if range is None:
                return None
            range = torch.Tensor(range)
            min, max = range.T
            center = torch.nn.Parameter((max + min) / 2, requires_grad=False)
            width = torch.nn.Parameter((max - min) / 2, requires_grad=False)

            return nn.ParameterDict({"center": center, "width": width})

        normalization = normalization.copy()
        normalization["x_range"] = range2centerwidth(normalization["x_range"])
        if self.NINPUT is not None:
            normalization["u_range"] = range2centerwidth(
                normalization["u_range"])
        if hasattr(self, "NPARAM"):
            normalization["p_range"] = range2centerwidth(
                normalization["p_range"])
        normalization["dt"] = torch.nn.Parameter(
            torch.tensor(normalization["dt"]), requires_grad=False)

        return normalization

    def load_dataset(self, dataset, nstep=None, normalize=True, unfold=False,
                     unfold_max_steps=None):
        """
        Loads the dataset, selects the relevant data, and normalizes it.
        Dataset is a tuple with three elements: x, u, and p.
        Defined for all subclasses. Must be called after self.NSTATE,
        self.NINPUT, and self.NPARAM (when applicable) are set.
        """

        dataset = load_dataset(dataset, device=self.device)
        # x.shape = (nsample, nstep+1, nstate)
        # u.shape = (nsample, nstep, ninput)
        # p.shape = (nsample, nparam)

        # check dataset length and adjust nstep if necessary
        dataset_len = dataset["x"].shape[1] - 1
        if nstep is None:
            nstep = dataset_len
        elif nstep > dataset_len:
            print("WARNING: Dataset is shorter than the given nstep. " +
                  "Reducing nstep to length of dataset.")
            nstep = dataset_len

        if unfold_max_steps is None:
            unfold_max_steps = dataset_len
        elif unfold_max_steps > dataset_len:
            print("WARNING: Unfold max steps is larger than dataset length. " +
                  "Reducing unfold max steps to length of dataset.")
            unfold_max_steps = dataset_len

        x = dataset["x"]
        if normalize:
            x = self.normalize(x, self.normalization["x_range"])
        if not unfold:
            x = x[:, :nstep + 1]
        elif dataset_len > nstep:
            x = x[:, :unfold_max_steps + 1]
            x = x.unfold(dimension=1, size=nstep + 1, step=1).contiguous()
            x = x.permute(0, 1, 3, 2).contiguous()
            x = x.view(-1, nstep + 1, self.NSTATE)

        if self.NINPUT is not None:
            u = dataset["u"]
            if normalize:
                u = self.normalize(u, self.normalization["u_range"])
            if not unfold:
                u = u[:, :nstep]
            elif dataset_len > nstep:
                u = u[:, :unfold_max_steps]
                u = u.unfold(dimension=1, size=nstep, step=1).contiguous()
                u = u.permute(0, 1, 3, 2).contiguous()
                u = u.view(-1, nstep, self.NINPUT)

        if hasattr(self, "NPARAM"):
            p = dataset["p"]
            if normalize:
                p = self.normalize(p, self.normalization["p_range"])
            if unfold:
                p = p.repeat(dataset_len - nstep + 1, 1)

        if self.NINPUT is not None:
            if hasattr(self, "NPARAM"):
                dataset = {"x": x, "u": u, "p": p}
            else:
                dataset = {"x": x, "u": u}
        else:
            if hasattr(self, "NPARAM"):
                dataset = {"x": x, "p": p}
            else:
                dataset = {"x": x}

        return dataset, nstep

    def reset_parameters(self):
        """ Resets the parameters of the model. Defined for all subclasses. """
        if hasattr(self, "A") and self.A.requires_grad:
            # M = torch.randn_like(self.A)/self.NLIFT
            # U, _, V = torch.svd(M)
            # self.A = nn.Parameter(
            #     torch.mm(U, V.t())*0.9 - torch.eye(self.NLIFT),
            #     requires_grad=True)
            nn.init.normal_(self.A, mean=0, std=0.01)
        if hasattr(self, "B") and self.B.requires_grad:
            # nn.init.kaiming_uniform_(self.B, a=5.0**0.5)
            nn.init.normal_(self.B, mean=0, std=0.01)
        if hasattr(self, "As") and self.As.requires_grad:
            nn.init.normal_(self.As, mean=0, std=0.01)
        if hasattr(self, "Bs") and self.Bs.requires_grad:
            nn.init.normal_(self.Bs, mean=0, std=0.01)
        if isinstance(self.basis, SNMLP):
            self.basis.reset_parameters()
        if hasattr(self, "pbasis0") and isinstance(self.pbasis0, SNMLP):
            self.pbasis0.reset_parameters()

    def load(self, filepath, verbose=True):
        if os.path.exists(filepath):
            self.load_state_dict(
                torch.load(
                    filepath, map_location=self.device, weights_only=True
                )
            )
            if verbose:
                print(f"Loaded '{filepath}' to {self.__class__.__name__}")
        else:
            if verbose:
                print(f"Failed loading model. '{filepath}' does not exist!")

    def save(self, filepath, verbose=True):
        if filepath is not None:
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            torch.save(self.state_dict(), filepath)
            torch.save(self.basis.state_dict(),
                       filepath + ".basis")
            if verbose:
                print(f"Saved to '{filepath}'")
        else:
            if verbose:
                print("Filepath not provided! Model not saved.")

    def get(self, name, p=None):

        valid = name in ["A", "As", "B", "Bs", "Cx", "dx", "Cu", "du"]
        assert valid, "Invalid name"

        if p is not None and hasattr(self, "NPARAM"):  # parametric
            p = torch.tensor(
                p[None, :], device=self.device, dtype=torch.float32
            )
            p = self.normalize(p, self.normalization["p_range"])
            if hasattr(self, "pbasis0"):  # nonaffine
                plift = self.pbasis(p)[0]
            else:  # affine
                ones = torch.ones((p.shape[0], 1), device=self.device)
                plift = torch.cat((p, ones), dim=-1)[0]
            if hasattr(self, "param_type") and self.param_type == "linear":
                if name == "A":
                    A = self.As@plift + torch.eye(self.NLIFT)
                    return A.detach().numpy()
                elif name == "B":
                    B = self.Bs@plift
                    return B.detach().numpy()
            else:
                A = torch.einsum("abc,c->ab", self.As, plift)
                B = torch.einsum("aec,c->ae", self.Bs, plift)

                if self.param_type != "exponential":
                    top = torch.cat([A, B], dim=2)
                    bottom = torch.zeros(
                        (self.NINPUT, self.NINPUT + self.NLIFT),
                        dtype=A.dtype, device=A.device)
                    M = torch.cat([top, bottom], dim=1)
                    Md = torch.matrix_exp(M)
                    A = Md[:self.NLIFT, :self.NLIFT]
                    B = Md[:self.NLIFT, self.NLIFT:]

                if name == "A":
                    return A.detach().numpy()
                elif name == "B":
                    return B.detach().numpy()

        elif name == "A":
            A = self.A.detach().numpy().copy()
            if self.A.dim() == 3:
                A[:, :, -1] += np.eye(self.NLIFT)
            else:
                A += np.eye(self.NLIFT)
            return A
        return getattr(self, name).detach().numpy()

    def normalize(self, input, normalization):
        numpy = isinstance(input, np.ndarray)
        if input is None:  # no input
            return None
        if numpy:
            input = torch.tensor(input, device=self.device,
                                 dtype=torch.float32)

        normalized = (input - normalization["center"]) / normalization["width"]
        return normalized if not numpy else normalized.numpy()

    def unnormalize(self, input, normalization):
        numpy = isinstance(input, np.ndarray)
        if input is None:
            return None
        if numpy:
            input = torch.tensor(input, device=self.device,
                                 dtype=torch.float32)

        unnormalized = input * normalization["width"] + normalization["center"]
        return unnormalized if not numpy else unnormalized.numpy()

    def forward(self, *args):
        return self.predict(*args)

    def lift(self, x):
        """Handy implementation of the basis method for np.arrays.
        Supports 1D and 2D (batched) input. Assumes unnormalized input."""

        x = torch.Tensor(x)
        x = self.normalize(x, self.normalization["x_range"])
        onedim = len(x.shape) == 1

        if onedim:
            return self.basis(x=x[None, :]).detach().numpy()[0]
        return self.basis(x).detach().numpy()

    def predict(self, x0, u, p=None):
        """
        Predicts the next trajectory given the current state, and input
        trajectory. Assumes unnormalized input.
        Args:
            x (np.ndarray or torch.Tensor): The current state.
                Can be 1D (nstate,) or 2D (batch_size, nstate).
            u (np.ndarray or torch.Tensor, optional): The input trajectory.
                Can be 1D (ninput,), 2D (batch_size, ninput) or (nstep,
                ninput), or 3D (batch_size, nstep, ninput). For autonomous
                systems, an empty tensor with ninput=0 should be provided.
        Returns:
            np.ndarray or torch.Tensor: The predicted state or trajectory.
        """
        isnumpy = isinstance(x0, np.ndarray)
        if isnumpy:
            x0 = torch.tensor(x0, device=self.device, dtype=torch.float32)
        x0 = self.normalize(x0, self.normalization["x_range"])
        if isinstance(u, np.ndarray):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            if self.NINPUT is not None:
                u = self.normalize(u, self.normalization["u_range"])
        if hasattr(self, "NPARAM") and p is not None:
            if isinstance(p, np.ndarray):
                p = torch.tensor(p, device=self.device, dtype=torch.float32)
            p = self.normalize(p, self.normalization["p_range"])

        onesample = len(x0.shape) == 1
        onestep = len(u.shape) == len(x0.shape)

        if onesample:
            x0 = x0[None]
            if hasattr(self, "NPARAM") and p is not None:
                p = p[None]
            u = u[None]
        if onestep:
            u = u[:, None]

        if hasattr(self, "NPARAM") and p is not None:
            xlift_pred = self._predict(x0, u, p)
        else:
            xlift_pred = self._predict(x0, u)

        x_pred = xlift_pred @ self.Cx.T + self.dx

        if onestep:
            x_pred = x_pred[:, 0]
        if onesample:
            x_pred = x_pred[0]

        return x_pred.detach().numpy() if isnumpy else x_pred

    def _predict(self):
        """ Dummy method. To be implemented in subclasses. Predicts the nstep
        lifted states given the batch dataset. Assumes normalized input."""
        pass

    def loss(self, batch, decay=0.8, invariance=True, state_only=False,
             l2=True):
        """
        Computes the loss of the model given the batch.

        The loss is the sum of the MSE of the predicted lifted state and the
        true lifted state for each timestep, weighted by the decay factor.
        If state_only=True, the loss is computed only for the first NSTATE part
        of the lifted state (i.e., the unlifted state).

        Optionally computes the invariance loss and L2 regularization loss.
        - Invariance loss encourages the Koopman invariance, such that the
        predicted lifted state lies within the subspace spanned by the lifting
        function.
        - L2 loss is the L2 norm of the Koopman matrices.

        """

        # x.shape = (batch_size, nstep+1, nstate)
        # u.shape = (batch_size, nstep, ninput) or None
        # p.shape = (batch_size, nparam) or None

        x = batch['x']
        nstep = x.shape[1] - 1
        batch_size = x.shape[0]
        if self.NINPUT is not None:
            u = batch['u']
        else:
            u = torch.empty((batch_size, nstep, 0), device=self.device)

        if hasattr(self, "NPARAM") and "p" in batch:
            p = batch['p']
            xlift_pred = self._predict(x[:, 0], u, p)
        else:
            xlift_pred = self._predict(x[:, 0], u)

        xlift = self.basis(x)

        loss = 0.0
        invariance_loss = 0.0
        l2_loss = 0.0
        for step in range(nstep):
            if state_only:
                loss += decay**step * self.lossfn(
                    xlift_pred[:, step, :self.NSTATE],
                    xlift[:, step + 1, :self.NSTATE])
            else:
                loss += decay**step * self.lossfn(
                    xlift_pred[:, step], xlift[:, step + 1])
            if invariance:
                invariance_loss += self.lossfn(
                    xlift_pred[:, step],
                    self.basis(xlift_pred[:, step, :self.NSTATE]))

        denom = (1 - decay**nstep) / (1 - decay) if decay != 1.0 else 1.0
        loss = loss / denom
        invariance_loss = invariance_loss / denom

        if l2:
            if hasattr(self, "NPARAM"):
                l2_loss = self.As.pow(2).sum()
                if self.NINPUT is not None:
                    l2_loss += self.Bs.pow(2).sum()
            else:
                l2_loss = self.A.pow(2).sum()
                if self.NINPUT is not None:
                    l2_loss += self.B.pow(2).sum()

        return loss, invariance_loss, l2_loss

    def _fit(self, num_epochs, learning_rate, batch_size, trainset, nstep=1,
             decay=0.8, robust=None, l2_weight=0.0, invariance_weight=0.5,
             lr_lambda=None, valset=None, validate=True, plotext=False,
             savepath=None, save_intermediate=False, save_best=False,
             reset=True, verbose=2, cycle_all=False, unfold=False,
             unfold_max_steps=None, no_param=False, val_nstep=None, state_only=False):
        """
        Pre-trains the model using the provided training dataset.

        Parameters:
        -----------
        num_epochs : int
            Number of epochs to train the model.
        learning_rate : float
            Learning rate for the optimizer.
        batch_size : int
            Size of the mini-batches for training. If None, all samples are
            used.
        trainset : list of np.array, list of torch.tensor, or str
            Training dataset. The list should contain the following elements:
            xset: State trajectory. Shape: (nsample, nstep + 1, nstate)
            uset: Input, optional. Shape: (nsamples, nstep, ninput)
            If a string is provided, it is assumed to be a filepath to a
            pickled dataset.
        nstep : int, optional, default = 1
            Number of steps for multi-step prediction.
        decay : float, optional, default = 0.8
            Decay factor for the multi-step prediction error loss.
        robust : float or None, optional, default = None
            Noise level to add to the data for robustness.
        l2_weight : float, optional, default=0.0
            Weight for L2 regularization of the Koopman matrices.
        invariance_weight : float, optional, default=0.5
            Weight for the invariance loss.
        lr_lambda : function, optional
            Learning rate scheduler function. Default is None.
        valset : list of np.array, list of torch.tensor, or str, optional
            Validation dataset. The list should contain the following elements:
            xset: State trajectory. Shape: (nsample, nstep + 1, nstate)
            uset: Input, optional. Shape: (nsamples, nstep, ninput)
            If a string is provided, it is assumed to be a filepath to a
            pickled dataset.
        validate : bool, optional, default = True
            Whether to use the validation set during training.
            Ignored if the valset is not provided.
        plotext : bool, optional, default = False
            Whether to plot the training and validation loss. Ignored if
            plotext is not available.
        savepath : str, optional
            Filepath to save the trained model.
            The trained model is saved to the path if the filename is provided.
            Default is None.
        save_intermediate : bool, optional, default = False
            Whether to save intermediate models during training.
        reset : bool, optional, default = True
            Whether to reset the model parameters before training.
            For warmstarts.
        verbose : int or Boolean, optional, default = 2
            Verbosity level.

        Notes:
        ------
        - The method trains the model using the provided training dataset and
        optionally validates it using the validation dataset.
        - The training and validation loss history is printed during training.
        - If 'savepath' is provided, the trained model is saved to the
        specified filepath.
        - If 'plotext' is True, the training and validation loss is plotted
        after training.
        """

        # load datasets
        trainset_fn = trainset
        valset_fn = valset

        trainset, nstep = self.load_dataset(
            trainset, nstep, unfold=unfold,
            unfold_max_steps=unfold_max_steps)
        if no_param and hasattr(self, "NPARAM"):
            trainset.pop('p', None)
            self.basis.freeze()
            self.pbasis0.freeze()

        nsample = trainset['x'].shape[0]

        if valset is None:
            validate = False
        else:
            if val_nstep is None:
                val_nstep = nstep
            valset, _ = self.load_dataset(
                valset, nstep=val_nstep, unfold=unfold,
                unfold_max_steps=unfold_max_steps)
            if no_param and hasattr(self, "NPARAM"):
                valset.pop('p', None)

        # logging
        train_loss_hist = []
        if validate:
            val_loss = 0
            if save_best:
                best_val_loss = float("inf")
                best_epoch = 0
            val_loss_hist = []

        # setup parameters
        if reset:
            self.reset_parameters()
        if l2_weight is None:
            l2_weight = 0.0
        if robust is None:
            robust = 0.0

        l2_regularization = l2_weight > 0.0

        # setup optimizer and scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if lr_lambda is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )

        if cycle_all:
            trainset = TensorDataset(*trainset)
            loader = DataLoader(trainset, batch_size=batch_size,
                                shuffle=True, drop_last=True)

        if verbose:
            lr_lambda_desc = f"lr schedule: {lr_lambda},\n" \
                if lr_lambda is not None else ""

            print(
                f"TRAINING [{self.__class__.__name__}]:\n"
                + f"basis: {self.basis},\n"
                + f"trainset: {trainset_fn},\n"
                + f"valset: {valset_fn},\n"
                + lr_lambda_desc
                + f"num. samples: {nsample}, batch size: {batch_size}, "
                + f"num. steps: {nstep}, decay = {decay}\n"
                + f"lr: {learning_rate:.2E}, "
                + f"noise: {robust:.2E}, l2 weight: {l2_weight:.2E}, "
                + f"inv. weight: {invariance_weight:.2E}\n"
                + f"cycle all: {cycle_all}, unfold: {unfold}, "
                + f"unfold max steps: {unfold_max_steps}"
            )

        """ Training """
        pbar = tqdm(range(num_epochs), ncols=80, leave=True)
        self.train()
        for epoch in pbar:

            def do(batch):
                optimizer.zero_grad()
                train_loss, invariance_loss, l2_loss = self.loss(
                    batch, decay,
                    invariance=(invariance_weight != 0.0),
                    l2=(l2_weight != 0.0), state_only=state_only)
                loss = train_loss + invariance_weight * invariance_loss
                if l2_regularization:
                    loss += l2_weight * l2_loss
                loss_value = train_loss.item()
                pbar.set_postfix(loss=f'{loss_value:.2E}')
                loss.backward()
                optimizer.step()

                return loss_value

            if cycle_all:
                loss_value = 0.0
                for batch in loader:
                    loss_value += do(batch)
                loss_value /= len(loader)

            else:
                if batch_size is None or nsample <= batch_size:
                    # no minibatching. all samples in the batch
                    batch = trainset
                else:
                    # select batch_size samples randomly at each epoch
                    ind = random.sample(range(nsample), batch_size)
                    batch = {
                        key: trainset[key][ind] for key in trainset.keys()
                    }

                if robust != 0.0:
                    with torch.no_grad():
                        batch['x'] += torch.randn_like(batch['x']) * robust
                        batch['p'] += torch.randn_like(batch['p']) * robust

                loss_value = do(batch)
                pbar.set_postfix(loss=f'{loss_value:.2E}')

            if lr_lambda is not None:
                scheduler.step()

            message = (f"epoch: {epoch}, "
                       + f"t-loss: {loss_value:.2E}")

            if epoch % 100 == 99:
                train_loss_hist.append(loss_value)

            if validate and epoch % 100 == 99:
                self.eval()
                with torch.no_grad():
                    val_loss = self.loss(
                        valset, state_only=state_only,
                        invariance=False, l2=False)[0].item()
                self.train()
                val_loss_hist.append(val_loss)
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    self.save(savepath + ".best", verbose=False)

            validate_end = epoch == num_epochs - 1 and valset is not None
            if validate_end:
                self.eval()
                with torch.no_grad():
                    val_loss = self.loss(
                        valset, state_only=state_only,
                        invariance=False, l2=False)[0].item()
                self.train()

            if validate or validate_end:
                message += f", v-loss: {val_loss:.2E}"

            if save_best:
                message += f", best: {best_val_loss:.2E} at {best_epoch}"
            if verbose and epoch % 10 == 9:
                if epoch % 1000 == 9 and verbose == 2:
                    pbar.write(message)
                else:
                    pbar.write("\033[A\033[K" + message)
            elif epoch == num_epochs - 1 or epoch == 0:
                pbar.write(message)

            if save_intermediate and epoch % 1000 == 999:
                self.save(savepath + f".e{epoch}", verbose=False)

        self.save(savepath, verbose)

        if save_best:
            print(f"Best v-loss: {best_val_loss:.2E} at epoch {best_epoch}")
            print(f"Saved best model to {savepath}.best")

        if plotext and plotext_available:
            if validate:
                self.plot_loss(num_epochs, train_loss_hist, val_loss_hist)
            else:
                self.plot_loss(num_epochs, train_loss_hist)

    def _fit_lstsq(self, trainset, valset=None, savepath=None, verbose=True):
        """ Dummy method. To be implemented in subclasses. Fit method for basis
        functions that do not need training, i.e. EDMD type.
        """
        pass

    def plot_loss(self, num_epochs, train_loss_hist, val_loss_hist=None):

        pltx.theme("pro")

        steps = np.linspace(1, num_epochs,
                            len(train_loss_hist)).astype("int")
        pltx.scatter(
            steps[1::2], train_loss_hist[1::2], label="Train"
        )
        if val_loss_hist is not None:
            pltx.scatter(
                steps[::2], val_loss_hist[::2], label="Val"
            )
        pltx.yscale("log")
        pltx.plotsize(80, 40)
        pltx.show()
        print("")
