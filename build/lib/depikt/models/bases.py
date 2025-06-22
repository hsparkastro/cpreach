#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from itertools import combinations_with_replacement as combi


def mono_powers(nstate, degree):
    """ 
    Generate powers for monomials of degree 2 through `degree` 
    in `nstate` variables.
    """
    powers = list(combi(list(range(nstate+1)), degree))
    powers = powers[1:]  # selecting only orders higher than 0
    # removing zero'th powers, convert to zero-based indexing, and list
    powers = [list(x-1 for x in tup if x != 0) for tup in powers]

    return powers


class Monomials(nn.Module):

    def __init__(self, nstate, degree=None, powers=None,
                 nlift=None):

        super(Monomials, self).__init__()

        """ 
        Initialize the monomial basis. if degree is given, a 
        powers corresponding to a monomial basis of degree 'degree' in
        nstate variables is generated. If powers is given,
        the monomial basis is generated from the given powers. 
        
        Parameters
        ----------
        nstate: int
            Number of states (variables).
        degree: int or None, optional
            Degree of the monomial basis. If None, powers must be provided.
        powers: list of tuples or None, optional
            List of tuples representing the powers of the monomials.
            Each tuple corresponds to a monomial, where each element is the
            power of the corresponding variable. If None, powers are generated
            from the degree.

            Example:
            [(0, 0)] corresponds to a monomial x1**2.
            [(0,), (0, 0), (0, 2, 2)] corresponds to x1, x1**2, x1*x3**2
        nlift: int or None, optional
            Number of lifted states. If None, the number of lifted states is
            set to the number of powers. If nlift is provided, a trainable linear mapping is applied to the monomial basis to reduce the number of lifted states to nlift. Original states (monomials of degree 1) are always included in the mapping.
            
        """
        assert degree and not powers \
            or powers and not degree, \
            "Either provide degree or powers"

        if powers:
            self.powers = powers
        else:
            self.powers = mono_powers(nstate, degree)

        self.nstate = nstate

        if nlift is None:
            self.nlift = len(self.powers)
            self.reduce = False
        else:
            self.nlift = nlift
            assert nlift > 0, "nlift must be greater than 0"
            assert nlift <= len(self.powers), \
                "nlift must be less than or equal to the number of powers"
            self.reduce = nlift < len(self.powers)

        if self.reduce:
            self.M = nn.Parameter(torch.empty(
                (len(self.powers)-self.nstate, len(self.powers))-self.nstate))
            nn.init.xavier_uniform_(self.M)

    def mapping(self):
        return torch.linalg.qr(self.M)[0][:, :(self.nlift - self.nstate)]

    def forward(self, x):
        y = torch.stack([torch.prod(x[..., c], dim=-1)
                         for c in self.powers], dim=-1)
        if self.reduce:
            y = torch.cat((
                y[:, :self.nstate],
                y[:, self.nstate:] @ self.mapping()
            ), dim=-1)
        return y


class Identity(nn.Module):
    """
    Identity basis class, used to indicate that no lifting is done.
    """

    def __init__(self, nstate):
        super(Identity, self).__init__()
        self.nstate = nstate
        self.nlift = nstate

    def forward(self, x):
        return x
