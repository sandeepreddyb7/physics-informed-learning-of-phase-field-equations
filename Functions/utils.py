#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:42:40 2022

@author: bukka
"""


import torch

### the loss function

## loss function for Allen-Cahn equation
def loss_func_AC(u, u_pred, f_pred):

    loss_data = torch.mean((u - u_pred) ** 2)
    loss_eq = torch.mean(f_pred ** 2)
    loss = loss_data + loss_eq

    return loss, loss_data, loss_eq


## loss function for Cahn-Hilliard equation
def loss_func_CH(u, u_pred, f_pred, f_pred_mu):

    loss_data = torch.mean((u - u_pred) ** 2)
    loss_eq = torch.mean(f_pred ** 2)
    loss_mu = torch.mean(f_pred_mu ** 2)
    loss = loss_data + loss_eq + loss_mu

    return loss, loss_data, loss_eq, loss_mu


## loss function for Cahn-Hilliard equation with gradient descent for learning parameters
def loss_func_CH_GD(
    u, u_pred, f_pred, u_mu, mu_cal,
):

    loss_data = torch.mean((u - u_pred) ** 2)
    loss_eq = torch.mean(f_pred ** 2)
    loss_mu = torch.mean((u_mu - mu_cal) ** 2)
    loss = loss_data + loss_eq + loss_mu

    return loss, loss_data, loss_eq, loss_mu


### Least squares fit of the equation parameters


def leastsquares_fit(thetas, time_derivs):

    Q, R = torch.qr(thetas)  # solution of lst. sq. by QR decomp.
    coeff_vectors = torch.inverse(R) @ Q.T @ time_derivs

    return coeff_vectors


### calculation of the equation residual of AC equation


def equation_residual_AC(thetas, time_derivs, coeff_vectors):
    residual = time_derivs - thetas @ coeff_vectors

    return residual


### calculation of the mu equation residual
def equation_residual_mu_CH(thetas, mu_pred, coeff_vectors):
    residual = mu_pred - thetas @ coeff_vectors

    return residual


### calculation of equation residual for CH equation
def equation_residual_CH(features):
    residual = features[:, 0:1] + features[:, 2:3]

    return residual
