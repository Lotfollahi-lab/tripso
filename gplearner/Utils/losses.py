# from Kevin

from typing import Dict

import torch
import torch.nn.functional as F

from .utils import one_hot_encoder


def mse(x_pred, x_true):
    """
    Description:
    ------------
    Compute mean squared error (MSE) loss.

    Parameters:
    -----------
    x_pred : torch.Tensor
        Predicted data.
    x_true : torch.Tensor
        True data.

    Returns:
    --------
    loss : torch.Tensor
        MSE loss.
    """
    loss = torch.nn.MSELoss()(x_pred, x_true)
    return loss


def mse_loss(x_pred, x_true):
    """
    Description:
    ------------
    Compute mean squared error (MSE) loss.

    Parameters:
    -----------
    x_pred : torch.Tensor
        Predicted data.
    x_true : torch.Tensor
        True data.

    Returns:
    --------
    loss : torch.Tensor
        MSE loss.
    """
    loss = torch.nn.functional.mse_loss(x_pred, x_true, reduction='none')
    return loss


def nb(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps=1e-8):
    """
    This negative binomial function was taken from:
    Title: scvi-tools
    Authors: Romain Lopez <romain_lopez@gmail.com>,
             Adam Gayoso <adamgayoso@berkeley.edu>,
             Galen Xing <gx2113@columbia.edu>
    Date: 16th November 2020
    Code version: 0.8.1
    Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py # noqa

    Computes negative binomial loss.
    Parameters
    ----------
    x: torch.Tensor
         Torch Tensor of ground truth data.
    mu: torch.Tensor
         Torch Tensor of means of the negative binomial (has to be positive support).
    theta: torch.Tensor
         Torch Tensor of inverse dispersion parameter (has to be positive support).
    eps: Float
         numerical stability constant.

    Returns
    -------
    If 'mean' is 'True' NB loss value gets returned,
    otherwise Torch tensor of losses gets returned.
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    return res


def zinb(
    x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor, eps=1e-8
):
    """
    This zero-inflated negative binomial function was taken from:
    Title: scvi-tools
    Authors: Romain Lopez <romain_lopez@gmail.com>,
             Adam Gayoso <adamgayoso@berkeley.edu>,
             Galen Xing <gx2113@columbia.edu>
    Date: 16th November 2020
    Code version: 0.8.1
    Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py # noqa

    Computes zero inflated negative binomial loss.
    Parameters
    ----------
    x: torch.Tensor
         Torch Tensor of ground truth data.
    mu: torch.Tensor
         Torch Tensor of means of the negative binomial (has to be positive support).
    theta: torch.Tensor
         Torch Tensor of inverses dispersion parameter (has to be positive support).
    pi: torch.Tensor
         Torch Tensor of logits of the dropout parameter (real support)
    eps: Float
         numerical stability constant.

    Returns
    -------
    If 'mean' is 'True' ZINB loss value gets returned,
    otherwise Torch tensor of losses gets returned.
    """
    # theta is the dispersion rate. If .ndimension() == 1,
    # it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)  # uses log(sigmoid(x)) = -softplus(-x)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    return res


def compute_count_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    reconstruction_loss: str,
    theta: torch.Tensor,
    n_condition_combined: int,
):
    true_counts = batch['counts']
    batch_size_factor = torch.tensor(batch['size_factor']).to(true_counts.device)

    if reconstruction_loss == 'mse':
        loss = (
            mse_loss(outputs['count_output']['count_lognorm'], true_counts)
            .sum(dim=-1)
            .mean()
            .float()
        )
        return loss

    elif reconstruction_loss == 'zinb':
        dec_mean_gamma, dec_dropout = (
            outputs['count_output']['count_mean'],
            outputs['count_output']['count_dropout'],
        )
        size_factor_view = batch_size_factor.unsqueeze(1).expand(
            dec_mean_gamma.size(0), dec_mean_gamma.size(1)
        )
        dec_mean = dec_mean_gamma * size_factor_view

        dispersion = F.linear(
            one_hot_encoder(batch['batch_key_id'], n_condition_combined),
            theta,
        )

        dispersion = torch.exp(dispersion)
        loss = (
            -zinb(x=true_counts, mu=dec_mean, theta=dispersion, pi=dec_dropout)
            .sum(dim=-1)
            .mean()
        )
        return loss

    elif reconstruction_loss == 'nb':
        dec_mean_gamma = outputs['count_output']['count_mean']
        size_factor_view = batch_size_factor.unsqueeze(1).expand(
            dec_mean_gamma.size(0), dec_mean_gamma.size(1)
        )
        dec_mean = dec_mean_gamma * size_factor_view
        dispersion = F.linear(
            one_hot_encoder(batch['batch_key_id'], n_condition_combined),
            theta,
        )
        dispersion = torch.exp(dispersion)
        loss = -nb(x=true_counts, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()
        return loss

    else:
        raise ValueError(
            'Reconstruction loss not supported' 'Please choose from mse, nb or zinb'
        )
