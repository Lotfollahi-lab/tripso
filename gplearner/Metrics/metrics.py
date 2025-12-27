import math
from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np
import ot
import pandas as pd
import torch
from ott.geometry import pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import tqdm


def calc_gp_stats(model, dm):
    """Calculate number of genes per cell for each gene program.

    Uses validation set only to save computation time.

    Parameters
    ----------
    model : nn.Module
        Model with gp_inputs attribute listing gene programs.
    dm : DataModule
        Data module with validation dataloader and metadata.

    Returns
    -------
    pd.DataFrame
        DataFrame with gene counts per cell for each GP and metadata.
    """
    dm.setup()

    def count_genes_per_cell(batch, gp_tokens_list):
        input_ids = batch['input_ids']

        # Get list of gp tokens
        gp_tokens = np.array(list(gp_tokens_list)).astype(np.int16)

        # Convert input IDs (list of lists) to array:
        holder = []

        # Function to pad a list with a specified value
        def pad_array(arr, desired_length=2048, padding_value=-100):
            current_length = len(arr)

            if current_length >= desired_length:
                return arr

            padding_size = desired_length - current_length
            padding = np.full(padding_size, padding_value)

            return np.concatenate([arr, padding])

        # Find max value for padding
        max_value = 2048

        for i in range(len(input_ids)):
            if len(input_ids[i]) == max_value:
                holder.append(input_ids[i].cpu().numpy())
            else:
                padded = pad_array(input_ids[i].cpu().numpy(), desired_length=max_value)
                holder.append(padded)

        # Build an array (n_cells, 2048) with token IDs at each position
        tokens_arr = np.array(holder)

        # binary mask (h, i, k)
        # in cell h, is the gene as position i in our GP of interest at position k?
        # print("Tokens:", np.array(gp_tokens))
        mask = (tokens_arr[:, :, np.newaxis] == gp_tokens[np.newaxis, :]).astype(int)

        # Count number of genes in each cell:
        # mask is (batch, 2048, n_gp_genes)
        # sum once to indicate whether a gene is in our gp
        # sum twice to count all of the GP genes in our cell
        return mask.sum(axis=-1).sum(axis=-1)

    count_dict = {}

    for gp in model.gp_inputs:
        count_dict[gp] = []

    for m in dm.metadata:
        count_dict[m] = []

    for batch in tqdm(dm.val_dataloader()):
        for i, gp in enumerate(model.gp_inputs):
            count_dict[gp] += count_genes_per_cell(
                batch,  # getattr(, f'gp{i}_tokens') #TO FIX
            ).tolist()

            if i == 0:
                for m in dm.metadata:
                    if isinstance(batch[m], torch.Tensor):
                        count_dict[m] += batch[m].cpu().tolist()
                    else:
                        count_dict[m] += batch[m]

    # Convert to dataframe
    df = pd.DataFrame(count_dict)

    return df


##################################################
# For evaluating distribution of generated counts
##################################################


def mmd_loss_calc(source_features, target_features, gamma):
    """Calculate Maximum Mean Discrepancy (MMD) loss.

    Computes MMD between source and target feature distributions using
    RBF kernel.

    Parameters
    ----------
    source_features : torch.Tensor
        Source distribution features, shape (batch_size, z_dim).
    target_features : torch.Tensor
        Target distribution features, shape (batch_size, z_dim).
    gamma : float
        RBF kernel scaling parameter.

    Returns
    -------
    torch.Tensor
        Computed MMD value between distributions.

    References
    ----------
    Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.
    """

    xx = rbf_kernel(source_features, source_features, gamma)
    xy = rbf_kernel(source_features, target_features, gamma)
    yy = rbf_kernel(target_features, target_features, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


# Metrics below were taken from:
# https://github.com/facebookresearch/CPA/blob/main/cpa/helper.py
# Date of access: 2024.01.08


def evaluate_mmd(adata, pred_adata, condition_key, de_genes_dict=None):
    mmd_list = []
    for cond in pred_adata.obs[condition_key].unique():
        adata_ = adata[adata.obs[condition_key] == cond].copy()
        pred_adata_ = pred_adata[pred_adata.obs[condition_key] == cond].copy()
        if issparse(adata_.X):
            adata_.X = adata_.X.A
        if issparse(pred_adata_.X):
            pred_adata_.X = pred_adata_.X.A

        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
        print('start mmd calculation')
        mmd = np.mean(
            list(map(lambda x: mmd_loss_calc(adata_.X, pred_adata_.X, x), gammas))
        )
        print('end mmd calculation')

        mmd_list.append({'condition': cond, 'mmd': mmd})

        if de_genes_dict:
            de_genes = de_genes_dict[cond]
            sub_adata_ = adata_[:, de_genes]
            sub_pred_adata_ = pred_adata_[:, de_genes]
            mmd_deg = mmd_loss_calc(
                torch.Tensor(sub_adata_.X), torch.Tensor(sub_pred_adata_.X)
            )
            mmd_list[-1]['mmd_deg'] = mmd_deg

    mmd_df = pd.DataFrame(mmd_list).set_index(condition_key)

    return mmd_df


def evaluate_emd(true_data, pred_data, condition_key=None, de_genes_dict=None):
    emd_list = []
    if condition_key:
        for cond in pred_data.obs[condition_key].unique():
            adata_ = true_data[true_data.obs[condition_key] == cond].copy()
            pred_adata_ = pred_data[pred_data.obs[condition_key] == cond].copy()
            if issparse(adata_.X):
                adata_.X = adata_.X.A
            if issparse(pred_adata_.X):
                pred_adata_.X = pred_adata_.X.A
            wd = []
            for i, _ in enumerate(adata_.var_names):
                wd.append(
                    wasserstein_distance(
                        torch.Tensor(adata_.X[:, i]), torch.Tensor(pred_adata_.X[:, i])
                    )
                )
            emd_list.append({'condition': cond, 'emd': np.mean(wd)})

            if de_genes_dict:
                de_genes = de_genes_dict[cond]
                sub_adata_ = adata_[:, de_genes]
                sub_pred_adata_ = pred_adata_[:, de_genes]
                wd_deg = []
                for i, _ in enumerate(sub_adata_.var_names):
                    wd_deg.append(
                        wasserstein_distance(
                            torch.Tensor(sub_adata_.X[:, i]),
                            torch.Tensor(sub_pred_adata_.X[:, i]),
                        )
                    )
                emd_list[-1]['emd_deg'] = np.mean(wd_deg)

        emd_df = pd.DataFrame(emd_list).set_index(condition_key)
    else:
        true_data_ = true_data.copy()
        pred_data_ = pred_data.copy()
        wd = []
        for i, _ in enumerate(true_data_.var_names):
            wd.append(
                wasserstein_distance(
                    torch.Tensor(true_data_.X[:, i]), torch.Tensor(pred_data_.X[:, i])
                )
            )
        emd_list.append({'emd': np.mean(wd)})
        emd_df = pd.DataFrame(emd_list).set_index(true_data_.var_names)
    return emd_df


def evaluate_emd_ref_vs_query(
    ref_data, query_data, ref_condition_key, query_condition_key, method=None
):
    emd_list = []

    for ref_cond in ref_data.obs[ref_condition_key].unique():
        for query_cond in query_data.obs[query_condition_key].unique():
            ref_adata_ = ref_data[ref_data.obs[ref_condition_key] == ref_cond].copy()

            query_adata_ = query_data[
                query_data.obs[query_condition_key] == query_cond
            ].copy()

            if issparse(ref_adata_.X):
                ref_adata_.X = ref_adata_.X.A
            if issparse(query_adata_.X):
                query_adata_.X = query_adata_.X.A

            ref = torch.tensor(ref_adata_.X)
            query = torch.tensor(query_adata_.X)

            metrics_dict = compute_distribution_distances(query, ref, method=method)

            out_dict = {'ref_condition': ref_cond, 'query_condition': query_cond}

            for k, v in metrics_dict.items():
                out_dict[k] = v

            emd_list.append(out_dict)

    emd_df = pd.DataFrame(emd_list)

    return emd_df


#############################################
# Distribution metrics
# from https://github.com/theislab/CFGen/blob/
# main/cfgen/eval/compute_evaluation_metrics.py#L71
#############################################


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """Compute Wasserstein distance between two distributions.

    Parameters
    ----------
    x0 : torch.Tensor
        The first distribution.
    x1 : torch.Tensor
        The second distribution.
    method : {'exact', 'sinkhorn'} or None, optional
        Method for computing Wasserstein distance. If None, uses 'exact'
        (default: None).
    reg : float, optional
        Regularization parameter for the Sinkhorn method (default: 0.05).
    power : {1, 2}, optional
        Power for the distance computation (default: 2).
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    float
        The computed Wasserstein distance.

    Raises
    ------
    ValueError
        If an unknown method is provided.

    References
    ----------
    From https://github.com/atong01/conditional-flow-matching/blob/
    v0/src/models/components/optimal_transport.py
    """
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == 'exact' or method is None:
        ot_fn = ot.emd2
    elif method == 'sinkhorn':
        ot_fn = partial(ot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f'Unknown method: {method}')

    a, b = ot.unif(x0.shape[0]), ot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2

    if method == 'sinkhorn':
        ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    else:
        ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret


# From https://github.com/atong01/conditional-flow-matching/
# blob/v0/src/models/components/mmd.py

min_var_est = 1e-8


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def compute_distribution_distances(
    pred: torch.Tensor, true: torch.Tensor, method='sinkhorn'
):
    """Compute distribution distances between predicted and true distributions.

    Calculates multiple distance metrics including Wasserstein distances and
    Maximum Mean Discrepancy (MMD) variants.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted tensor of shape (batch, times, dims).
    true : torch.Tensor or list
        True tensor of shape (batch, times, dims) or list of tensors of
        length times.
    method : str, optional
        Method for Wasserstein computation (default: 'sinkhorn').

    Returns
    -------
    dict
        Dictionary containing computed distribution distances with keys:
        '1-Wasserstein', '2-Wasserstein', 'Linear_MMD', 'Poly_MMD'.

    References
    ----------
    From https://github.com/theislab/CFGen/blob/main/cfgen/
    eval/distribution_distances.py#L16
    (accessed 24/09/24)
    """
    min_size = min(pred.shape[0], true.shape[0])

    names = ['1-Wasserstein', '2-Wasserstein', 'Linear_MMD', 'Poly_MMD']
    dists = []
    to_return = []
    w1 = wasserstein(pred, true, method=method, power=1)
    w2 = wasserstein(pred, true, method=method, power=2)
    pred_4_mmd = pred[:min_size]
    true_4_mmd = true[:min_size]
    mmd_linear = linear_mmd2(pred_4_mmd, true_4_mmd).item()
    mmd_poly = poly_mmd2(pred_4_mmd, true_4_mmd).item()
    dists.append((w1, w2, mmd_linear, mmd_poly))

    to_return.extend(np.array(dists).mean(axis=0))
    return dict(zip(names, to_return))


# =============================================================================


def euclidean_kernel_matrix(X, Y, gamma=1.0):
    """Compute Euclidean kernel matrix from pairwise distances.

    Parameters
    ----------
    X : ndarray
        Samples from distribution P, shape (m, d).
    Y : ndarray
        Samples from distribution Q, shape (n, d).
    gamma : float, optional
        Kernel scaling factor (default: 1.0).

    Returns
    -------
    K_xx : ndarray
        Kernel matrix for X with itself.
    K_yy : ndarray
        Kernel matrix for Y with itself.
    K_xy : ndarray
        Cross kernel matrix between X and Y.
    """
    return (
        np.exp(-gamma * cdist(X, X, 'euclidean')),
        np.exp(-gamma * cdist(Y, Y, 'euclidean')),
        np.exp(-gamma * cdist(X, Y, 'euclidean')),
    )


def compute_mmd(X, Y, gammas):
    """Compute Maximum Mean Discrepancy (MMD) between distributions.

    Parameters
    ----------
    X : ndarray
        Samples from distribution P, shape (m, d).
    Y : ndarray
        Samples from distribution Q, shape (n, d).
    gammas : list of float
        List of kernel scaling factors to test.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'gamma' and 'mmd' columns for each scaling factor.
    """

    mmd = []

    for g in gammas:
        m, n = len(X), len(Y)
        K_xx, K_yy, K_xy = euclidean_kernel_matrix(X, Y, g)

        # Compute MMD^2
        mmd_squared = (
            (K_xx.sum() / (m * m)) + (K_yy.sum() / (n * n)) - (2 * K_xy.sum() / (m * n))
        )

        mmd.append(np.sqrt(mmd_squared))

    return pd.DataFrame({'gamma': gammas, 'mmd': mmd})


def compute_sinkhorn(adata1, adata2, epsilons):
    """Compute Sinkhorn divergence between two datasets.

    Parameters
    ----------
    adata1 : AnnData
        First dataset.
    adata2 : AnnData
        Second dataset.
    epsilons : list of float
        Regularization parameters to test.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'epsilon' and 'sinkhorn_divergence' columns.
    """
    x, y = jnp.array(adata1.X), jnp.array(adata2.X)
    results = [
        sinkhorn_divergence(pointcloud.PointCloud, x=x, y=y, epsilon=eps)[0]
        for eps in epsilons
    ]
    return pd.DataFrame({'epsilon': epsilons, 'sinkhorn_divergence': results})
