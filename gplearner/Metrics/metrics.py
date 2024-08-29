import warnings

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from scipy.stats import wasserstein_distance
from sklearn.metrics import homogeneity_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm


def calc_gp_stats(model, dm):
    """
    Calculate the number of genes per cell per GP
    (only using validation set to save time)

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
    """Initializes Maximum Mean Discrepancy(MMD)
    between source_features and target_features.
    - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.
    Parameters
    ----------
    source_features: torch.Tensor
         Tensor with shape [batch_size, z_dim]
    target_features: torch.Tensor
         Tensor with shape [batch_size, z_dim]
    Returns
    -------
    Returns the computed MMD between x and y.
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

    mmd_df = pd.DataFrame(mmd_list).set_index('condition')

    return mmd_df


def evaluate_emd(true_data, pred_data, condition_key=None, de_genes_dict=None):
    emd_list = []
    if condition_key:  # instead of condition have it per timepoint
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

        emd_df = pd.DataFrame(emd_list).set_index('condition')
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


#############################################
# Concept alignment score
# from https://github.com/mateoespinosa/cem
#############################################


def concept_alignment_score(
    c_vec,
    c_test,
    step,
    progress_bar=False,
):
    """
    Computes the concept alignment score between learnt concepts and labels.

    :param c_vec: predicted concept representations (can be concept embeddings)
    :param c_test: concept ground truth labels
    :param y_test: task ground truth labels
    :param step: number of integration steps
    :return: concept alignment AUC, task alignment AUC

    adapted from https://github.com/mateoespinosa/cem/blob/main/cem/metrics/cas.py
    accessed 27.04.2024

    EDIT : removed option to force alignment
    """

    warnings.simplefilter('ignore', UserWarning)

    # First lets compute an alignment between concept
    # scores and ground truth concepts
    # compute the maximum value for the AUC
    n_clusters = np.linspace(
        2,
        c_vec.shape[0],
        step,
    ).astype(int)

    max_auc = np.trapz(np.ones(len(n_clusters)))

    # for each concept:
    #   1. find clusters
    #   2. compare cluster assignments with ground truth concept/task labels
    concept_auc = []
    if progress_bar:
        bar = tqdm(range(c_test.shape[1]))
    else:
        bar = range(c_test.shape[1])
    for concept_id in bar:
        concept_homogeneity = []
        for nc in n_clusters:
            kmedoids = KMedoids(n_clusters=nc, random_state=0)
            if c_vec.shape[1] != c_test.shape[1]:
                c_cluster_labels = kmedoids.fit_predict(
                    np.hstack(
                        [
                            c_vec[:, concept_id][:, np.newaxis],
                            c_vec[:, c_test.shape[1] :],
                        ]
                    )
                )
            elif c_vec.shape[1] == c_test.shape[1] and len(c_vec.shape) == 2:
                c_cluster_labels = kmedoids.fit_predict(
                    c_vec[:, concept_id].reshape(-1, 1)
                )
            else:
                c_cluster_labels = kmedoids.fit_predict(c_vec[:, concept_id, :])

            # compute alignment with ground truth labels
            concept_homogeneity.append(
                homogeneity_score(c_test[:, concept_id], c_cluster_labels)
            )

            # EDIT ---- here we only have one set of labels
            # task_homogeneity.append(
            #     homogeneity_score(y_test, c_cluster_labels)
            # )

        # compute the area under the curve
        concept_auc.append(np.trapz(np.array(concept_homogeneity)) / max_auc)
        # task_auc.append(np.trapz(np.array(task_homogeneity)) / max_auc)

    # return the average alignment across all concepts
    concept_auc = np.mean(concept_auc)

    return concept_auc
