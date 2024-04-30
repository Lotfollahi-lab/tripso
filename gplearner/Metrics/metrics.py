import random
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse
from scipy.stats import wasserstein_distance
from sklearn.metrics import homogeneity_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

from ..Utils.utils import do_logistic_regression


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


def evaluate_by_gene_singleGP(
    model,
    data_module,
    output_directory,
):
    data_module.setup()

    # First get list of genes that are present in exactly one gene program
    # Function to find genes that are present in exactly one column
    def find_genes_in_single_gp(df):
        all_genes = set()
        genes_in_single_column = set()

        for col in df.columns:
            col_genes = set(df[col])
            genes_in_single_column.update(col_genes - all_genes)
            all_genes.update(col_genes)

        return list(genes_in_single_column)

    genes_in_single_gp = find_genes_in_single_gp(model.gpdb)

    if np.nan in genes_in_single_gp:
        genes_in_single_gp.remove(np.nan)

    if model.model.do_ensembl_conversion:
        genes_in_single_gp = [
            model.gene_name_dict[g]
            for g in genes_in_single_gp
            if g in model.gene_name_dict.keys()
        ]

    tokens_to_keep = [
        # to fix --> get this from geneformer wrapper
        model.token_dict[g]
        for g in genes_in_single_gp
        if g in model.token_dict.keys()
    ]

    print(len(genes_in_single_gp), 'genes present in exactly one GP')

    # Get embeddings for these genes

    loader = data_module.dataloader_for_token_extraction()

    # Create empty lists to store embeddings and tokens
    x_gf = []
    tokens_gf = []
    x_scgpl = []
    tokens_scgpl = []
    gp_labels = []

    # to(available_device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting gene embeddings', leave=False):
            for i in batch:
                # move tensors to gpu
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].to(
                        model.available_device
                    )  # does this slow things down?

            emb_gf = model.gfWrapper(model=model.gf, input_data=batch)

            # Filter geneformer embeddings to only keep genes in single GP
            # Get list of tokens for genes in this batch
            tokens = batch['input_ids']

            # Only select genes which are in single GP
            emb_gf, tokens, _ = model.build_input_matrix(
                emb_gf[0], tokens, tokens_to_keep, mode='extract_genes'
            )

            # Reshape so each row is a gene and remove rows which are all 0
            emb_gf = emb_gf.reshape(emb_gf.shape[0] * emb_gf.shape[1], -1)
            non_missing = (emb_gf != 0).all(dim=-1)
            emb_gf = emb_gf[non_missing]

            tokens = tokens.reshape(tokens.shape[0] * tokens.shape[1])
            tokens = tokens[tokens != -100]

            # Add to list
            x_gf.append(emb_gf)
            tokens_gf.append(tokens)

            # Now run through scGPL
            gene_emb_list, tokens_list, gp_out_list = model(
                batch, return_gene_embeddings=True
            )

            # Filter to only keep genes in multiple GP
            # loop through emb list = embeddings are grouped by GP
            for i in range(len(gene_emb_list)):
                x_out, tokens, _ = model.model.build_input_matrix(
                    gene_emb_list[i],
                    tokens_list[i],
                    tokens_to_keep,
                    mode='extract_genes',
                )
                gp_label = gp_out_list[i]

                # remove missing values
                x_out = x_out.reshape(x_out.shape[0] * x_out.shape[1], -1)
                non_missing = (x_out != 0).all(dim=1)
                x_out = x_out[non_missing]

                tokens = tokens.reshape(tokens.shape[0] * tokens.shape[1])
                tokens = tokens[tokens != -100]

                gp_label = [gp_label[0] for _ in range(tokens.shape[0])]

                # Add to list
                x_scgpl.append(x_out)
                tokens_scgpl.append(tokens)
                gp_labels += gp_label

    # Concatenate tensors
    x_gf = torch.cat(x_gf, dim=0).cpu().numpy()
    x_scgpl = torch.cat(x_scgpl, dim=0).cpu().numpy()
    tokens_gf = torch.cat(tokens_gf, dim=0).cpu().numpy()
    tokens_scgpl = torch.cat(tokens_scgpl, dim=0).cpu().numpy()

    # For our labels, we actually want to use the gene program names not gene tokens
    # Create a mapping of integers to column names
    gene_to_GP = {
        value: column
        for column in model.gpdb.columns
        for value in model.gpdb[column].dropna()
    }

    token_to_gene = {v: k for k, v in model.token_dict.items()}

    if model.do_ensembl_conversion:
        ensembl_to_name = {v: k for k, v in model.gene_name_dict.items()}

    # Create anndata object for clustering and visualisation
    adata_gf = sc.AnnData(X=x_gf)
    adata_gf.obs['token'] = list(tokens_gf)
    adata_gf.obs['ensembl'] = adata_gf.obs['token'].map(token_to_gene)

    if model.do_ensembl_conversion:
        adata_gf.obs['gene'] = adata_gf.obs['ensembl'].map(ensembl_to_name)
        adata_gf.obs['GP'] = adata_gf.obs['gene'].map(gene_to_GP)
    else:
        adata_gf.obs['GP'] = adata_gf.obs['ensembl'].map(gene_to_GP)

    # Now for scgpl
    adata_scgpl = sc.AnnData(X=x_scgpl)
    adata_scgpl.obs['token'] = list(tokens_scgpl)
    adata_scgpl.obs['ensembl'] = adata_scgpl.obs['token'].map(token_to_gene)

    if model.do_ensembl_conversion:
        adata_scgpl.obs['gene'] = adata_scgpl.obs['ensembl'].map(ensembl_to_name)
        adata_scgpl.obs['GP'] = adata_scgpl.obs['gene'].map(gene_to_GP)
    else:
        adata_scgpl.obs['GP'] = adata_scgpl.obs['ensembl'].map(gene_to_GP)

    # Run logistic regression models
    do_logistic_regression(
        adata_gf,
        'GP',
        output_directory,
        filename='gp_prediction_from_geneformer',
        variable_to_track={'embedding_type': 'geneformer'},
    )

    do_logistic_regression(
        adata_scgpl,
        'GP',
        output_directory,
        filename='gp_prediction_from_scgpl',
        variable_to_track={'embedding_type': 'scGPL'},
    )

    return adata_gf, adata_scgpl


def evaluate_by_gene_multiGP(
    model, data_module, output_directory, min_cells=500, downsample_to_n_genes=False
):
    """
    GP prediction task when a gene is present in multiple GPs
    """
    # First find genes present in multiple GPs #####

    data_module.setup()

    # Function to find the intersection of genes for a specific combination of columns
    def find_gene_intersection(df, column_combination):
        genes = set(df[column_combination[0]])
        for col in column_combination[1:]:
            genes = genes.intersection(df[col])
        return genes

    # Create a set to store genes present in more than one column
    common_genes_set = set()

    # Loop through different pairs of columns (2 to 5)
    for num_columns in range(2, len(model.gp_inputs)):
        column_combinations = combinations(model.gpdb.columns, num_columns)
        for combination in column_combinations:
            common_genes = find_gene_intersection(
                model.gpdb, combination
            )  # CHANGE HERE

            common_genes_set.update(common_genes)

    if np.nan in common_genes_set:
        common_genes_set.remove(np.nan)

    print(f'Union of genes present in more than one GP: {len(common_genes_set)}')

    # Of all these genes, how many are present in at least min_cells cells?
    # Extract the 'input_ids' column as a list of lists
    input_ids_lists = data_module.dataset['input_ids']

    # Flatten the list of lists into a single list
    flat_input_ids = [item for sublist in input_ids_lists for item in sublist]
    # Count the occurrences of each unique value
    value_counts = Counter(flat_input_ids)

    # Create a DataFrame from the counts
    token_df = pd.DataFrame(
        {'token': list(value_counts.keys()), 'counts': list(value_counts.values())}
    )

    # map tokens back to ENSEMBL IDs and gene names
    token_to_gene = {v: k for k, v in model.token_dict.items()}
    ensembl_to_name = {v: k for k, v in model.gene_name_dict.items()}

    token_df['ensembl'] = token_df['token'].map(token_to_gene)
    token_df['gene'] = token_df['ensembl'].map(ensembl_to_name)

    token_df['total'] = len(data_module.dataset)
    token_df['prop'] = token_df['counts'] / token_df['total']

    token_df = token_df[['gene', 'ensembl', 'token', 'counts', 'prop', 'total']]

    if model.do_ensembl_conversion:
        # then common_genes_set is storing gene names
        token_multi = token_df[token_df['gene'].isin(common_genes_set)]
    else:
        # then common_genes_set is storing ensembl IDs
        token_multi = token_df[token_df['ensembl'].isin(common_genes_set)]

    print(
        f'Range of counts: {token_multi.counts.min()}' f'- {token_multi.counts.max()}'
    )

    token_multi = token_multi[token_multi['counts'] > min_cells]
    tokens_to_keep = token_multi['token'].tolist()

    print(
        f'Number of genes present in more than one GP'
        f'and at least {min_cells} cells: {len(token_multi)}'
    )

    if len(token_multi) == 0:
        raise ValueError(
            f'No genes present in more than one GP are present'
            f'in at least {min_cells} cells. Please relax the threshold.'
        )

    if downsample_to_n_genes:
        print(f'Downsampling to {downsample_to_n_genes} genes')
        print('')
        tokens_to_keep = random.sample(tokens_to_keep, downsample_to_n_genes)

    # Get embeddings for these genes #####
    loader = data_module.dataloader_for_token_extraction()

    # Create empty lists to store embeddings and tokens
    x_scgpl = []
    tokens_scgpl = []
    gp_labels = []

    # model.to(available_device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting gene embeddings', leave=False):
            for i in batch:
                # move tensors to gpu
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i]  # .to(
                    # TO FIX -> HANDLING GPU
            gene_emb_list, tokens_list, gp_out_list = model(
                batch, return_gene_embeddings=True
            )
            # z, logits_lm_list, gene_labels_list =
            # model(batch, return_gene_embeddings = False)

            # Filter to only keep genes in multiple GP
            # loop through emb list = embeddings are grouped by GP
            for i in range(len(gene_emb_list)):
                x_out, tokens, _ = model.model.build_input_matrix(
                    gene_emb_list[i],
                    tokens_list[i],
                    tokens_to_keep,
                    mode='extract_genes',
                )
                gp_label = gp_out_list[i]

                # remove missing values
                x_out = x_out.reshape(x_out.shape[0] * x_out.shape[1], -1)
                non_missing = (x_out != 0).all(dim=1)
                x_out = x_out[non_missing]

                tokens = tokens.reshape(tokens.shape[0] * tokens.shape[1])
                tokens = tokens[tokens != -100]

                gp_label = [gp_label[0] for _ in range(tokens.shape[0])]

                # Add to list
                x_scgpl.append(x_out)
                tokens_scgpl.append(tokens)
                gp_labels += gp_label

    # Concatenate tensors
    x_scgpl = torch.cat(x_scgpl, dim=0).cpu().numpy()
    tokens_scgpl = torch.cat(tokens_scgpl, dim=0).cpu().numpy()
    # gp_labels = torch.cat(gp_labels, dim = 0).cpu().numpy()

    # Create anndata object for clustering and visualisation
    adata = sc.AnnData(X=x_scgpl)
    adata.obs['token'] = list(tokens_scgpl)
    adata.obs['GP'] = list(gp_labels)

    # Map gene names for interpretability
    adata.obs['ensembl'] = adata.obs['token'].map(token_to_gene)
    adata.obs['gene'] = adata.obs['ensembl'].map(ensembl_to_name)

    # Run logistic regression models - PER GENE
    print('Run logistic regression models - PER GENE')
    for g in adata.obs['ensembl'].unique():
        print(g)
        tdata = adata[adata.obs['ensembl'] == g]
        do_logistic_regression(
            tdata,
            'GP',
            output_directory,
            filename=f'gp_prediction_from_scgpl_{g}',
            variable_to_track={'ensembl': g, 'gene': tdata.obs['gene'].iloc[0]},
        )

    print('')
    return adata


def evaluate_by_cell(model, adata, output_directory):
    """
    Use z_gp to predict cell type

    """

    for gp in model.gp_inputs:
        print('Running cell evaluation metrics for', gp)

        do_logistic_regression(
            adata[:, adata.var['gp_idx'].str.startswith(gp)],
            'cell_type',
            output_directory,
            filename=f"cell_type_prediction_{gp.replace('/', '_')}",
            variable_to_track={'GP': gp},
        )

        do_logistic_regression(
            adata[:, adata.var['gp_idx'].str.startswith(gp)],
            'condition',
            output_directory,
            filename=f"condition_prediction_{gp.replace('/', '_')}",
            variable_to_track={'GP': gp},
        )


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
