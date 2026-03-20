import argparse
import glob
import logging
import math
import os
import pickle
import random
import re
import sys
import tarfile
import warnings
from collections import Counter
from multiprocessing import Pool
from typing import (
    List,
    Optional,
    Union,
)

import anndata as ad
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import requests  # type: ignore
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import torch
import torch.nn as nn
import triton
import triton.language as tl
from datasets import (
    DatasetDict,
    concatenate_datasets,
    load_from_disk,
)
from scipy.stats import t as student_t
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    davies_bouldin_score,
    mean_squared_error,
    normalized_mutual_info_score,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import PearsonCorrCoef
from tqdm import tqdm

from ..Metrics.metrics import evaluate_emd, evaluate_mmd

logger = logging.getLogger(__name__)

random.seed(0)

# for exporting pdfs
matplotlib.rcParams['pdf.fonttype'] = 42  # to export text as editable

###################################
# Generic
###################################


def one_hot_encoder(idx, n_cls):
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot


def find_latest_file(output_dir, tissue, supervised_tag):
    # Define the pattern to match the desired file format
    pattern = f'*_gp_transformer_{tissue}_{supervised_tag}*.ckpt'

    # Search for files in the directory matching the pattern
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    matching_files = glob.glob(os.path.join(checkpoint_dir, pattern))

    # Filter for only .ckpt files and sort by modification time
    latest_file = max(matching_files, key=os.path.getmtime) if matching_files else None

    if latest_file is None:
        raise FileNotFoundError(
            f'No .ckpt files matching {tissue} with model type'
            f' {supervised_tag} found in {checkpoint_dir}. '
            'Did you train the model?'
        )

    print('Loading model from', latest_file)

    return latest_file


def average_nz(x):
    # Replace zero values with NaN to facilitate ignoring them during averaging
    x[x == 0] = float('nan')

    # Calculate the mean along the last dimension (embedding_dim)
    # Specify 'nanmean' to ignore NaN values during the mean calculation
    x = torch.nanmean(x, dim=1)

    return x


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {'off', 'false', '0'}
    TRUTHY_STRINGS = {'on', 'true', '1'}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError('invalid value for a boolean flag')


def load_gmt(path, rm_col_1=True):
    """
    Load a GMT file into a pandas dataframe.
    """
    # Load GOBP for gene sets
    df = pd.read_fwf(path, sep='\t', header=None)
    gobp = df[0].str.split('\t', expand=True)

    # drop column with GP URL
    if rm_col_1:
        gobp = gobp.drop(gobp.columns[1], axis=1)

    # wrangle so column names are gene program names
    gobp = gobp.set_index(0)
    gobp = gobp.T

    return gobp


def remove_leading_numbers_and_underscore(input_string):
    return re.sub(r'^[\d_]+', '', input_string)


class MidpointNormalize(mcolors.Normalize):
    '''
    Palette normalization with centering and adapted dynamic range to correspond to
    the distance of vmin and vmax from the cenetr
    Adapted from https://stackoverflow.com/a/50003503

    taken directly from
    https://scanpy-tutorials.readthedocs.io/en/latest/plotting/advanced.html#colors
    '''

    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        value = np.array(value).astype(float)
        normalized_min = max(
            0.0,
            0.5
            * (1.0 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))),
        )
        normalized_max = min(
            1.0,
            0.5
            * (1.0 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))),
        )
        normalized_mid = 0.5
        x, y = (
            [self.vmin, self.midpoint, self.vmax],
            [normalized_min, normalized_mid, normalized_max],
        )
        return np.ma.masked_array(np.interp(value, x, y))


def pad_tensor(tensor, pad_token_id, max_len):
    tensor = torch.nn.functional.pad(
        tensor, pad=(0, max_len - tensor.numel()), mode='constant', value=pad_token_id
    )

    return tensor


def pad_3d_tensor(tensor, pad_token_id, max_len, dim):
    if dim == 0:
        raise Exception('dim 0 usually does not need to be padded.')
    if dim == 1:
        pad = (0, 0, 0, max_len - tensor.size()[dim])
    elif dim == 2:
        pad = (0, max_len - tensor.size()[dim], 0, 0)
    tensor = torch.nn.functional.pad(
        tensor, pad=pad, mode='constant', value=pad_token_id
    )
    return tensor


# pad list of tensors and convert to tensor
def pad_tensor_list(
    tensor_list,
    dynamic_or_constant,
    pad_token_id,
    model_input_size,
    dim=None,
    padding_func=None,
):
    '''
    From Geneformer
    geneformer.perturber_utils accessed June 2025
    '''
    # determine maximum tensor length
    if dynamic_or_constant == 'dynamic':
        max_len = max([tensor.squeeze().numel() for tensor in tensor_list])
    elif isinstance(dynamic_or_constant, int):
        max_len = dynamic_or_constant
    else:
        max_len = model_input_size
        logger.warning(
            'If padding style is constant, must provide integer value. '
            f'Setting padding to max input size {model_input_size}.'
        )

    # pad all tensors to maximum length
    if dim is None:
        tensor_list = [
            pad_tensor(tensor, pad_token_id, max_len) for tensor in tensor_list
        ]
    else:
        tensor_list = [
            padding_func(tensor, pad_token_id, max_len, dim) for tensor in tensor_list
        ]
    # return stacked tensors
    if padding_func != pad_3d_tensor:
        return torch.stack(tensor_list)
    else:
        return torch.cat(tensor_list, 0)


###################################
# Wrangling hugging face dataset
###################################


def pivot_single_column(x, col, values_to, cols_to_keep, pivot_cols_suffix, names_to):
    z = x.rename_column(col, values_to)
    z = z.select_columns(values_to)

    # add desired metadata
    for meta in cols_to_keep:
        z = z.add_column(meta, x[meta])

    # add gene column
    clean_name = col
    for suffix in pivot_cols_suffix:
        if suffix != '':
            clean_name = col.replace(suffix, '')

    z = z.add_column(names_to, [clean_name] * len(z))

    return z


def dataset_pivot_longer(
    in_dir,
    out_dir,
    filename,
    pivot_cols_start_with,
    pivot_cols_suffix,
    values_to,
    names_to,
    cols_to_keep,
):
    '''
    Pivot longer for huggingface dataset
    '''

    # Load the dataset
    x = load_from_disk(os.path.join(in_dir, filename))

    # Extract the data
    if isinstance(pivot_cols_suffix, str):
        pivot_cols_suffix = [pivot_cols_suffix]

    col_groups = []

    for suffix in pivot_cols_suffix:
        if suffix == '':
            cols = [
                col
                for prefix in pivot_cols_start_with
                for col in x.column_names
                if col == prefix
            ]
        else:
            cols = [
                col
                for prefix in pivot_cols_start_with
                for col in x.column_names
                if col.startswith(prefix) and col.endswith(suffix)
            ]

        col_groups.append(cols)

    long_dataset = None

    for i, c in enumerate(col_groups):
        if i == 0:
            for j, col in enumerate(c):
                if j == 0:
                    long_dataset = pivot_single_column(
                        x, col, values_to[i], cols_to_keep, pivot_cols_suffix, names_to
                    )

                else:
                    z = pivot_single_column(
                        x,
                        col,
                        values_to[i],
                        cols_to_keep,
                        pivot_cols_suffix,
                        names_to,
                    )
                    long_dataset = concatenate_datasets([long_dataset, z])
        else:
            for j, col in enumerate(c):
                z = pivot_single_column(
                    x, col, values_to[i], cols_to_keep, pivot_cols_suffix, names_to
                )
                long_dataset = concatenate_datasets([long_dataset, z])

    # Save the dataset
    long_dataset.save_to_disk(os.path.join(out_dir, filename))

    return None


def encode_labels(input_data, input_col, new_col):
    """
    Encode labels as integers
    works on Huggingface dataset class
    """
    label_values = input_data.unique(input_col)
    label_dict = {l: i for i, l in enumerate(label_values)}

    def classes_to_ids(example):
        example[new_col] = label_dict[example[input_col]]
        return example

    labeled_dataset = input_data.map(classes_to_ids, num_proc=4)

    return labeled_dataset


def encode_labels_h5ad(input_adata, input_col, new_col):
    """
    Encode labels as integers
    works on AnnData
    """
    label_values = list(input_adata.obs[input_col].unique())
    label_dict = {l: i for i, l in enumerate(label_values)}

    input_adata.obs[new_col] = input_adata.obs[input_col].map(label_dict)

    return input_adata


def do_balanced_downsampling(class_values, input_data, n_cells_per_class=None):
    """
    Perform balanced subsampling of input data
    for Huggingface dataset class

    """
    # Calculate class frequencies
    class_counts = Counter(class_values)

    if n_cells_per_class is None:
        n_cells_per_class = np.array(list(class_counts.values())).min()

    # Perform balanced subsampling
    balanced_samples = []
    for label, count in class_counts.items():
        subsample_count = min(count, n_cells_per_class)
        class_indices = [i for i, l in enumerate(class_values) if l == label]
        subsample_indices = random.sample(class_indices, subsample_count)
        balanced_samples.extend(subsample_indices)

    input_data = input_data.select(balanced_samples)

    return input_data


def do_balanced_downsampling_anndata(adata, subsample_by, n_cells_per_class=None):
    """
    Perform balanced subsampling of input data

    """
    # Calculate class frequencies
    class_counts = adata.obs[subsample_by].value_counts()

    if n_cells_per_class is None:
        n_cells_per_class = class_counts.min()

    # Perform balanced subsampling
    balanced_samples = []

    for label, count in class_counts.items():
        subsample_count = min(count, n_cells_per_class)
        class_indices = adata.obs.index[adata.obs[subsample_by] == label]
        subsample_indices = np.random.choice(
            class_indices, subsample_count, replace=False
        )
        balanced_samples.extend(subsample_indices)

    input_data = adata[balanced_samples, :]

    return input_data


def label_encoder(adata, encoder, condition_key=None):
    """
    Description:
    ------------
    Encode labels of Annotated `adata` matrix.

    Parameters:
    ----------
    adata: : `~anndata.AnnData`
         Annotated data matrix.
    encoder: Dict
         dictionary of encoded labels.
    condition_key: String
         column name of conditions in `adata.obs` data frame.

    Returns:
    -------
    labels: `~numpy.ndarray`
         Array of encoded labels
    label_encoder: Dict
         dictionary with labels and encoded labels as key, value pairs.
    """
    unique_conditions = list(np.unique(adata.obs[condition_key]))
    labels = np.zeros(adata.shape[0])

    if not set(unique_conditions).issubset(set(encoder.keys())):
        missing_labels = set(unique_conditions).difference(set(encoder.keys()))
        print(
            f'Warning: Labels in adata.obs[{condition_key}]'
            'is not a subset of label-encoder!'
        )
        print(f'The missing labels are: {missing_labels}')
        print('Therefore integer value of those labels is set to -1')
        for data_cond in unique_conditions:
            if data_cond not in encoder.keys():
                labels[adata.obs[condition_key] == data_cond] = -1

    for condition, label in encoder.items():
        labels[adata.obs[condition_key] == condition] = label
    labels = [int(x) for x in labels]
    return labels


def sample_cells(adata, column, n_cells):
    # numpy set seed
    np.random.seed(0)

    sampled_indices = []

    # Group by the column in obs and sample n_cells per group
    for value in adata.obs[column].unique():
        group_indices = adata.obs[adata.obs[column] == value].index
        n_cells_i = min(n_cells, len(group_indices))
        sampled_indices.extend(
            np.random.choice(group_indices, n_cells_i, replace=False)
        )

    # Return the sampled AnnData object
    return adata[sampled_indices, :]


def align_indices(idx_cell, idx_genes, gp_arr, gene_arrays):
    '''
    Align indices of cells and genes

    Parameters
    ----------
    idx_cell : list
        List of cell indices.
    idx_genes : list
        List of cell indices for the gene data
    gp_arr : np.ndarray
        GP cls (cell-level)
    gene_arrays : dict
        Dictionary where keys are gene names and values are gene embedings.
    '''
    # Convert indices to Pandas Index for fast alignment
    idx_cell = pd.Index(idx_cell)
    idx_genes = pd.Index(idx_genes)

    # Identify shared indices using Pandas intersection (faster than sets)
    shared_indices = idx_cell.intersection(idx_genes)
    print('Cells not in genes:', len(idx_cell.difference(idx_genes)))
    print('Genes not in cells:', len(idx_genes.difference(idx_cell)))
    print('Both:', len(shared_indices))

    # Get the position of the shared indices in the original arrays
    cell_indexer = idx_cell.get_indexer(shared_indices)
    gene_indexer = idx_genes.get_indexer(shared_indices)

    # Use NumPy advanced indexing to reorder gp_arr and gene arrays
    gp_arr = gp_arr[cell_indexer]
    gene_arrays = {
        gene: gene_arr[gene_indexer] for gene, gene_arr in gene_arrays.items()
    }

    return shared_indices, gp_arr, gene_arrays


###################################
# Gene expression transformation
###################################


def _digitize(x: np.ndarray, bins: np.ndarray, side='both') -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.


    from https://github.com/bowang-lab/scGPT/blob/main/scgpt/preprocess.py#L13

    accessed 03.04.2024
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == 'one':
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def bin_gene_expression(x, n_bins=10, norm=False, log1p=False):
    '''
    Based on scGPT preprocessor
    https://github.com/bowang-lab/scGPT/blob/main/scgpt/preprocess.py#L13
    Accessed 03.04.2024
    '''
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    adata = sc.AnnData(X=x)

    if norm:
        sc.pp.normalize_total(adata, target_sum=1e4)
    if log1p:
        sc.pp.log1p(adata)

    binned_rows = []
    bin_edges = []

    if x.min() < 0:
        raise ValueError(f'Assuming non-negative data, but got min value {x.min()}.')
    for row in x:
        if row.max() == 0:
            binned_rows.append(np.zeros_like(row, dtype=np.int64))
            bin_edges.append(np.array([0] * n_bins))
            continue
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        # bins = np.sort(np.unique(bins))
        # NOTE: comment this line for now, since this will make the each category
        # has different relative meaning across datasets
        non_zero_digits = _digitize(non_zero_row, bins)
        assert non_zero_digits.min() >= 1
        assert non_zero_digits.max() <= n_bins - 1
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
        binned_rows.append(binned_row)
        bin_edges.append(np.concatenate([[0], bins]))

    return np.stack(binned_rows)


###################################
# Padding
###################################


def pad_array(arr, desired_length=2048, padding_value=-100):
    current_length = len(arr)

    if current_length >= desired_length:
        return arr

    padding_size = desired_length - current_length
    padding = np.full(padding_size, padding_value)

    return np.concatenate([arr, padding])


###################################
# GP wrangling
###################################


def build_token_to_gene_name_dict(
    name_dictionary_path, token_dictionary_path, genes_to_keep, do_ensembl_conversion
):
    # converting between different gene labels
    with open(name_dictionary_path, 'rb') as f:
        name_dictionary = pickle.load(f)
    with open(token_dictionary_path, 'rb') as f:
        token_dictionary = pickle.load(f)

    # Convert the dictionaries into DataFrames for easy merging
    name_df = pd.DataFrame(
        list(name_dictionary.items()), columns=['gene_name', 'ensembl_id']
    )
    token_df = pd.DataFrame(
        list(token_dictionary.items()), columns=['ensembl_id', 'token']
    )

    # Only keep genes of interest
    if do_ensembl_conversion:
        # Merge on ensembl_id
        mapping_df = name_df.join(
            token_df.set_index('ensembl_id'), on='ensembl_id', how='inner'
        )

        genes_to_keep_df = mapping_df[mapping_df['gene_name'].isin(genes_to_keep)]

    else:
        # debugging
        genes_to_keep_df = token_df[token_df['ensembl_id'].isin(genes_to_keep)]

    # Merge ensembl_ids with the token DataFrame to get tokens
    tokens_to_keep = genes_to_keep_df['token'].tolist()

    # Display the number of genes to keep
    print(f'Number of genes to keep: {len(tokens_to_keep)}')

    # Create dictionary for conversion
    if do_ensembl_conversion:
        token_to_gene_to_keep_dict = dict(
            zip(genes_to_keep_df['token'], genes_to_keep_df['gene_name'])
        )
    else:
        token_to_gene_to_keep_dict = dict(
            zip(genes_to_keep_df['token'], genes_to_keep_df['ensembl_id'])
        )

    return tokens_to_keep, token_to_gene_to_keep_dict


def convert_gene_names_to_tokens(
    genes,
    name_dictionary,
    token_dictionary,
    do_ensembl_conversion=True,
    gp_name=None,
):
    # Convert gene names to Ensembl IDs
    if do_ensembl_conversion:
        ensembl_ids = [name_dictionary.get(gene_name, 'Unknown') for gene_name in genes]
    else:
        ensembl_ids = genes

    # Convert ensembl IDs to tokens:
    gp_tokens = [
        token_dictionary.get(gene_name, 'Unknown') for gene_name in ensembl_ids
    ]

    # Unknown values later cause issues for indexing -> remove
    if 'Unknown' in gp_tokens:
        print(f"In {gp_name}, dropped {gp_tokens.count('Unknown')} unknown genes")
        while 'Unknown' in gp_tokens:
            gp_tokens.remove('Unknown')

    return gp_tokens


def get_gp_tokens(
    gp_genes: Union[List[str], pd.Series],
    do_ensembl_conversion: bool,
    gp_name: Optional[str],
    gene_token_path: str,
    gene_name_path: str,
) -> set[int]:
    """Get genes that belong to input GP program and convert them to
    relevant gene token.

    Parameters
    ----------
    gp_genes : list or pd.Series
        List of gene (names) that belong to the current GP.
    do_ensembl_conversion : bool
        Whether to convert gene names to ensembl IDs before converting to tokens.
    gp_name : str
        Label for the current GP (only used for printing).
    gene_token_path : str
        Path to token dictionary (ensembl_id -> token)
    gene_name_path: str
        Path to token name dictionary (gene name -> ensembl_id)

    Returns
    -------
    gp_tokens_set : set of ints
        Set of gene tokens that belong to the current GP.
    """

    # Remove missing values (NaN) from the column
    if isinstance(gp_genes, pd.Series):
        genes = list(gp_genes.dropna())
    else:
        genes = gp_genes

    name_dictionary = pd.read_pickle(gene_name_path)
    token_dictionary = pd.read_pickle(gene_token_path)

    gp_tokens = convert_gene_names_to_tokens(
        genes,
        name_dictionary,
        token_dictionary,
        do_ensembl_conversion,
        gp_name,
    )

    # Remove rare genes
    # rare_genes = []
    # if gene_counts_df is not None:
    #     for t in list(gp_tokens):
    #         if t not in gene_counts_df['token'].tolist():
    #             rare_genes.append(t)
    #             gp_tokens.remove(t)

    #     print(f'In {GP}, dropped {len(rare_genes)} rare genes')

    gp_tokens_set = set(gp_tokens)

    return gp_tokens_set


def count_genes_per_cell(
    dataset: DatasetDict,
    token_dictionary: dict[str, int],
    name_dictionary: dict[str, str],
) -> pd.DataFrame:
    """Count how many times each gene appears in a dataset.

    Parameters
    ----------
    dataset : DatasetDict
        HuggingFace dataset containing tokenized sequences for each cell.
    token_dictionary : dict
        Dictionary relating ensembl ids with token (integer) IDs.
    name_dictionary : dict
        Dictionary relating gene names with ensembl ids.

    Returns
    -------
    token_df : DataFrame
        DataFrame with counts per gene.
    """
    # Of all these genes, how many are present in at least min_cells cells?
    # Extract the 'input_ids' column as a list of lists
    input_ids_lists = dataset['input_ids']  # (n_cells, seq_len), token IDs

    # Flatten the list of lists into a single list
    flat_input_ids = [item for sublist in input_ids_lists for item in sublist]
    # Count the occurrences of each unique value/token ID
    value_counts = Counter(flat_input_ids)

    # Create a DataFrame from the counts
    token_df = pd.DataFrame(
        {'token': list(value_counts.keys()), 'counts': list(value_counts.values())}
    )

    # map tokens back to ENSEMBL IDs and gene names
    token_to_gene = {v: k for k, v in token_dictionary.items()}
    ensembl_to_name = {v: k for k, v in name_dictionary.items()}

    token_df['ensembl'] = token_df['token'].map(token_to_gene)
    token_df['gene'] = token_df['ensembl'].map(ensembl_to_name)

    token_df['total'] = len(dataset)
    token_df['prop'] = token_df['counts'] / token_df['total']

    token_df = token_df[['gene', 'ensembl', 'token', 'counts', 'prop', 'total']]

    return token_df


def build_gp_input_matrix(
    gf: torch.Tensor,
    input_ids: torch.Tensor,
    gp_tokens: torch.Tensor,
    crop_to_gp_len: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a matrix for input to the GP encoder.

    Output matrix is of shape (n_cells, n_gp_tokens, gene_embed_dim), where
    n_gp_tokens is the number of genes in the current GP.

    Zero-valued if a gene does not belong to the current GP;
    i.e., [:, j, :] == 0 for gene j.

    Token selection maintains gene embedding order (i.e., ordered by expression level)

    Parameters
    ----------
    gf : Tensor
        Gene embeddings: Input token embedding sequence.
        shape (n_cells, seq_len, gene_embed_dim)
    input_ids : Tensor
        Tokenization in terms of (gene) token ID:
        shape (n_cells, seq_len).
    gp_tokens : Tensor
        Sequence of gene tokens that belong to the current GP.
        shape (n_gp_tokens,)
    crop_to_gp_len : bool
        Whether to crop the sequence to the max number of non-zero GP genes
        across cells.

    Returns
    -------
    result_matrix : Tensor
        Gene embeddings that belong to the current GP.
        shape (n_cells, seq_len or gp_len, gene_embed_dim)
    masked_labels_output : Tensor
        Input token IDs for the current GP, padded to either seq_len or gp_len.
        shape (n_cells, seq_len or gp_len)
    num_genes_per_cell : Tensor
        Number of genes in the current GP that are active in each cell.
        shape (n_cells,).
    attn_mask : Tensor
        Binary version of masked_labels_output, with an additional sequence position
        at the beginning (1-valued) for the cls token. Ensures that pad tokens and
        non-GP genes are not attended to.
        shape (n_cells, seq_len+1 or gp_len+1).
    """
    # model:
    #     "full_model" : set for input into geneformer
    #     "extract_genes" : when extracting gene embeddings
    #                     -> max size is total GP size
    #     # NEED TO REIMPLEMENT

    # Get list of gp tokens
    # convert gp_tokens bf16 tensor to integers
    # gp_tokens = gp_tokens.to(torch.int)
    gp_tokens = gp_tokens.long()

    # FOR ONE HOT ENCODER VERSION ONLY
    b1, s1 = input_ids.shape
    b2, s2, e2 = gf.shape

    if s1 != s2:
        input_ids = input_ids[:, :s2]

    # Create a binary mask (h, i, k)
    # In cell h, is the gene at position i in our GP at position k?
    # Using broadcasting to compare tokens_arr with gp_tokens

    mask = input_ids.unsqueeze(2) == gp_tokens.unsqueeze(0)
    mask = mask.to(torch.int)

    # Now reshape so that we will zero out non GP genes in each cell
    # Sum along the last dimension to count how many GP tokens each gene matches
    mask_expanded = mask.sum(dim=-1).unsqueeze(2)  # (num_cells, seq_len, 1)
    # how many times does gene i appear in gp_tokens?

    result_matrix = (
        gf * mask_expanded
    )  # (n_cells, seq_len, embed_dim) # zero'd for genes that don't belong to GP

    # Now do the same for labels
    # masked_labels_output = mask.sum(axis=-1) * input_ids
    masked_labels_output = torch.where(
        mask.sum(axis=-1) == 0, torch.zeros_like(input_ids), input_ids
    )  # (n_cells, seq_len) where zero'd for genes that don't feature in GP

    if crop_to_gp_len:
        masked_labels_non_zero = masked_labels_output != 0  # (n_cells, seq_len)
        labels_non_zero = masked_labels_output[masked_labels_non_zero]
        result_matrix_non_zero = result_matrix[masked_labels_non_zero]

        num_genes = masked_labels_non_zero.int().sum(-1)  # (n_cells,)
        max_num_genes = num_genes.max()

        row_indices = torch.repeat_interleave(
            torch.arange(len(num_genes), device=labels_non_zero.device), num_genes
        )
        col_indices = torch.cat(
            [torch.arange(n, device=labels_non_zero.device) for n in num_genes]
        )
        idxs = torch.stack([row_indices, col_indices], dim=0)

        masked_labels_output = torch.sparse_coo_tensor(
            idxs, labels_non_zero, (len(num_genes), max_num_genes)
        ).to_dense()
        result_matrix = torch.sparse_coo_tensor(
            idxs, result_matrix_non_zero, (len(num_genes), max_num_genes, e2)
        ).to_dense()

    # count number of genes per cell
    num_genes_per_cell = mask.sum(axis=-1).sum(axis=-1)

    # Make tensor for forward pass
    # comment the line below to leave 0s because they are actually informative
    # (this gene was not in the top 1000 of this cell)
    # masked_labels_output[masked_labels_output == 0] = -100
    # happens when do LOOKUP

    # Set up attention mask
    # to avoid attention to padding tokens
    attn_mask = torch.zeros_like(masked_labels_output)
    attn_mask[masked_labels_output != 0] = 1

    # never mask cls
    attn_mask = torch.cat(
        [torch.ones_like(attn_mask)[:, 0].unsqueeze(-1), attn_mask], dim=-1
    )

    return result_matrix, masked_labels_output, num_genes_per_cell, attn_mask


def viz_gp(GP, adata, color_by='cell_type', save_to=False):
    """
    Run UMAP on GP embeddings and visualize
    """
    gdata = adata[:, adata.var['gp_idx'].str.startswith(GP)]
    sc.pp.neighbors(gdata, use_rep='X')
    sc.tl.umap(gdata, min_dist=0.4)

    if isinstance(color_by, str):
        color_by = [color_by]

    for c in color_by:
        gp1 = GP.replace('/', '')
        c1 = c.replace('/', '')
        save_path = f'_{save_to}_{gp1}_{c1}.pdf'

        if save_to:
            sc.pl.umap(
                gdata,
                color=c,
                title=f'{GP}',
                save=save_path,
                frameon=False,
            )

        else:
            sc.pl.umap(
                gdata,
                color=c,
                title=f'{GP}',
                frameon=False,
            )


###################################
# For self-attention
###################################


def trunc_normal_(
    tensor: Optional[torch.Tensor] = None, mean=0.0, std=1.0, a=-2.0, b=2.0
):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master
    # until it's in a few official releases - RW
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        v = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * v - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class mlm_mask_generator:
    """
    ## Masked LM (MLM)

    This class implements the masking procedure for a given batch of token sequences.

    adapted from
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/
        blob/master/labml_nn/transformers/mlm/__init__.py
    accessed 03/01/2024

    """

    def __init__(
        self,
        *,
        padding_token: int,
        mask_token: int,
        no_mask_tokens: List[int],
        n_tokens: int,
        masking_prob: float = 0.15,
        randomize_prob: float = 0.1,
        no_change_prob: float = 0.0,
    ):
        """
        * `padding_token` is the padding token `[PAD]`.
          We will use this to mark the labels that shouldn't be used
          for loss calculation.
        * `mask_token` is the masking token `[MASK]`.
        * `no_mask_tokens` is a list of tokens that should not be masked.
        This is useful if we are training the MLM with another task like classification
        at the same time, and we have tokens such as `[CLS]` that shouldn't be masked.
        * `n_tokens` total number of tokens (used for generating random tokens)
        * `masking_prob` is the masking probability
        * `randomize_prob` is the probability of replacing with a random token
        * `no_change_prob` is the probability of replacing with original token
        """
        self.n_tokens = n_tokens
        self.no_change_prob = no_change_prob
        self.randomize_prob = randomize_prob
        self.masking_prob = masking_prob
        # self.no_mask_tokens = no_mask_tokens + [padding_token, mask_token]

        # Convert no_mask_tokens to a set for fast membership checks
        self.no_mask_tokens = torch.tensor(
            no_mask_tokens + [padding_token, mask_token], dtype=torch.long
        )

        self.padding_token = padding_token
        self.mask_token = mask_token

    def __call__(self, x: torch.Tensor):
        """
        * `x` is the batch of input token sequences.
         It's a tensor of type `long` with shape `[seq_len, batch_size]`.
        """
        # Mask `masking_prob` of tokens
        full_mask = torch.rand(x.shape, device=x.device) < self.masking_prob

        # # Unmask `no_mask_tokens`
        # for t in self.no_mask_tokens:
        #     full_mask &= x != t

        # no_mask is True if id is in no_mask_tokens
        # (same shape as x)
        no_mask = (x[..., None] == self.no_mask_tokens.to(x.device)).any(dim=-1)

        full_mask &= ~no_mask

        # A mask for tokens to be replaced with original tokens
        unchanged = full_mask & (
            torch.rand(x.shape, device=x.device) < self.no_change_prob
        )

        # A mask for tokens to be replaced with random tokens
        random_mask = (
            full_mask
            & ~unchanged
            & (torch.rand(x.shape, device=x.device) < self.randomize_prob)
        )

        mask = full_mask & ~unchanged & ~random_mask

        # mask = full_mask

        # Return the masks for processing inside transformer
        return full_mask, mask, random_mask


###################################
# Downstream evaluation
###################################


def evaluate_gene_expr_reconstruction(true_counts, pred_counts, meta, output_dir):
    # shuffle the counts
    true_counts_shuffled = true_counts[torch.randperm(true_counts.size(0))]

    pearson_val = PearsonCorrCoef(num_outputs=true_counts.shape[0]).to(
        true_counts.device
    )

    pearson = pearson_val(pred_counts.T, true_counts.T)
    mean_pearson = torch.mean(pearson)

    pearson_shuffled = pearson_val(pred_counts.T, true_counts_shuffled.T)
    mean_pearson_shuffled = torch.mean(pearson_shuffled)

    # Pearson correlation for non zero genes
    n_cells, n_genes = pred_counts.shape
    mean_pearson_non_zero = []

    for cell_idx in range(n_cells):
        # For each cell, identify non-zero genes
        non_zero_genes = true_counts[cell_idx, :] > 0

        # Filter out zero-expression genes for this cell
        # in both pred and true counts
        pred_non_zero = pred_counts[cell_idx, non_zero_genes]
        true_non_zero = true_counts[cell_idx, non_zero_genes]

        if (
            len(pred_non_zero) > 1
        ):  # Ensure there's more than one gene to calculate Pearson correlation
            # Calculate Pearson correlation for the non-zero genes in this cell
            pearson_corr = torch.corrcoef(torch.stack((pred_non_zero, true_non_zero)))[
                0, 1
            ]
            mean_pearson_non_zero.append(pearson_corr)

    # Compute the mean Pearson correlation across all cells
    mean_pearson_non_zero = torch.tensor(mean_pearson_non_zero).mean()

    # # MSE
    # mse = self.metric['mse'](pred_counts, true_counts)
    # mean_mse = torch.mean(mse)

    # mse_shuffled = self.metric['mse'](pred_counts, true_counts_shuffled)
    # mean_mse_shuffled = torch.mean(mse_shuffled)

    # # set up anndata object for subsetting by condition
    # meta_dict = self.cell_metadata

    # meta_dict.pop('counts', None)
    # meta_dict.pop('size_factor', None)

    if 'batch_key' not in meta.columns:
        meta['batch_key'] = 'single_condition'

    adata_true = sc.AnnData(X=true_counts.cpu().numpy(), obs=meta)
    adata_pred = sc.AnnData(X=pred_counts.cpu().numpy(), obs=meta)

    mmd = evaluate_mmd(adata_true, adata_pred, condition_key='batch_key')

    mmd.to_csv(os.path.join(output_dir, 'global_recon_mmd.csv'))

    emd = evaluate_emd(adata_true, adata_pred, condition_key='batch_key')
    emd.to_csv(os.path.join(output_dir, 'global_recon_emd.csv'))

    # count zero values in true and predicted
    true_zeros = torch.sum(true_counts == 0).item()
    pred_zeros = torch.sum(pred_counts == 0).item()
    true_prop_zeros = true_zeros / true_counts.numel()
    pred_prop_zeros = pred_zeros / pred_counts.numel()

    # write to disk
    metrics_df = pd.DataFrame(
        {
            'metric': [
                'pearson',
                'pearson_shuffled',
                'pearson_non_zero',
                # 'mse',
                # 'mse_shuffled',
                'true_zeros',
                'pred_zeros',
                'true_prop_zeros',
                'pred_prop_zeros',
                'max true counts',
                'max pred counts',
            ],
            'value': [
                mean_pearson.item(),
                mean_pearson_shuffled.item(),
                mean_pearson_non_zero.item(),
                # mean_mse.item(),
                # mean_mse_shuffled.item(),
                true_zeros,
                pred_zeros,
                true_prop_zeros,
                pred_prop_zeros,
                true_counts.max().item(),
                pred_counts.max().item(),
            ],
        }
    )

    metrics_df.to_csv(
        os.path.join(output_dir, 'random_baseline_metrics.csv'),
        index=False,
    )

    return metrics_df


def wrangle_classification_report(report):
    # Prepare dataframe for output
    # Initialize empty lists for each column
    output_label = []
    metrics = []
    values = []

    # Iterate through the dictionary to extract the data
    for output_class, metrics_dict in report.items():
        if output_class != 'accuracy':
            for metric, value in metrics_dict.items():
                output_label.append(output_class)
                metrics.append(metric)
                values.append(value)

    # Save to disk
    output_df = pd.DataFrame(
        {'output_class': output_label, 'metric': metrics, 'value': values}
    )
    output_df['accuracy'] = report['accuracy']

    return output_df


def subsample_to_rarest_category(adata, col):
    # Get the counts of each category in the 'celltype' column
    category_counts = adata.obs[col].value_counts()

    # Get the number of observations for the rarest category
    rarest_category_count = category_counts.min()

    # Initialize a list to store subsampled DataFrames
    ad_holder = []

    # Subsample each category to the rarest category count and append to the list
    for category in category_counts.index:
        indices_to_keep = adata.obs.index[adata.obs[col] == category]
        subsampled_indices = np.random.choice(
            indices_to_keep, rarest_category_count, replace=False
        )
        sdata = adata[subsampled_indices, :].copy()
        ad_holder.append(sdata)

    # Concatenate the list of DataFrames into a single AnnData object
    subsampled_adata = ad.concat(ad_holder)

    return subsampled_adata


def do_logistic_regression(
    adata,
    labels_var,
    output_directory,
    filename,
    variable_to_track=None,
    hparam_to_track=None,
):
    """
    Logistic regression for mutlinormial classification based on embeddings in adata.X
    """
    # Split training and testing data
    train_idx, test_idx = train_test_split(
        range(len(adata)), test_size=0.2, stratify=adata.obs[labels_var]
    )

    # Get train and test data -
    # nb this works because adata.obs indices
    # are initialised when we get cell embeddings
    # not cell barcodes
    train_data = adata.X[train_idx, :]
    test_data = adata.X[test_idx, :]
    train_labels = adata.obs[labels_var][train_idx]
    test_labels = adata.obs[labels_var][test_idx]

    # Train classifier
    clf = LogisticRegression(max_iter=10_000, multi_class='multinomial').fit(
        train_data, train_labels
    )

    # Predict on test set
    pred_labels = clf.predict(test_data)

    # Get classification report
    report = classification_report(test_labels, pred_labels, output_dict=True)

    output_df = wrangle_classification_report(report)

    if variable_to_track is not None:
        for k, v in variable_to_track.items():
            output_df[k] = v

    output_df.to_csv(os.path.join(output_directory, f'{filename}.csv'), index=False)


def do_linear_regression(
    adata,
    labels_var,
    output_directory,
    filename,
    variable_to_track=None,
):
    """
    Linear regression for continuous regression based on embeddings in adata.X
    """
    # Split training and testing data
    train_idx, test_idx = train_test_split(
        range(len(adata)), test_size=0.2, random_state=42
    )

    # Get train and test data
    train_data = adata.X[train_idx, :]
    test_data = adata.X[test_idx, :]
    train_labels = adata.obs[labels_var][train_idx]
    test_labels = adata.obs[labels_var][test_idx]

    # Train regressor
    reg = LinearRegression().fit(train_data, train_labels)

    # Predict on test set
    pred_labels = reg.predict(test_data)

    # Get mean squared error
    mse = mean_squared_error(test_labels, pred_labels)

    # Get coefficient of determination
    r2 = r2_score(test_labels, pred_labels)

    # Prepare dataframe for output
    output_df = pd.DataFrame(
        {
            'metric': ['Mean Squared Error', 'R2'],
            'value': [mse, r2],
        }
    )

    if variable_to_track is not None:
        for k, v in variable_to_track.items():
            output_df[k] = v

    # Save to disk
    output_df.to_csv(os.path.join(output_directory, f'{filename}.csv'), index=False)


def evaluate_clustering(
    adata, gene_name, output_dir, metrics_filename, plot=False, plot_filename=None
):
    """
    Evaluate clustering performance
    """
    print('Computing clusters...')
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.2)

    # Visualize
    if plot:
        sc.pl.umap(adata, color='leiden', save=f'{plot_filename}_leiden.pdf')
        sc.pl.umap(adata, color='GP', save=f'{plot_filename}_GP.pdf')

    # Evaluate clustering performance
    # Using ARS, NMI, and Silhouette score
    print('Running cluster evaluation metrics...')
    ari = adjusted_rand_score(adata.obs['leiden'], adata.obs['GP'])
    nmi = normalized_mutual_info_score(adata.obs['leiden'], adata.obs['GP'])
    sil = silhouette_score(adata.obsm['X_umap'], adata.obs['GP'])

    # Save to disk
    output_df = pd.DataFrame(
        {
            'gene': gene_name,
            'metric': ['ARI', 'NMI', 'Silhouette'],
            'value': [ari, nmi, sil],
        }
    )
    output_df.to_csv(
        f'{output_dir}/{metrics_filename}_clustering_metrics.csv', index=False
    )
    print('...done!')

    return adata


def evaluate_clustering_cells(adata):
    if 'leiden' not in adata.obs.columns:
        sc.tl.leiden(adata)

    print('Running cluster evaluation metrics...')
    ari_ct = adjusted_rand_score(adata.obs['leiden'], adata.obs['cell_type'])
    ari_cond = adjusted_rand_score(adata.obs['leiden'], adata.obs['condition'])
    sil = silhouette_score(adata.obsm['X_umap'], adata.obs['leiden'])
    db = davies_bouldin_score(adata.obsm['X_umap'], adata.obs['leiden'])

    output_df = pd.DataFrame(
        {
            'metric': [
                'ARI_cell',
                'ARI_env',
                'Silhouette_leiden',
                'Davies_Bouldain_leiden',
            ],
            'value': [ari_ct, ari_cond, sil, db],
        }
    )

    return output_df


def remove_single_data_points(adata, obs_column):
    """

    Given an anndata object,
    drop the cells which are the only data point
    for a given value in a given obs column

    """
    # Count occurrences of obs values
    value_counts = adata.obs[obs_column].value_counts()

    # Get values with a count of one
    values_to_remove = value_counts[value_counts == 1].index

    # Filter cells with values that have a count of one
    cells_to_remove = adata.obs[adata.obs[obs_column].isin(values_to_remove)].index

    # Create a new Anndata object without the cells to remove
    filtered_anndata = adata[~adata.obs.index.isin(cells_to_remove)]

    return filtered_anndata


# -----------------------------------------------------
# (gene, GP) cosine similarity
# -----------------------------------------------------


# ---------- helpers ----------
def _bh_with_nans(pvals: pd.Series, alpha=0.05, method='fdr_bh'):
    """Benjamini–Hochberg while preserving NaNs."""
    pvals = pd.Series(pvals, index=pvals.index)
    mask = pvals.notna() & np.isfinite(pvals)
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    if mask.any():
        _, padj, _, _ = multipletests(pvals[mask].values, alpha=alpha, method=method)
        out.loc[mask] = padj
    return out


def _stars(p):
    if not np.isfinite(p):
        return ''
    return '***' if p < 1e-3 else ('**' if p < 1e-2 else ('*' if p < 5e-2 else ''))


def _to_csc(M):
    if sp.issparse(M):
        return M if sp.isspmatrix_csc(M) else M.tocsc(copy=False)
    return sp.csc_matrix(M)


# ---------- core sparse stats ----------
def _t_equal_var_sparse(X_ref, X_query):
    """
    Compute per-gene means and pooled-variance t-test directly
    from sparse/dense matrices.
    Implicit zeros are treated as 0.
    Returns dict(mean_ref, mean_query, effect, p_value).
    """
    R = _to_csc(X_ref)
    Q = _to_csc(X_query)

    m, n = R.shape[0], Q.shape[0]
    assert R.shape[1] == Q.shape[1]
    G = R.shape[1]

    # sums and squared sums (zeros implicit)
    sum_R = np.asarray(R.sum(axis=0)).ravel()
    sum_Q = np.asarray(Q.sum(axis=0)).ravel()
    sumsq_R = np.asarray(R.power(2).sum(axis=0)).ravel()
    sumsq_Q = np.asarray(Q.power(2).sum(axis=0)).ravel()

    mean_R = sum_R / max(m, 1)
    mean_Q = sum_Q / max(n, 1)
    effect = mean_Q - mean_R

    # sample variances (unbiased). if group size <2 -> NaN variance
    var_R = np.full(G, np.nan)
    var_Q = np.full(G, np.nan)
    if m > 1:
        var_R = (sumsq_R - (sum_R**2) / m) / (m - 1)
        var_R[var_R < 0] = 0.0  # numerical guard
    if n > 1:
        var_Q = (sumsq_Q - (sum_Q**2) / n) / (n - 1)
        var_Q[var_Q < 0] = 0.0

    p_value = np.full(G, np.nan)
    if m > 1 and n > 1 and (m + n - 2) > 0:
        sp2 = ((m - 1) * var_R + (n - 1) * var_Q) / (m + n - 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            se = np.sqrt(sp2 * (1.0 / m + 1.0 / n))
            t = effect / se
        df = m + n - 2
        p_value = 2.0 * student_t.sf(np.abs(t), df)

        # handle 0/NaN SE: if effect==0 => p=1; if effect!=0 => p=0 (perfect separation)
        bad = ~np.isfinite(se) | (se == 0)
        if bad.any():
            same = bad & (np.abs(effect) == 0)
            diff = bad & (np.abs(effect) > 0)
            p_value[same] = 1.0
            p_value[diff] = 0.0

    return dict(mean_ref=mean_R, mean_query=mean_Q, effect=effect, p_value=p_value)


def assign_bar_colors(genes, gp_to_color, gpdb):
    colors = []
    for gene in genes:
        color_assigned = 'gray'
        for gp, color in gp_to_color.items():
            if gene in gpdb[gp].values:
                color_assigned = color
                break
        colors.append(color_assigned)
    return colors


def _resolve_sig_colors(
    significance_palette, default_ref='tab:red', default_query='tab:blue'
):
    """
    Accepts either:
      - a matplotlib cmap name (str), or
      - a list/tuple of two color specs [ref_color, query_color] for gradient mode.
    Returns (is_listlike, ref_color, query_color, cmap_name_or_None)
    """
    if (
        isinstance(significance_palette, (list, tuple))
        and len(significance_palette) >= 2
    ):
        ref_col = significance_palette[0]
        qry_col = significance_palette[1]
        return True, ref_col, qry_col, None
    elif isinstance(significance_palette, str):
        return False, default_ref, default_query, significance_palette
    else:
        # sensible fallback
        return False, default_ref, default_query, 'Blues'


#################
# GP wrangling
#################


def make_overlap_matrix(df, save_to=None):
    # Initialize a matrix to store intersection values
    intersection_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    # Calculate intersection over length of non-null elements
    for i in tqdm(df.columns, desc='Calculating overlap', leave=False):
        for j in df.columns:
            intersection = len(set(df[i].dropna()) & set(df[j].dropna()))
            intersection_ratio = (
                intersection / len(df[i].dropna()) if len(df[i].dropna()) > 0 else 0
            )
            intersection_matrix.loc[i, j] = intersection_ratio

    if save_to:
        np.save(save_to, intersection_matrix)

    return intersection_matrix


def make_similarity_matrix(df, save_to=None):
    # Initialize a matrix to store intersection values
    intersection_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    # Calculate intersection over length of non-null elements
    for i in tqdm(df.columns):
        for j in df.columns:
            intersection = len(set(df[i].dropna()) & set(df[j].dropna()))
            intersection_ratio = (
                intersection / len(df[i].dropna()) if len(df[i].dropna()) > 0 else 0
            )
            intersection_matrix.loc[i, j] = intersection_ratio

    # Set diagonal values to 0 for visualization
    np.fill_diagonal(intersection_matrix.values, 0)

    # Normalize each row to ensure they sum up to 1
    row_sums = intersection_matrix.sum(axis=1)

    n_columns = intersection_matrix.shape[1]  # Number of columns in the matrix
    row_sums_nonzero = np.where(row_sums != 0, row_sums, 1)  # Replace zero sums with 1

    # Divide each element in the matrix by its corresponding row sum (if not zero)
    normalized_matrix = intersection_matrix.div(row_sums_nonzero, axis=0)

    # Replace rows where row_sums are zero with 1/n_columns
    row_sums_zero_mask = row_sums == 0
    normalized_matrix[row_sums_zero_mask] = 1 / n_columns

    if save_to:
        np.save(save_to, normalized_matrix)

    return normalized_matrix


def intersection_heatmap(df, save_to=None):
    # Initialize a matrix to store intersection values
    intersection_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    # Calculate intersection over length of non-null elements
    for i in tqdm(df.columns, desc='Calculating overlap', leave=False):
        for j in df.columns:
            intersection = len(set(df[i].dropna()) & set(df[j].dropna()))
            intersection_ratio = (
                intersection / len(df[i].dropna()) if len(df[i].dropna()) > 0 else 0
            )
            intersection_matrix.loc[i, j] = intersection_ratio

    # Set diagonal values to 0 for visualization
    np.fill_diagonal(intersection_matrix.values, 0)

    # Create the heatmap
    plt.figure(figsize=(16, 15))
    ax = sns.heatmap(
        intersection_matrix.astype(float), annot=False, cmap='coolwarm', fmt='.2f'
    )

    # Adjust x-axis ticks to display every label
    ax.set_xticks(np.arange(len(intersection_matrix.columns)) + 0.5)
    ax.set_xticklabels(intersection_matrix.columns, rotation=90)

    ax.set_yticks(np.arange(len(intersection_matrix.columns)) + 0.5)
    ax.set_yticklabels(intersection_matrix.columns)  # , rotation=90)

    plt.title('Overlap of selected pathways')
    plt.tight_layout()
    plt.show()

    if save_to:
        plt.savefig(save_to, dpi=300)


# ------------------------------------------------------------------
# Gears utils functions
# from https://github.com/snap-stanford/GEARS/blob/master/gears/utils.py
# Accessed 17/06/2024
# ------------------------------------------------------------------


def print_sys(s):
    """system print

    Args:
        s (str): the string to print
    """
    print(s, flush=True, file=sys.stderr)


def tar_data_download_wrapper(url, save_path, data_path):
    """
    Wrapper for tar file download

    Args:
        url (str): the url of the dataset
        save_path (str): the path where the file is donwloaded
        data_path (str): the path to save the extracted dataset

    """

    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        dataverse_download(url, save_path + '.tar.gz')
        print_sys('Extracting tar file...')
        with tarfile.open(save_path + '.tar.gz') as tar:
            tar.extractall(path=data_path)
        print_sys('Done!')


def dataverse_download(url, save_path):
    """
    Dataverse download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """

    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        print_sys('Downloading...')
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()


def make_GO(data_path, pert_list, data_name, num_workers=25, save=True):
    """
    Creates Gene Ontology graph from a custom set of genes
    """

    # fname = './data/go_essential_' + data_name + '.csv'
    fname = 'go_essential_' + data_name + '.csv'
    if os.path.exists(fname):
        return pd.read_csv(fname)

    with open(os.path.join(data_path, 'gene2go_all.pkl'), 'rb') as f:
        gene2go = pickle.load(f)

    gene2go = {i: gene2go[i] for i in pert_list if i in gene2go.keys()}
    print(f'{len(pert_list) - len(gene2go)} genes not found in gene2go file')  # noqa

    print('Creating custom GO graph, this can take a few minutes')
    with Pool(num_workers) as p:
        all_edge_list = list(
            tqdm(
                p.imap(get_GO_edge_list, ((g, gene2go) for g in gene2go.keys())),
                total=len(gene2go.keys()),
            )
        )
    edge_list = []
    for i in all_edge_list:
        edge_list = edge_list + i

    df_edge_list = pd.DataFrame(edge_list).rename(
        columns={0: 'source', 1: 'target', 2: 'importance'}
    )

    if save:
        print('Saving edge_list to file')
        df_edge_list.to_csv(fname, index=False)

    return df_edge_list


def get_GO_edge_list(args):
    """
    Get gene ontology edge list
    """
    g1, gene2go = args
    edge_list = []
    for g2 in gene2go.keys():
        score = len(gene2go[g1].intersection(gene2go[g2])) / len(
            gene2go[g1].union(gene2go[g2])
        )
        if score > 0.1:
            edge_list.append((g1, g2, score))
    return edge_list


def get_similarity_network(
    data_path, data_name, k, default_pert_graph=True, pert_list=None
):
    '''
    Modified to only include GO version
    '''

    if default_pert_graph:
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934319'
        tar_data_download_wrapper(
            server_path, os.path.join(data_path, 'go_essential_all'), data_path
        )
        df_jaccard = pd.read_csv(
            os.path.join(data_path, 'go_essential_all/go_essential_all.csv')
        )

    else:
        df_jaccard = make_GO(data_path, pert_list, data_name)

    df_out = (
        df_jaccard.groupby('target')
        .apply(lambda x: x.nlargest(k + 1, ['importance']))
        .reset_index(drop=True)
    )

    return df_out


#################
# Scheduling
#################


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    """
    from https://github.com/facebookresearch/dino/blob/main/utils.py
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class WDScheduler(pl.Callback):
    def __init__(self, weight_decay, weight_decay_end, epochs, data_loader):
        super().__init__()
        self.wd_schedule = cosine_scheduler(
            weight_decay, weight_decay_end, epochs, len(data_loader)
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """
        adapted from
        https://github.com/facebookresearch/dino/blob/main/main_dino.py#L301
        """
        global_iteration = trainer.global_step  # Get the global training iteration
        wd = self.wd_schedule[global_iteration]

        optimizer = trainer.optimizers[0]  # we only use one optimizer
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['weight_decay'] = wd


class CosineLRwithWarmUp(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=eta_min,
            last_epoch=last_epoch - warmup_epochs,
        )

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            return self.cosine_scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.last_epoch >= self.warmup_epochs:
            self.cosine_scheduler.step(epoch - self.warmup_epochs)
        else:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr


class FrequentLoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        # Ensure that train/val_loss is logged after validation step
        pl_module.log(
            'val/intermediate_loss',
            pl_module.current_val_loss,
            on_step=True,
            on_epoch=False,
        )


###################################
# Attributions helper functions
###################################


def summarize_attributions(attributions):
    '''
    from https://captum.ai/tutorials/Bert_SQUAD_Interpret
    '''
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


###################################
# Differential transformer
###################################

# Copyright (c) 2023, Tri Dao.
# from https://github.com/microsoft/unilm/blob/master/
# Diff-Transformer/kernel/rotary.py


@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    nheads,
    rotary_dim,
    seqlen_ro,
    CACHE_KEY_SEQLEN,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X,
        # do calculation, then store to 1st and 2nd halves of OUT
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(
            X,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x1 = tl.load(
            X + rotary_dim_half * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        # write back result
        OUT = OUT + (
            rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim
        )
        tl.store(
            OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half)
        )
        tl.store(
            OUT + rotary_dim_half * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        # We don't want to load X[0, 2, 4, ...] and X[1, 3, 5, ...]
        # separately since both are slow.
        # Instead, we load x0 = X[0, 1, 2, 3, ...] and x1 = X[1, 0, 3, 2, ...].
        # Loading x0 will be fast but x1 will be slow.
        # Then we load cos = COS[0, 0, 1, 1, ...]
        # and sin = SIN[0, 0, 1, 1, ...].
        # Then we do the calculation and use tl.where
        # to pick put the right outputs for the even
        # and for the odd indices.
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(
            X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert (
            max_seqlen is not None
        ), 'If cu_seqlens is passed in, then max_seqlen must be passed'
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]  # type: ignore
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, 'rotary_dim must be <= headdim'
    assert headdim <= 256, 'Only support headdim <= 256'
    assert seqlen_ro >= seqlen, 'seqlen_ro must be >= seqlen'

    assert (
        cos.dtype == sin.dtype
    ), f'cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}'
    assert (
        x.dtype == cos.dtype
    ), f'Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}'

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = (
        32
        if rotary_dim <= 32
        else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    )
    grid = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), batch, nheads)  # noqa
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](
            output,  # data ptrs
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,  # shapes
            nheads,
            rotary_dim,
            seqlen_ro,
            seqlen // 128,  # key for triton cache (limit number of compilations)
            output.stride(0)
            if not is_varlen
            else 0,  # batch_strides if not varlen else 0
            output.stride(-3),  # seqlen_stride or total_seqlen_stride
            output.stride(-2),  # nheads_stride
            output.stride(-1),  # headdim_stride
            x.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            x.stride(-3),  # seqlen stride or total_seqlen_stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride
            BLOCK_K,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M,
        )
    return output


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            # Can't save int with save_for_backward
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
        # "[CUDA]: invalid device context", and cloning makes it work.
        # Idk why. Triton 2.1.0 works.
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd
            dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int.
            Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )


class RMSNorm(nn.Module):
    '''
    From https://github.com/microsoft/unilm/blob/master/Diff-Transformer/rms_norm.py
    '''

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine=True,
        memory_efficient=False,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, eps={self.eps}, '
            f'elementwise_affine={self.elementwise_affine}'
        )
