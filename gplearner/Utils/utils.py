import argparse
import glob
import math
import os
import pickle
import random
import re
import sys
import tarfile
import warnings
from collections import Counter
from itertools import combinations
from multiprocessing import Pool
from typing import List, Optional

import anndata as ad
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import requests  # type: ignore
import scanpy as sc
import seaborn as sns
import torch
from datasets import concatenate_datasets, load_from_disk
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import PearsonCorrCoef
from tqdm import tqdm

from ..Metrics.metrics import evaluate_emd, evaluate_mmd

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


def do_balanced_downsampling(class_values, input_data, n_cells_per_class):
    """
    Perform balanced subsampling of input data
    for Huggingface dataset class

    """
    # Calculate class frequencies
    class_counts = Counter(class_values)

    # Perform balanced subsampling
    balanced_samples = []
    for label, count in class_counts.items():
        subsample_count = min(count, n_cells_per_class)
        class_indices = [i for i, l in enumerate(class_values) if l == label]
        subsample_indices = random.sample(class_indices, subsample_count)
        balanced_samples.extend(subsample_indices)

    input_data = input_data.select(balanced_samples)

    return input_data


def do_balanced_downsampling_anndata(adata, subsample_by, n_cells_per_class):
    """
    Perform balanced subsampling of input data

    """
    # Calculate class frequencies
    class_counts = adata.obs[subsample_by].value_counts()

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

GENE_NAME_FILE = '/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/Geneformer/geneformer/gene_name_id_dict.pkl'  # noqa


# for converting between gene formats
# load gene token dict
with open(TOKEN_DICTIONARY_FILE, 'rb') as f:
    token_dictionary = pickle.load(f)

# load gene name to ensembl dict
with open(GENE_NAME_FILE, 'rb') as f:
    name_dictionary = pickle.load(f)

ensembl_to_name = {v: k for k, v in name_dictionary.items()}
token_to_gene = {v: k for k, v in token_dictionary.items()}


def get_gp_tokens(
    gp_genes,
    do_ensembl_conversion,
    gp_name,
    gene_token_path=TOKEN_DICTIONARY_FILE,
    gene_name_path=GENE_NAME_FILE,
):
    """
    Get genes that belong to input GP program
    and convert them to relevant geneformer token

    Inputs are:
    GP : str
        Gene program name

    db : pd.DataFrame
        Gene program database where GP names are columns

    do_ensembl_conversion : bool
        Whether to convert gene names to ensembl IDs before converting to tokens

    gp_name : str
        Label for the GP of interest (only used for printing)

    """
    with open(gene_token_path, 'rb') as f:
        token_dictionary = pickle.load(f)

    # load gene name to ensembl dict
    with open(gene_name_path, 'rb') as f:
        name_dictionary = pickle.load(f)

    # Remove missing values (NaN) from the column
    if isinstance(gp_genes, pd.Series):
        genes = list(gp_genes.dropna())
    else:
        genes = gp_genes

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


def count_genes_per_cell(dataset):
    # Of all these genes, how many are present in at least min_cells cells?
    # Extract the 'input_ids' column as a list of lists
    input_ids_lists = dataset['input_ids']

    # Flatten the list of lists into a single list
    flat_input_ids = [item for sublist in input_ids_lists for item in sublist]
    # Count the occurrences of each unique value
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


def find_gene_intersection(df, column_combination):
    genes = set(df[column_combination[0]])
    for col in column_combination[1:]:
        genes = genes.intersection(df[col])
    return genes


def find_genes_in_single_gp(df):
    all_genes = set()
    genes_in_single_column = set()

    for col in df.columns:
        col_genes = set(df[col])
        genes_in_single_column.update(col_genes - all_genes)
        all_genes.update(col_genes)

    return list(genes_in_single_column)


def find_genes_in_multiple_gp(
    gp_inputs, gpdb, token_df, do_ensembl_conversion, min_cells, downsample_to_n_genes
):
    """
    Get genes that belong to more than one GP program
    and convert them to relevant geneformer token
    """

    # Create a set to store genes present in more than one column
    common_genes_set = set()

    # Loop through different pairs of columns (2 to 5)
    for num_columns in range(2, len(gp_inputs)):
        column_combinations = combinations(gpdb.columns, num_columns)
        for combination in column_combinations:
            common_genes = find_gene_intersection(gpdb, combination)
            common_genes_set.update(common_genes)

    if np.nan in common_genes_set:
        common_genes_set.remove(np.nan)

    print(f'Union of genes present in more than one GP: {len(common_genes_set)}')

    if do_ensembl_conversion:
        # then common_genes_set is storing gene names
        token_multi = token_df[token_df['gene'].isin(common_genes_set)]
    else:
        # then common_genes_set is storing ensembl IDs
        token_multi = token_df[token_df['ensembl'].isin(common_genes_set)]

    print(
        f'Range of counts: {token_multi["counts"].min()}'
        f'- {token_multi["counts"].max()}'
    )

    token_multi = token_multi[token_multi['counts'] > min_cells]
    tokens_to_keep = token_multi['token'].tolist()

    print(
        'Number of genes present in more than one GP'
        f'and at least {min_cells} cells: {len(token_multi)}'
    )

    if len(token_multi) == 0:
        raise ValueError(
            'No genes present in more than one GP '
            f'are present in at least {min_cells} cells.'
            'Please relax the threshold.'
        )

    if downsample_to_n_genes:
        if downsample_to_n_genes < len(tokens_to_keep):
            print(f'Downsampling to {downsample_to_n_genes} genes')
            print('')
            tokens_to_keep = random.sample(tokens_to_keep, downsample_to_n_genes)

    return tokens_to_keep


def get_genes_in_single_gp(gpdb, do_ensembl_conversion, downsample_to_n_genes):
    genes_in_single_gp = find_genes_in_single_gp(gpdb)

    if np.nan in genes_in_single_gp:
        genes_in_single_gp.remove(np.nan)

    if do_ensembl_conversion:
        genes_in_single_gp = [
            name_dictionary[g]
            for g in genes_in_single_gp
            if g in name_dictionary.keys()
        ]

    tokens_to_keep = [
        token_dictionary[g] for g in genes_in_single_gp if g in token_dictionary.keys()
    ]

    print(len(genes_in_single_gp), 'genes present in exactly one GP')

    if downsample_to_n_genes:
        print(f'Downsampling to {downsample_to_n_genes} genes')
        print('')
        tokens_to_keep = random.sample(tokens_to_keep, downsample_to_n_genes)

    return tokens_to_keep


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
        self.no_mask_tokens = no_mask_tokens + [padding_token, mask_token]
        self.padding_token = padding_token
        self.mask_token = mask_token

    def __call__(self, x: torch.Tensor):
        """
        * `x` is the batch of input token sequences.
         It's a tensor of type `long` with shape `[seq_len, batch_size]`.
        """
        # Mask `masking_prob` of tokens
        full_mask = torch.rand(x.shape, device=x.device) < self.masking_prob

        # Unmask `no_mask_tokens`
        for t in self.no_mask_tokens:
            full_mask &= x != t

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
    print(f'{len(pert_list) - len(gene2go)} genes not found in gene2go file')

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
