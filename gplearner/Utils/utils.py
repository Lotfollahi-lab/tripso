import argparse
import glob
import math
import os
import pickle
import random
import warnings
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR

random.seed(0)

###################################
# Generic
###################################


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
            f'{supervised_tag} found in {checkpoint_dir}.'
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


###################################
# Wrangling hugging face dataset
###################################


def encode_labels(input_data, input_col, new_col):
    """
    Encode labels as integers
    """
    label_values = list(set(input_data[input_col]))
    label_dict = {l: i for i, l in enumerate(label_values)}

    def classes_to_ids(example):
        example[new_col] = label_dict[example[input_col]]
        return example

    labeled_dataset = input_data.map(classes_to_ids, num_proc=16)

    return labeled_dataset


def do_balanced_downsampling(class_values, input_data, n_cells_per_class):
    """
    Perform balanced subsampling of input data

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

# for converting between gene formats
# load gene token dict
with open(
    '/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/'
    'Geneformer/geneformer/token_dictionary.pkl',
    'rb',
) as f:
    token_dictionary = pickle.load(f)

# load gene name to ensembl dict
with open(
    '/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/'
    'Geneformer/geneformer/gene_name_id_dict.pkl',
    'rb',
) as f:
    name_dictionary = pickle.load(f)


def get_gp_tokens(
    GP, db, do_ensembl_conversion, gene_counts_df, gene_token_path, gene_name_path
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

    gene_counts_df : pd.DataFrame
        DataFrame containing counts of each gene in the dataset

    """
    with open(gene_token_path, 'rb') as f:
        token_dictionary = pickle.load(f)

    # load gene name to ensembl dict
    with open(gene_name_path, 'rb') as f:
        name_dictionary = pickle.load(f)

    # Check if GP exists in the reactome columns
    if GP not in db.columns:
        raise ValueError(f'{GP} not found in {db}.')

    # Extract the column 'GP' from the DataFrame
    gp_column = db[GP]

    # Remove missing values (NaN) from the column
    genes = list(gp_column.dropna())

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
        print(f"In {GP}, dropped {gp_tokens.count('Unknown')} unknown genes")
        while 'Unknown' in gp_tokens:
            gp_tokens.remove('Unknown')

    # Remove rare genes
    rare_genes = []
    if gene_counts_df is not None:
        for t in list(gp_tokens):
            if t not in gene_counts_df['token'].tolist():
                rare_genes.append(t)
                gp_tokens.remove(t)

        print(f'In {GP}, dropped {len(rare_genes)} rare genes')

    gp_tokens_set = set(gp_tokens)

    return gp_tokens_set


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
        no_change_prob: float = 0.1,
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

        mask = full_mask & ~unchanged

        # Return the masks for processing inside transformer
        return mask


###################################
# Downstream evaluation
###################################


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

    if variable_to_track is not None:
        for k, v in variable_to_track.items():
            output_df[k] = v

    # if hparam_to_track is not None:
    #     # convert to list for iteration
    #     if type(hparam_to_track) is not list:
    #         hparam_to_track = [hparam_to_track]
    #     for h in hparam_to_track:
    #         output_df[h] = getattr(self, h)

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
