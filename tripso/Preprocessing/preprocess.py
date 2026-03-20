import glob
import os
import random
import warnings
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
import scanpy as sc
from datasets import concatenate_datasets, load_from_disk

from .. import ENSEMBL_DICTIONARY_FILE
from ..Utils.geneformer_utils import get_gf_repo
from ..Utils.utils import do_balanced_downsampling_anndata, encode_labels
from .geneformer_tokenizer import TranscriptomeTokenizer
from .tokenizer import GPTokenizer

seed = 0
np.random.seed(seed)
random.seed(seed)

# Search for Geneformer in the site-packages directories
geneformer_repo_path = get_gf_repo()


def pp_and_tokenize(
    root_dir: str,
    adata_path: Optional[str] = None,
    input_size: int = 2048,
    vars_to_keep: Union[Dict, List] = ['cell_type'],
    subsample_by: Optional[List] = ['cell_type'],
    n_cells_per_class: int = 20_000,
    chunk_size: int = 50_000,
    name_tag: Optional[str] = 'Reactome',
    cov_to_encode: Union[List[str], str] = ['cell_type', 'condition'],
    batch_keys: Optional[List[str]] = None,
    tissue: Optional[str] = None,
    hvg_batch_key: Optional[str] = None,
    save_gp_genes_object: Optional[bool] = False,
    calculate_hvg: Optional[bool] = True,
    do_tokenization: Optional[bool] = True,
    use_gp_tokenizer: Optional[bool] = False,
    do_ensembl_conversion: Optional[bool] = True,
    gp_genes_union: Optional[List[str]] = None,
    output_data_name: Optional[str] = None,
):
    """Preprocess and tokenize scRNA-seq data for GPformer training.

    This function performs the complete preprocessing pipeline including:
    - Loading and optionally subsampling AnnData objects
    - Optionally calculating highly variable genes (HVGs)
    - Splitting data into chunks (for reasonable RAM usage)
    - Tokenizing data using Geneformer or custom tokenizer
    - Encoding categorical covariates
    - Optionally saving GP genes subset AnnData object

    Parameters
    ----------
    root_dir : str
        Output directory where processed h5ad and tokenized data will be saved.
        Directory structure will be created as: root_dir/data/processed/
    adata_path : str, optional
        Path to input AnnData h5ad file to preprocess and tokenize.
    input_size : int, default=2048
        Maximum input sequence length for tokenization. Determines which
        Geneformer token dictionary to use (2048 or 4096).
    vars_to_keep : dict or list, default=['cell_type']
        Metadata column names from adata.obs to retain in tokenized dataset.
        If dict, maps obs column names to output column names.
    subsample_by : list of str, optional, default=['cell_type']
        Metadata columns to use for balanced downsampling. Set to None to
        skip subsampling. Multiple columns will be combined.
    n_cells_per_class : int, default=20000
        Minimum number of cells to keep per class during balanced subsampling.
        Classes with fewer cells will keep all available cells.
    chunk_size : int, default=50000
        Number of cells per chunk when splitting large datasets for tokenization.
    name_tag : str, default='Reactome'
        Identifier tag for gene program database filename (gpdb_{name_tag}.csv).
    cov_to_encode : str or list of str, default=['cell_type', 'condition']
        Metadata columns to encode as integer IDs (creates {column}_id columns).
    batch_keys : list of str, optional
        Metadata columns to combine into a 'batch_key' column (joined with '_').
        Used for HVG calculation, and will be used in decoder
        of count reconstruction step.
    tissue : str, optional
        Tissue type identifier for naming output files. If None, uses last
        component of root_dir path.
    hvg_batch_key : str, optional
        Column name to use as batch key for highly variable gene calculation.
        If None and batch_keys provided, uses 'batch_key'.
    save_gp_genes_object : bool, default=False
        Whether to save a separate h5ad file containing only genes from gene
        programs in gpdb_{name_tag}.csv.
    calculate_hvg : bool, default=True
        Whether to calculate highly variable genes using Seurat v3 method
        (top 2000 genes) and subset to HVGs.
    do_tokenization : bool, default=True
        Whether to perform tokenization step. Set to False to only preprocess.
    use_gp_tokenizer : bool, default=False
        Whether to use GPTokenizer (True) or standard TranscriptomeTokenizer
        (False) for tokenization.
    do_ensembl_conversion : bool, default=True
        Whether to convert gene names to Ensembl IDs during tokenization.
    gp_genes_union : list of str, optional
        Union of all GP genes to be used in tokenizer. If None and
        use_gp_tokenizer=True, will be loaded from gpdb_{name_tag}.csv.
    output_data_name : str, optional
        Custom name for output dataset directory. If None, uses 'input_dataset'.

    Raises
    ------
    ValueError
        If adata_path is not provided and no existing h5ad found in root_dir.
        If hvg_batch_key cannot be determined when calculate_hvg=True.
        If no GP genes found in dataset when save_gp_genes_object=True.
        If gpdb_{name_tag}.csv file not found in root_dir.

    Notes
    -----
    Expected gene program database format: CSV file where each column represents
    a gene program and contains gene identifiers (one per row).

    Output directory structure:
        root_dir/
            data/processed/
                input_h5ad/          - Preprocessed h5ad files
                tokenized/           - Tokenized datasets
                input_dataset/       - Final encoded dataset
                    or {output_data_name}/
            gpdb_{name_tag}.csv      - Gene program database (must exist)
    """
    # Step 1 : Tokenize data

    if tissue is None:
        tissue = root_dir.split('/')[-1]

    # check for anndata object in input_h5ad directory
    if not os.path.exists(os.path.join(root_dir, 'data/processed/input_h5ad')):
        if adata_path is None:
            raise ValueError('Please provide path to anndata object')

        adata = sc.read_h5ad(adata_path)
        print('Input anndata object', adata.shape)

        if 'idx' not in adata.obs.columns:
            # make unique
            if adata.obs.index.duplicated().any():
                adata.obs_names_make_unique()
            adata.obs['idx'] = adata.obs.index

        if batch_keys is not None:
            if isinstance(batch_keys, str):
                batch_keys = [batch_keys]
            adata.obs['batch_key'] = adata.obs[batch_keys].apply(
                lambda x: '_'.join(x), axis=1
            )

        # optionally downsample
        if subsample_by is not None:
            print('Subsampling anndata object')

            if isinstance(subsample_by, str):
                subsample_by = [subsample_by]

            adata.obs['subsampling_col'] = adata.obs[subsample_by].apply(
                lambda x: '_'.join(str(x)), axis=1
            )

            adata = do_balanced_downsampling_anndata(
                adata,
                subsample_by='subsampling_col',
                n_cells_per_class=n_cells_per_class,
            )

            adata.obs.drop('subsampling_col', axis=1, inplace=True)

            # save to disk - dataset with only HVG
            os.makedirs(
                os.path.join(root_dir, 'data/processed/input_h5ad'), exist_ok=True
            )
            adata.write_h5ad(
                os.path.join(root_dir, f'data/processed/input_h5ad/{tissue}.h5ad')
            )

        if calculate_hvg and ('highly_variable' not in adata.var.columns):
            if hvg_batch_key is None:
                if batch_keys is not None:
                    hvg_batch_key = 'batch_key'
                else:
                    raise ValueError('Please provide batch key for HVG calculation')
            sc.pp.highly_variable_genes(
                adata, batch_key=hvg_batch_key, flavor='seurat_v3', n_top_genes=2000
            )

            adata = adata[:, adata.var.highly_variable]
            os.makedirs(
                os.path.join(root_dir, 'data/processed/input_h5ad'), exist_ok=True
            )
            adata.write_h5ad(
                os.path.join(root_dir, f'data/processed/input_h5ad/{tissue}_hvg.h5ad')
            )

        # Save chunks
        # Split the cells into groups of chunk_size
        obs_groups = [
            adata.obs_names[i : i + chunk_size]
            for i in range(0, len(adata.obs_names), chunk_size)
        ]

        # for dealing with missing values in pyarrow
        for column in adata.obs.columns:
            if column != 'n_counts':
                adata.obs[column] = np.where(
                    adata.obs[column].isnull(), ' ', adata.obs[column]
                )
                # print(column, adata.obs[column].dtype)
                # print('Number of missing values:', adata.obs[column].isnull().sum())

        # Drop genes which are missing ensembl_id
        if 'ensembl_id' not in adata.var.columns:
            warnings.warn('Converting ensembl_id to index')

            if input_size == 2048:
                ensembl_to_name = pd.read_pickle(
                    os.path.join(
                        geneformer_repo_path,
                        'geneformer/gene_dictionaries_30m/gene_id_name_dict_gc30M.pkl',
                    )
                )
            else:
                ensembl_to_name = pd.read_pickle(ENSEMBL_DICTIONARY_FILE)

            adata.var['ensembl_id'] = adata.var.index.map(ensembl_to_name)

        adata = adata[:, adata.var['ensembl_id'].notnull()]

        # Iterate over each group and subset the AnnData object
        n_splits = 0
        for i, obs_names in enumerate(obs_groups):
            subset_adata = adata[obs_names, :].copy()

            # Create a directory for the subset if it doesn't exist
            output_directory = os.path.join(root_dir, 'data/processed/input_h5ad')
            subset_directory = os.path.join(output_directory, f'subset_{i+1}')
            os.makedirs(subset_directory, exist_ok=True)

            # remove nan ensembl_ids
            subset_adata = subset_adata[
                :, subset_adata.var['ensembl_id'].notnull()
            ].copy()

            # Write the subset to disk
            filename = os.path.join(subset_directory, 'adata.h5ad')
            subset_adata.write_h5ad(filename)

            n_splits += 1

    subset_dirs = glob.glob(f'{root_dir}/data/processed/input_h5ad/subset_*')
    n_splits = (
        max([int(dir.split('_')[-1]) for dir in subset_dirs]) if subset_dirs else 0
    )

    # check if tokenized data exists
    if do_tokenization:
        vars_to_keep = {v: v for v in vars_to_keep}

        vars_to_keep['idx'] = 'idx'

        if batch_keys is not None:
            vars_to_keep['batch_key'] = 'batch_key'

        print('Tokenizing data')
        if use_gp_tokenizer:
            tk = GPTokenizer(
                custom_attr_name_dict=vars_to_keep,
                nproc=4,
                model_input_size=input_size,
                special_token=(input_size == 4096),
                gp_genes=gp_genes_union,
                do_ensembl_conversion=do_ensembl_conversion,
            )
        else:
            tk = TranscriptomeTokenizer(
                custom_attr_name_dict=vars_to_keep,
                model_input_size=input_size,
                special_token=(input_size == 4096),
                nproc=4,
            )

        if n_splits == 0:
            tk.tokenize_data(
                f'{root_dir}/data/processed/input_h5ad',  # h5ad data directory
                f'{root_dir}/data/processed/tokenized',
                tissue,
                file_format='h5ad',
            )

        else:
            for i in range(1, n_splits + 1):
                print(f'Tokenizing subset {i}')
                tk.tokenize_data(
                    f'{root_dir}/data/processed/input_h5ad/subset_{i}',
                    f'{root_dir}/data/processed/tokenized/',
                    f'{tissue}_{i}',
                    file_format='h5ad',
                )

        # Step 2 : Prepare for GPformer
        # change directory for outputs
        if output_data_name is None:
            folder_path = f'{root_dir}/data/processed/input_dataset'
        else:
            folder_path = f'{root_dir}/data/processed/{output_data_name}'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # check if folder is empty:
        if len(os.listdir(folder_path)) == 0:
            # load datasets
            if n_splits == 0:
                input_data = load_from_disk(
                    f'{root_dir}/data/processed/tokenized/{tissue}.dataset'
                )
            else:
                input_data = concatenate_datasets(
                    [
                        load_from_disk(
                            f'{root_dir}/data/processed/tokenized/{tissue}_{i}.dataset'
                        )
                        for i in range(1, n_splits + 1)
                    ]
                )

            # change labels to numerical ids
            if isinstance(cov_to_encode, str):
                cov_to_encode = [cov_to_encode]

            if batch_keys is not None:
                cov_to_encode.append('batch_key')

            for col in cov_to_encode:
                if col in input_data.column_names:
                    input_data = encode_labels(input_data, col, f'{col}_id')

            input_data.save_to_disk(folder_path)

            print('Saved', len(input_data), 'cells')

            # Remove intermediate directories
            os.system(f'rm -rf {root_dir}/data/processed/tokenized')
            # optionally remove all subset folders
            if n_splits > 0:
                os.system(f'rm -rf {root_dir}/data/processed/input_h5ad/subset_*')

        else:
            print('Skipping preprocessing step')

    # Step 3 : Check for GP databases
    if not os.path.exists(f'{root_dir}/gpdb_{name_tag}.csv'):
        raise ValueError(
            'GP database not found'
            'Please provide path to existing csv file'
            'where each column corresponds to a gene program'
        )

    # Step 4 : Save GP genes object
    if save_gp_genes_object:
        # Load GP genes
        if gp_genes_union is None:
            gpdb = pd.read_csv(f'{root_dir}/gpdb_{name_tag}.csv')
            gp_genes = set()
            for c in gpdb.columns:
                gp_genes.update(gpdb[c].dropna().tolist())
        else:
            gp_genes = set(gp_genes_union)

        # Load anndata object
        # look for existing object in input_h5ad directory
        if os.path.exists(
            os.path.join(root_dir, f'data/processed/input_h5ad/{tissue}.h5ad')
        ):
            adata = sc.read_h5ad(
                os.path.join(root_dir, f'data/processed/input_h5ad/{tissue}.h5ad')
            )
        else:
            adata = sc.read_h5ad(adata_path)

            if 'idx' not in adata.obs.columns:
                # make unique
                if adata.obs.index.duplicated().any():
                    adata.obs_names_make_unique()
                adata.obs['idx'] = adata.obs.index

            if batch_keys is not None:
                if isinstance(batch_keys, str):
                    batch_keys = [batch_keys]
                adata.obs['batch_key'] = adata.obs[batch_keys].apply(
                    lambda x: '_'.join(x), axis=1
                )

        # Select gp genes
        data_gp_genes_union = set(adata.var_names) & gp_genes

        if not data_gp_genes_union:
            raise ValueError(
                'No GP genes found in the dataset'
                '\nDo GP genes format match adata indices?'
            )
        adata = adata[:, list(data_gp_genes_union)]

        # Save to disk
        adata.write_h5ad(
            os.path.join(root_dir, f'data/processed/input_h5ad/{tissue}_gp_genes.h5ad')
        )

    # Clean up empty (temporary) tokenized directories
    tokenized_dir = os.path.join(root_dir, 'data/processed/tokenized')
    if os.path.exists(tokenized_dir):
        if not os.listdir(tokenized_dir):
            os.rmdir(tokenized_dir)
            print(f'Removed empty tokenized directory: {tokenized_dir}')
