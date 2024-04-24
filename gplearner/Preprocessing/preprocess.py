import glob
import os
import random
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import scanpy as sc
from datasets import concatenate_datasets, load_from_disk
from geneformer import TranscriptomeTokenizer

from ..Utils.utils import do_balanced_downsampling_anndata, encode_labels
from .gp_curation import make_gpdb

seed = 0
np.random.seed(seed)
random.seed(seed)


def pp_and_tokenize(
    root_dir: str,
    adata_path: Optional[str] = None,
    vars_to_keep: Union[Dict, List] = ['cell_type'],
    subsample_by: Optional[List] = ['cell_type'],
    n_cells_per_class: int = 20_000,
    chunk_size: int = 50_000,
    reference_gpdb: Union[List[str], str] = '/path/to/reference/databases',
    use_ontology: Optional[bool] = False,
    n_cells_to_count: Optional[int] = 100,
    threshold_value: Optional[int] = 6,
    overlap_threshold: Optional[float] = 0.5,
    max_gp_len: Optional[int] = 100,
    name_tag: Optional[str] = 'Reactome',
    cov_to_encode: Union[List[str], str] = ['cell_type', 'condition'],
    batch_keys: Optional[List[str]] = None,
    tissue: Optional[str] = None,
    save_intermediate: Optional[bool] = False,
    hvg_batch_key: Optional[str] = None,
):
    """
    Preprocess and tokenize data for scGPL

    Parameters:
    -----------
    root_dir : str
        Root directory where h5ad / tokenized data is stored
    adata_path : str
        Path to anndata object to tokenize
    vars_to_keep : list
        obs column names to keep from anndata object
        these will be kept as columns in the input_dataset
    subsample_by : list
        Whether to subsample the dataset to balance across specific classes
    n_cells_per_class : int
        When doing balanced subsampling,
        what is the minimum number of cells to keep in each class
        If the number of cells in a category is less than this number,
        keep all cells in that category
    chunk_size : int
        Size of chunks to split the data into

    reference_gpdb : list
        List of paths to reference databases
    use_ontology : bool
        Whether to use ontology information to curate reference databases
    n_cells_to_count : int
        How many cells to use to count genes in reference databases
    threshold_value : int
        threshold for number of genes which must be expressed in 50% of cells
    overlap_threshold : float
        Threshold for overlap between GP
    max_gp_len : int
        Maximum length of GP
    name_tag : str
        Name tag for reference databases

    """
    # Step 1 : Tokenize data

    if tissue is None:
        tissue = root_dir.split('/')[-1]

    # check for anndata object in input_h5ad directory
    if not os.path.exists(os.path.join(root_dir, 'data/input_h5ad')):
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
            os.makedirs(os.path.join(root_dir, 'data/input_h5ad'), exist_ok=True)
            adata.write_h5ad(os.path.join(root_dir, f'data/input_h5ad/{tissue}.h5ad'))

        if 'highly_variable' not in adata.var.columns:
            if hvg_batch_key is None:
                if batch_keys is not None:
                    hvg_batch_key = 'batch_key'
                else:
                    raise ValueError('Please provide batch key for HVG calculation')
            sc.pp.highly_variable_genes(
                adata, batch_key=hvg_batch_key, flavor='seurat_v3', n_top_genes=2000
            )

            adata = adata[:, adata.var.highly_variable]
            os.makedirs(os.path.join(root_dir, 'data/input_h5ad'), exist_ok=True)
            adata.write_h5ad(
                os.path.join(root_dir, f'data/input_h5ad/{tissue}_hvg.h5ad')
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

        # Iterate over each group and subset the AnnData object
        n_splits = 0
        for i, obs_names in enumerate(obs_groups):
            subset_adata = adata[obs_names, :].copy()

            # Create a directory for the subset if it doesn't exist
            output_directory = 'data/input_h5ad'
            subset_directory = os.path.join(output_directory, f'subset_{i+1}')
            os.makedirs(subset_directory, exist_ok=True)

            # Write the subset to disk
            filename = os.path.join(subset_directory, 'adata.h5ad')
            subset_adata.write(filename)

            n_splits += 1

    subset_dirs = glob.glob(f'{root_dir}/data/input_h5ad/subset_*')
    n_splits = (
        max([int(dir.split('_')[-1]) for dir in subset_dirs]) if subset_dirs else 0
    )

    # check if tokenized data exists
    if not os.path.exists(os.path.join(root_dir, 'data/tokenized')):
        vars_to_keep = {v: v for v in vars_to_keep}

        vars_to_keep['idx'] = 'idx'

        if batch_keys is not None:
            vars_to_keep['batch_key'] = 'batch_key'

        print('Tokenizing data')
        tk = TranscriptomeTokenizer(vars_to_keep, nproc=4)

        if n_splits == 0:
            tk.tokenize_data(
                f'{root_dir}/data/input_h5ad',  # h5ad data directory
                f'{root_dir}/data/tokenized',
                tissue,
                file_format='h5ad',
            )

        else:
            for i in range(1, n_splits + 1):
                print(f'Tokenizing subset {i}')
                tk.tokenize_data(
                    f'{root_dir}/data/input_h5ad/subset_{i}',  # h5ad data directory
                    f'{root_dir}/data/tokenized/',
                    f'{tissue}_{i}',
                    file_format='h5ad',
                )

    # Step 2 : Prepare for scGPL
    # change directory for outputs
    folder_path = f'{root_dir}/data/input_dataset'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # check if folder is empty:
    if len(os.listdir(folder_path)) == 0:
        # load datasets
        if n_splits == 0:
            input_data = load_from_disk(f'{root_dir}/data/tokenized/{tissue}.dataset')
        else:
            input_data = concatenate_datasets(
                [
                    load_from_disk(f'{root_dir}/data/tokenized/{tissue}_{i}.dataset')
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

    else:
        print('Data already exists in', folder_path)
        print('Skipping preprocessing step')

    # Step 3 : Prepare GP databases
    if not os.path.exists(f'{root_dir}/gpdb_{name_tag}.csv'):
        make_gpdb(
            dataset_path=folder_path,
            output_path=root_dir,
            gp_inputs=reference_gpdb,
            use_ontology=use_ontology,
            n_cells_to_count=n_cells_to_count,
            threshold_value=threshold_value,
            overlap_threshold=overlap_threshold,
            max_gp_len=max_gp_len,
            name_tag=name_tag,
            save_intermediate=save_intermediate,
        )
