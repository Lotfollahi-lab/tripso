import os
import random
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
from datasets import concatenate_datasets, load_from_disk
from geneformer import TranscriptomeTokenizer

from ..Utils.utils import do_balanced_downsampling, encode_labels
from .gp_curation import make_gpdb

seed = 0
np.random.seed(seed)
random.seed(seed)


def pp_and_tokenize(
    root_dir: str,
    vars_to_keep: Union[Dict, List] = ['cell_type'],
    subsample_by: Optional[List] = ['cell_type'],
    n_cells_per_class: int = 10_000,
    n_splits: Optional[int] = None,
    reference_gpdb: Union[List[str], str] = '/path/to/reference/databases',
    use_ontology: Optional[bool] = False,
    n_cells_to_count: Optional[int] = 100,
    threshold_value: Optional[int] = 6,
    overlap_threshold: Optional[float] = 0.5,
    max_gp_len: Optional[int] = 100,
    name_tag: Optional[str] = 'Reactome',
):
    """
    Preprocess and tokenize data for scGPL

    Parameters:
    -----------
    root_dir : str
        Root directory where h5ad / tokenized data is stored
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
    n_splits : int
        If the data is split into multiple h5ad files, how many splits are there?
        This is necessary to avoid memory issues with datasets >50k cells (approx)


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

    tissue = root_dir.split('/')[-1]

    # check if tokenized data exists
    if not os.path.exists(os.path.join(root_dir, 'data/tokenized')):
        vars_to_keep = {v: v for v in vars_to_keep}

        tk = TranscriptomeTokenizer(vars_to_keep, nproc=4)

        if n_splits is None:
            tk.tokenize_data(
                f'{root_dir}/data/input_h5ad',  # h5ad data directory
                f'{root_dir}/data/tokenized',
                tissue,
                file_format='h5ad',
            )

        else:
            for i in range(1, n_splits + 1):
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
        if n_splits is None:
            input_data = load_from_disk(f'{root_dir}/data/tokenized/{tissue}.dataset')
        else:
            input_data = concatenate_datasets(
                [
                    load_from_disk(f'{root_dir}/data/tokenized/{tissue}_{i}.dataset')
                    for i in range(1, n_splits)
                ]
            )

        # change labels to numerical ids
        if 'cell_type' in input_data.column_names:
            input_data = encode_labels(input_data, 'cell_type', 'label')

        if 'condition' in input_data.column_names:
            input_data = encode_labels(input_data, 'condition', 'env')

        # Subsampling
        if subsample_by is not None:
            if isinstance(subsample_by, str):
                input_data = input_data.map(
                    lambda example: {'downsample_col': example[subsample_by]}
                )

            elif isinstance(subsample_by, list):

                def make_combined_col(example, cols=subsample_by):
                    example['downsample_col'] = '_'.join(
                        [str(example[col]) for col in cols]
                    )
                    return example

                input_data = input_data.map(make_combined_col, num_proc=16)

            input_data = do_balanced_downsampling(
                input_data['downsample_col'], input_data, n_cells_per_class
            )

            # Now drop the downsample_col column
            input_data = input_data.remove_columns(['downsample_col'])
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
        )
