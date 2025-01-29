import os
import pickle
import random
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from geneformer import TOKEN_DICTIONARY_FILE
from geneformer.perturber_utils import pad_tensor_list
from pytorch_lightning import LightningDataModule
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
    random_split,
)
from transformers.trainer_pt_utils import LengthGroupedSampler

from ..Models.gp_model import gfWrapper
from ..Utils.utils import build_gp_input_matrix, get_gp_tokens
from .mapped_collection import MappedCollection

random.seed(0)


class AnnDataset(Dataset):
    def __init__(self, adata_path):
        if isinstance(adata_path, str):
            adata_path = [adata_path]

        self.dataloader = MappedCollection(
            path_list=adata_path,
            obs_keys=['idx', 'batch_key'],
            encode_labels=False,
            parallel=(torch.cuda.device_count() > 1),
        )

        self.n_condition_combined = len(
            np.unique(self.dataloader.get_merged_labels('batch_key'))
        )

    def __len__(self):
        return len(self.dataloader)

    def __getitem__(self, idx):
        data = self.dataloader[idx]
        return {
            'X': torch.tensor(data['X'], dtype=torch.float32),
            'idx': data['idx'],
            'size_factor': data['X'].sum(axis=-1),
        }

    def get_n_genes(self):
        # assuming either one anndata object
        # or all anndata have same number of genes
        return self.dataloader.original_shapes[0][1]


class tkDataset(Dataset):
    def __init__(
        self,
        folder='./data/tokenized.dataset',
        label_key=None,
        filter_key=None,
        filter_value=None,
    ):
        """Create a dataset from a directory with a tokenized Geneformer dataset

        Args:
            folder (str): Folder containing tokenized scRNA-seq dataset
            adata (optional):
                path to anndata object

        """
        if isinstance(folder, str):
            gdata = load_from_disk(folder)
        else:
            gdata = folder

        if filter_key is not None:
            if isinstance(filter_value, str):
                gdata = gdata.filter(lambda x: x[filter_key] == filter_value)
            else:
                gdata = gdata.filter(lambda x: x[filter_key] in filter_value)

        self.gdata = gdata

        # Metadata to keep track of
        # (we assume filtering of obs columns happens at
        # tokenization stage so now we want to keep everything)
        self.metadata = [
            c for c in self.gdata.column_names if c not in ['input_ids', 'length']
        ]

        if label_key is not None:
            self.labels = np.array(self.gdata[label_key])  # FIX this
        else:
            self.labels = None

    def __len__(self):
        return len(self.gdata)

    def __getitem__(self, ind):
        return self.gdata[ind]


class txDataset(Dataset):
    def __init__(self, tk_dataset, adata_dataset):
        self.tk_dataset = tk_dataset
        self.adata_dataset = adata_dataset

    def __len__(self):
        return len(self.tk_dataset)

    def __getitem__(self, idx):
        tk = self.tk_dataset[idx]

        if self.adata_dataset is not None:
            adata = self.adata_dataset[idx]
        else:
            adata = None

        return {
            'tk': tk,
            'adata': adata,
        }

    def get_label_weights(self, subsample_indices=None):
        """
        Calculate weights for each label to be used with WeightedRandomSampler.

        Args:
            subsample_indices (list or np.ndarray, optional): Indices of a subset.
            If provided, weights are calculated based on the subset.

        Returns:
            torch.Tensor: Weights for each label.
        """
        if self.tk_dataset.labels is None:
            raise ValueError('Labels are not available.')

        # If subsample_indices is provided, use it to filter labels
        if subsample_indices is not None:
            labels = self.tk_dataset.labels[subsample_indices]
        else:
            labels = self.tk_dataset.labels

        # Calculate the frequency of each label
        label_counts = Counter(labels)

        # Calculate the total number of samples
        total_count = len(labels)

        # Calculate weights inversely proportional to the frequency
        weights = {label: total_count / count for label, count in label_counts.items()}

        # Convert weights to a tensor, matching the order of labels
        weight_tensor = torch.tensor(
            [weights[label] for label in labels], dtype=torch.float
        )

        return weight_tensor


class EmbDataset(Dataset):
    def __init__(
        self,
        folder_path,
        data_type,
        label_key=None,  # for weighted sampling
        filter_key=None,
        filter_value=None,
        clf_label=None,  # classification label
        encode_covariates=False,
        frac_for_training=1,
        condition_variable=None,
    ):
        self.data_type = data_type
        if self.data_type == 'dataset':
            if isinstance(folder_path, str):
                emb = load_from_disk(folder_path)
            else:
                emb = folder_path

            if filter_key is not None:
                if isinstance(filter_value, str):
                    emb = emb.filter(lambda x: x[filter_key] == filter_value)
                elif isinstance(filter_value, list):
                    emb = emb.filter(lambda x: x[filter_key] in filter_value)
            if frac_for_training < 1:
                emb = emb.shuffle(seed=0).select(
                    range(int(len(emb) * frac_for_training))
                )

            self.emb = emb

            if clf_label is not None:
                unique_labels = emb.unique(clf_label)
                self.num_classes = len(unique_labels)
                if encode_covariates:
                    self.label_dict = {n: i for i, n in enumerate(unique_labels)}

            if condition_variable is not None:
                self.num_condition_classes = len(emb.unique(condition_variable))
            else:
                self.num_condition_classes = 0

        elif self.data_type == 'h5ad':
            emb = sc.read_h5ad(folder_path)
            if filter_key is not None:
                if isinstance(filter_value, str):
                    emb = emb[emb.obs[filter_key] == filter_value]
                elif isinstance(filter_value, list):
                    emb = emb[emb.obs[filter_key].isin(filter_value)]

            if frac_for_training < 1:
                num_cells = emb.n_obs  # Total number of cells
                num_sample = int(
                    emb.n_obs * frac_for_training
                )  # Number of cells to sample
                # Generate random indices
                random_indices = np.random.choice(num_cells, num_sample, replace=False)
                # Select the sampled cells from the anndata object
                emb = emb[random_indices, :]

            self.emb = emb
            if clf_label is not None:
                self.num_classes = emb.obs[clf_label].nunique()
                if encode_covariates:
                    self.label_dict = {
                        n: i for i, n in enumerate(emb.obs[clf_label].unique())
                    }

            if condition_variable is not None:
                self.num_condition_classes = len(emb.obs[condition_variable].unique())
            else:
                self.num_condition_classes = 0

        else:
            raise NotImplementedError('Data type not recognized')

        # for weighted sampling
        if label_key is not None:
            if self.data_type == 'dataset':
                self.labels = np.array(self.emb[label_key])
            elif self.data_type == 'h5ad':
                self.labels = np.array(self.emb.obs[label_key])
        else:
            self.labels = None

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        if self.data_type == 'dataset':
            return self.emb[idx]
        elif self.data_type == 'h5ad':
            return {
                'X': torch.tensor(self.emb.X[idx, :], dtype=torch.float32),
                'obs': self.emb.obs.iloc[idx, :],
                'var': self.emb.var,
            }

    def get_label_weights(self, subsample_indices=None):
        """
        Calculate weights for each label to be used with WeightedRandomSampler.

        Args:
            subsample_indices (list or np.ndarray, optional): Indices of a subset.
            If provided, weights are calculated based on the subset.

        Returns:
            torch.Tensor: Weights for each label.
        """
        if self.labels is None:
            raise ValueError('Labels are not available.')

        # If subsample_indices is provided, use it to filter labels
        if subsample_indices is not None:
            labels = self.labels[subsample_indices]
        else:
            labels = self.labels

        # Calculate the frequency of each label
        label_counts = Counter(labels)

        # Calculate the total number of samples
        total_count = len(labels)

        # Calculate weights inversely proportional to the frequency
        weights = {label: total_count / count for label, count in label_counts.items()}

        # Convert weights to a tensor, matching the order of labels
        weight_tensor = torch.tensor(
            [weights[label] for label in labels], dtype=torch.float
        )

        return weight_tensor


#####################
# Datamodules
#####################


class txDataModule(LightningDataModule):
    def __init__(
        self,
        folder='./data/tokenized.dataset',
        adata_path=None,  # should be h5ad object that matches tokenized dataset exactly
        batch_size=3,
        num_workers=4,
        shuffle=False,
        sampler=None,
        label_key=None,
        return_tuple=False,
        filter_key=None,
        filter_value=None,
        frac_for_generation=1,
        fm_encoder_name='gf-6L-30M-i2048',
        # development only:
        frac_for_training=1,
        data_split_to_pass_to_test_step='val',
        seed=0,
        load_exp=False,
        model_input_size=None,
    ):
        """Create a datamodule from a tokenized Geneformer dataset

        Args:
            folder (str): Folder containing geneformer tokenized dataset
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader.
                Defaults to 0.
            shuffle (bool, optional): Whether or not to have shuffling behavior
                during sampling. Defaults to False.
            frac_for_training (float, optional): The fraction of the dataset to use
                for training. Defaults to 1.
        """
        super().__init__()
        self.folder = folder
        self.adata_path = adata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.fm_encoder_name = fm_encoder_name

        self.frac_for_training = frac_for_training
        self.data_for_test_step = data_split_to_pass_to_test_step
        self.label_key = label_key
        self.return_tuple = return_tuple
        self.filter_key = filter_key
        self.filter_value = filter_value
        self.frac_for_generation = frac_for_generation
        self.seed = seed
        self.load_exp = load_exp
        self.model_input_size = model_input_size
        if model_input_size is None:
            raise ValueError('Please specify input sequence length')

        gene_token_dict = pd.read_pickle(TOKEN_DICTIONARY_FILE)

        self.pad_token_id = gene_token_dict.get('<pad>')
        warnings.warn(
            f'Setting pad token ID to {self.pad_token_id}.'
            'Please ensure this matches your tokenization.'
        )

        self.use_weighted_sampler = False
        self.use_length_sampler = False

        if sampler == 'weighted':
            self.use_weighted_sampler = True
        elif sampler == 'length':
            self.use_length_sampler = True

    def prepare_data(self):
        # Check if the folder path exists
        if isinstance(self.folder, str):
            folder_path = Path(self.folder)
            assert folder_path.exists(), 'tokenized folder does not exist'

        if self.adata_path is not None:
            adata_path = Path(self.adata_path)
            assert adata_path.exists(), 'adata path does not exist'

    def setup(self, stage=None):
        # Load the tokenized dataset
        tokenized_dataset = tkDataset(
            self.folder,
            label_key=self.label_key,
            filter_key=self.filter_key,
            filter_value=self.filter_value,
        )

        # Optionally load anndata object
        if self.adata_path is not None:
            anndata_dataset = AnnDataset(self.adata_path)

            if len(tokenized_dataset) != len(anndata_dataset):
                print('Tokenized dataset length:', len(tokenized_dataset))
                print('Anndata object length:', len(anndata_dataset))
                raise ValueError(
                    'Tokenized dataset and anndata object do not have the same length'
                )

            # Create main dataset
            self.dataset = txDataset(tokenized_dataset, anndata_dataset)

        else:
            self.dataset = txDataset(tokenized_dataset, None)

        self.metadata = tokenized_dataset.metadata

        # Calculate lengths for train, validation, and test sets
        dataset_size = len(self.dataset)

        train_size = int(
            0.8 * dataset_size * self.frac_for_training * self.frac_for_generation
        )  # 80% for training
        print(f'Training on {train_size} samples')
        self.train_size = train_size

        val_size = int(
            0.1 * dataset_size * self.frac_for_generation
        )  # 10% for validation
        self.val_size = val_size

        test_size = int(
            (dataset_size - int(0.8 * dataset_size) - val_size)
            * self.frac_for_generation
        )  # Remaining for test

        # # FOR DEBUGGING
        # train_size = 128
        # val_size = 128
        # test_size = 128

        discard = dataset_size - train_size - val_size - test_size

        # Assign Train/val split(s) for use in Dataloaders
        self.train_dataset, self.val_dataset, self.test_dataset, _ = random_split(
            self.dataset,
            [train_size, val_size, test_size, discard],
            generator=torch.Generator().manual_seed(self.seed),  # (42),
        )

        # Optionally store lengths for use with LengthGroupedSampler
        if self.use_length_sampler:
            print('\nLoading lengths for LengthGroupedSampler\n')
            self.lengths = [d['tk']['length'] for d in self.train_dataset]

    def train_dataloader(self):
        if self.use_weighted_sampler:
            sampler = WeightedRandomSampler(
                weights=self.train_dataset.dataset.get_label_weights(
                    subsample_indices=self.train_dataset.indices
                ),
                num_samples=len(self.train_dataset),
                replacement=True,
                generator=torch.Generator().manual_seed(self.seed),
            )

            dataloader = DataLoader(
                self.train_dataset,
                collate_fn=self.custom_collate,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
                pin_memory=True,
                drop_last=True,
            )

        elif self.use_length_sampler:
            sampler = LengthGroupedSampler(
                dataset=self.train_dataset,
                lengths=self.lengths,
                batch_size=self.batch_size,
                generator=torch.Generator().manual_seed(self.seed),
            )

            dataloader = DataLoader(
                self.train_dataset,
                collate_fn=self.custom_collate,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
                pin_memory=True,
                drop_last=True,
            )

        else:
            dataloader = DataLoader(
                self.train_dataset,
                collate_fn=self.custom_collate,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )

        return dataloader

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.custom_collate,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.data_for_test_step == 'train':
            return DataLoader(
                self.train_dataset,
                collate_fn=self.custom_collate,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        elif self.data_for_test_step == 'val':
            return DataLoader(
                self.val_dataset,
                collate_fn=self.custom_collate,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            return DataLoader(
                self.test_dataset,
                collate_fn=self.custom_collate,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

    def custom_collate(self, batch):
        # Step 1 : tokenized dataset
        tokenized_batch = [d['tk'] for d in batch]

        input_batch_id = [torch.tensor(d['input_ids']) for d in tokenized_batch]
        length = torch.stack([torch.tensor(d['length']) for d in tokenized_batch])

        input_batch_id = pad_tensor_list(
            input_batch_id, 'dynamic', self.pad_token_id, self.model_input_size
        )

        output_dict = {
            'input_ids': input_batch_id.clone().detach(),
            'length': length.clone().detach(),
        }

        if self.load_exp:
            norm_exp = [torch.tensor(d['norm_exp']) for d in tokenized_batch]
            output_dict['norm_exp'] = torch.stack(norm_exp)

        # Keep track of metadata
        for m in self.metadata:
            if m.endswith('_id'):
                try:
                    output_dict[m] = torch.tensor(
                        [d[m] for d in tokenized_batch], dtype=torch.long
                    )
                except ValueError:
                    raise ValueError(
                        f"Failed to convert to tensor for key '{m}'"
                        'due to non-integer type values in the batch.'
                        'GPformer expects all variables ending in _id'
                        'to be integer encodings of categorical variables.'
                    )
            elif m == 'norm_exp':
                continue
            else:
                output_dict[m] = [d[m] for d in tokenized_batch]

        # Optionally also pass the anndata object
        if self.adata_path is not None:
            adata_batch = [d['adata'] for d in batch]

            counts = torch.stack([d['X'] for d in adata_batch])
            idx = [d['idx'] for d in adata_batch]

            # check cell indices match
            assert all(
                [a == b for a, b in zip(output_dict['idx'], idx)]
            ), 'Cell indices do not match'

            output_dict['counts'] = counts
            output_dict['size_factor'] = [d['size_factor'] for d in adata_batch]

        if self.return_tuple:
            return (
                output_dict['input_ids'],
                output_dict['length'].unsqueeze(-1),
                output_dict['cell_type'],
            )

        return output_dict


class iTxDataModule(txDataModule):
    def __init__(
        self,
        gp,
        gpdb,
        do_ensembl_conversion,
        geneformer_model,
        gene_token_path,
        gene_name_path,
        peft_config_path=None,
        fm_layer_to_quant=-1,
        fm_encoder_pkg: str = 'geneformer',
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Initialize geneformer model for getting gene embeddings
        if fm_encoder_pkg == 'from_scratch':
            # extract bert wrapper from GPformer
            self.gf_wrapper = geneformer_model

        else:
            self.gf_wrapper = gfWrapper(
                geneformer_model=geneformer_model,
                fm_layer_to_quant=fm_layer_to_quant,
                peft_config_path=peft_config_path,
            )

        # Get vocab size
        with open(gene_token_path, 'rb') as f:
            token_dict = pickle.load(f)
        self.vocab_size = max(token_dict.values())

        # Set up encoded GP tokens
        gp_tokens = get_gp_tokens(
            gpdb[gp],
            do_ensembl_conversion,
            gp,
            gene_token_path,
            gene_name_path,
        )

        gp_tokens_tensor = torch.tensor(list(gp_tokens), dtype=torch.int32)
        self.gp_tokens = gp_tokens_tensor

    def custom_collate(self, batch):
        # Step 1 : tokenized dataset
        tokenized_batch = [d['tk'] for d in batch]

        input_batch_id = [torch.tensor(d['input_ids']) for d in tokenized_batch]
        length = torch.stack([torch.tensor(d['length']) for d in tokenized_batch])

        input_batch_id = pad_tensor_list(
            input_batch_id, 'dynamic', self.pad_token_id, self.model_input_size
        )

        # Get Geneformer embeddings
        input_dict = {
            'input_ids': input_batch_id,
            'length': length,
        }
        gf_emb = self.gf_wrapper(input_dict)

        # Wrangle gp genes
        (
            emb_pad,
            tokens_pad,
            num_genes_per_cell,
            attn_mask,
        ) = build_gp_input_matrix(
            gf_emb,  # geneformer embeddings
            input_batch_id,
            self.gp_tokens,
        )

        # Set up export

        output_dict = {
            'token_labels': tokens_pad,
            'num_genes_per_cell': num_genes_per_cell,
            'attn_mask': attn_mask,
        }

        # Keep track of metadata
        for m in self.metadata:
            if m.endswith('_id'):
                output_dict[m] = torch.tensor(
                    [d[m] for d in tokenized_batch], dtype=torch.long
                )
            else:
                output_dict[m] = [d[m] for d in tokenized_batch]

        return emb_pad, output_dict


class EmbDataModule(LightningDataModule):
    def __init__(
        self,
        folder_path,
        batch_size=3,
        num_workers=1,
        emb_label=None,
        meta_labels=None,
        data_type='dataset',
        continuous_cov=[],
        use_weighted_sampler=False,
        label_key=None,
        filter_key=None,
        filter_value=None,
        clf_label=None,
        encode_covariate=False,
        condition_variable=None,
        # for development
        frac_for_training=1,
        mode=None,
    ):
        super().__init__()
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.emb_to_keep = emb_label
        if isinstance(meta_labels, str):
            self.meta_labels = [meta_labels]
        else:
            self.meta_labels = meta_labels
        self.data_type = data_type
        self.continuous_cov = continuous_cov
        self.use_weighted_sampler = use_weighted_sampler
        self.label_key = label_key
        self.filter_key = filter_key
        self.filter_value = filter_value
        self.clf_label = clf_label
        self.encode_covariate = encode_covariate
        self.condition_variable = condition_variable
        self.frac_for_training = frac_for_training
        self.mode = mode

    def prepare_data(self):
        folder_path = Path(self.folder_path)
        assert folder_path.exists(), 'folder path does not exist'

    def setup(self, stage=None):
        tag = '.h5ad' if self.data_type == 'h5ad' else ''

        self.train_dataset = EmbDataset(
            os.path.join(self.folder_path, f'train_set{tag}'),
            data_type=self.data_type,
            label_key=self.label_key,
            filter_key=self.filter_key,
            filter_value=self.filter_value,
            clf_label=self.clf_label,
            encode_covariates=self.encode_covariate,
            frac_for_training=self.frac_for_training,
            condition_variable=self.condition_variable,
        )

        if self.condition_variable is not None:
            self.num_condition_classes = self.train_dataset.num_condition_classes
        else:
            self.num_condition_classes = 0

        self.val_dataset = EmbDataset(
            os.path.join(self.folder_path, f'val_set{tag}'),
            data_type=self.data_type,
            filter_key=self.filter_key,
            filter_value=self.filter_value,
            condition_variable=self.condition_variable,
        )
        self.test_dataset = EmbDataset(
            os.path.join(self.folder_path, f'test_set{tag}'),
            data_type=self.data_type,
            filter_key=self.filter_key,
            filter_value=self.filter_value,
            condition_variable=self.condition_variable,
        )

        if self.clf_label is not None:
            self.num_classes = self.train_dataset.num_classes

    def train_dataloader(self):
        if self.use_weighted_sampler:
            sampler = WeightedRandomSampler(
                weights=self.train_dataset.get_label_weights(),
                num_samples=len(self.train_dataset),
                replacement=True,
                generator=torch.Generator().manual_seed(42),
            )

            return DataLoader(
                self.train_dataset,
                collate_fn=self.custom_collate,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        else:
            return DataLoader(
                self.train_dataset,
                collate_fn=self.custom_collate,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.custom_collate,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.custom_collate,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def custom_collate(self, batch):
        if self.data_type == 'dataset':
            # Step 1 : get relevant embeddings
            if isinstance(self.emb_to_keep, str):
                emb = [torch.tensor(d[self.emb_to_keep]) for d in batch]

                if self.mode == 'ignore_0':
                    zero_arr = np.array(
                        [d[f'{self.emb_to_keep}_num_genes'] for d in batch]
                    )
                    zero_idx = zero_arr == 0  # This should be a boolean array
                    emb = [
                        torch.tensor(d[self.emb_to_keep])
                        for i, d in enumerate(batch)
                        if not zero_idx[i]
                    ]

                elif self.mode == 'set_to_0':
                    emb = []
                    for d in batch:
                        if d[f'{self.emb_to_keep}_num_genes'] == 0:
                            emb.append(
                                torch.zeros(torch.tensor(d[self.emb_to_keep]).shape)
                            )
                        else:
                            emb.append(torch.tensor(d[self.emb_to_keep]))

                elif self.mode == 'set_to_nan':
                    emb = []
                    for d in batch:
                        if d[f'{self.emb_to_keep}_num_genes'] == 0:
                            nan_tensor = torch.full(
                                torch.tensor(d[self.emb_to_keep]).shape, float('nan')
                            )
                            emb.append(nan_tensor)
                        else:
                            emb.append(torch.tensor(d[self.emb_to_keep]))

                elif self.mode == 'set_to_random':
                    emb = []
                    for d in batch:
                        if d[f'{self.emb_to_keep}_num_genes'] == 0:
                            x_tensor = torch.randn(
                                torch.tensor(d[self.emb_to_keep]).shape
                            )
                            emb.append(x_tensor)
                        else:
                            emb.append(torch.tensor(d[self.emb_to_keep]))

                # Optionally condition on a variable
                if self.condition_variable is not None:
                    condition = [d[self.condition_variable] for d in batch]

                    # do one hot encoding
                    condition = torch.tensor(condition)
                    condition = F.one_hot(
                        condition, num_classes=self.num_condition_classes
                    )

                    emb = [torch.cat([e, c]) for e, c in zip(emb, condition)]

                output_dict = {
                    self.emb_to_keep: torch.stack(emb),
                }

            else:
                for emb_label in self.emb_to_keep:
                    emb = [torch.tensor(d[emb_label]) for d in batch]

                    output_dict = {
                        emb_label: torch.stack(emb),
                    }

            # Step 2: get metadata
            for m in self.meta_labels:
                if m == self.clf_label:
                    if self.encode_covariate:
                        if self.mode == 'ignore_0':
                            output_dict[f'{m}_id'] = torch.tensor(
                                [
                                    self.train_dataset.label_dict[d[m]]
                                    for i, d in enumerate(batch)
                                    if not zero_idx[i]
                                ],
                                dtype=torch.long,
                            )

                            output_dict[m] = [
                                d[m] for i, d in enumerate(batch) if not zero_idx[i]
                            ]

                        else:
                            output_dict[f'{m}_id'] = torch.tensor(
                                [self.train_dataset.label_dict[d[m]] for d in batch],
                                dtype=torch.long,
                            )

                            output_dict[m] = [d[m] for d in batch]

                elif m.endswith('_id'):
                    if self.mode is None:
                        output_dict[m] = torch.tensor(
                            [d[m] for d in batch], dtype=torch.long
                        )
                    elif self.mode == 'ignore_0':
                        output_dict[m] = torch.tensor(
                            [d[m] for i, d in enumerate(batch) if not zero_idx[i]],
                            dtype=torch.long,
                        )
                elif m in self.continuous_cov:
                    output_dict[m] = torch.tensor([d[m] for d in batch])
                else:
                    if self.mode == 'ignore_0':
                        output_dict[m] = [
                            d[m] for i, d in enumerate(batch) if not zero_idx[i]
                        ]
                    else:
                        output_dict[m] = [d[m] for d in batch]

        elif self.data_type == 'h5ad':
            # only keep embedding of interest
            var = batch[0]['var']

            if self.emb_to_keep == 'cell_token':
                emb = [d['X'] for d in batch]
                emb = torch.stack(emb)
            else:
                emb_idx = var.index.get_loc(self.emb_to_keep)
                emb = [torch.tensor(d['X'][emb_idx]) for d in batch]

                # prepare for passing to output dict
                emb = torch.tensor(emb)

            if len(emb.shape) == 1:
                emb = emb.unsqueeze(-1)

            output_dict = {
                self.emb_to_keep: emb,
            }

            # get metadata
            for m in self.meta_labels:
                if m == self.clf_label:
                    if self.encode_covariate:
                        output_dict[f'{m}_id'] = torch.tensor(
                            [self.train_dataset.label_dict[d['obs'][m]] for d in batch],
                            dtype=torch.long,
                        )

                        output_dict[m] = [d['obs'][m] for d in batch]

                elif m.endswith('_id'):
                    output_dict[m] = torch.tensor(
                        [d['obs'][m] for d in batch], dtype=torch.long
                    )

                elif m in self.continuous_cov:
                    output_dict[m] = torch.tensor([d['obs'][m] for d in batch])

                else:
                    output_dict[m] = [d['obs'][m] for d in batch]

        return output_dict


class iEmbDataModule(EmbDataModule):
    def __init__(self, gp_inputs, **kwargs):
        super().__init__(**kwargs)

        if isinstance(gp_inputs, str):
            gp_inputs = [gp_inputs]

        self.gp_inputs = gp_inputs

    def custom_collate(self, batch):
        # Prepare data for input into cellwrapper
        # wants x['z'] and x['num_genes_per_cell_list']
        # we need to restack in the same order:
        # 1. embeddings

        # Accumulate embeddings for each gp across the batch
        gp_embs = [
            torch.stack([torch.tensor(d[gp]) for d in batch]) for gp in self.gp_inputs
        ]

        # Stack along a new dimension to get shape (len(gp_inputs), batch, emb)
        gp_embs_tensor = torch.stack(gp_embs)

        # Transpose to get shape (batch, len(gp_inputs), emb)
        z = gp_embs_tensor.transpose(0, 1)

        # 2. num_genes_per_cell_list
        genes_per_cell_list = []

        for gp in self.gp_inputs:
            gp_i = [torch.tensor(d[f'{gp}_num_genes']) for d in batch]
            genes_per_cell_list.append(torch.stack(gp_i))

        output_dict = {'num_genes_per_cell_list': genes_per_cell_list}

        # And metadata
        for m in self.meta_labels:
            if self.encode_covariate:
                if m == self.clf_label:
                    output_dict[f'{m}_id'] = torch.tensor(
                        [self.train_dataset.label_dict[d[m]] for d in batch],
                        dtype=torch.long,
                    )

                    output_dict[m] = [d[m] for d in batch]

            if m.endswith('_id'):
                output_dict[m] = torch.tensor([d[m] for d in batch], dtype=torch.long)
            elif m in self.continuous_cov:
                output_dict[m] = torch.tensor([d[m] for d in batch])
            else:
                output_dict[m] = [d[m] for d in batch]

        return z, output_dict
