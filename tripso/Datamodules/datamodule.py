import random
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from pytorch_lightning import LightningDataModule
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
    random_split,
)
from transformers.trainer_pt_utils import LengthGroupedSampler

from .. import TOKEN_DICTIONARY_FILE
from ..Utils import pad_tensor_list
from .mapped_collection import MappedCollection

random.seed(0)


class AnnDataset(Dataset):
    """Pytorch dataset class for working with AnnData objects.
    Based on MappedCollection (copied from LaminDB) for efficient data loading.

    Args:
        adata_path (str or list): Path(s) to h5ad file(s) containing anndata objects.
            Can be a single path or list of paths.
    """

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
    """Dataset for tokenized data (following Geneformer tokenization).

    Args:
        folder (str or datasets.Dataset): Path to directory containing tokenized
            scRNA-seq dataset, or a pre-loaded HuggingFace Dataset object.
            Defaults to './data/tokenized.dataset'.
        label_key (str, optional): Column which contains the labels which will be used
            for weighted data sampling.
        filter_key (str, optional): Metadata key to filter dataset by.
        filter_value (str or list, optional): Value(s) to filter by for filter_key.
    """

    def __init__(
        self,
        folder='./data/tokenized.dataset',
        label_key=None,
        filter_key=None,
        filter_value=None,
    ):
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
    """Combined dataset wrapping tokenized and AnnData datasets.

    Args:
        tk_dataset (tkDataset): Tokenized Geneformer dataset.
        adata_dataset (AnnDataset or None): AnnData dataset with expression counts.
            If None, only tokenized data will be returned.
    """

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
        """Calculate weights for each label for WeightedRandomSampler.

        Parameters
        ----------
        subsample_indices : list or np.ndarray, optional
            Indices of a subset. If provided, weights are calculated based
            on the subset (default: None).

        Returns
        -------
        torch.Tensor
            Weights for each label.
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


#####################
# Datamodules
#####################


class txDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for tokenized Geneformer data.

    Args:
        folder (str): Path to directory containing tokenized Geneformer
            dataset. Defaults to './data/tokenized.dataset'.
        adata_path (str or list, optional): Path(s) to h5ad file(s) with
            expression counts that match the tokenized dataset exactly.
        batch_size (int): Batch size for dataloaders. Defaults to 3.
        num_workers (int): Number of DataLoader workers. Defaults to 4.
        shuffle (bool): Whether to shuffle training data. Defaults to False.
        sampler (str, optional): Type of sampler to use. Options: 'weighted'
            for WeightedRandomSampler, 'length' for LengthGroupedSampler,
            or None.
        label_key (str, optional): Key for labels used with weighted sampler.
        return_tuple (bool): Whether to return data as tuple instead of
            dict. Defaults to False.
        filter_key (str, optional): Metadata key to filter dataset by.
        filter_value (str or list, optional): Value(s) to filter by for
            filter_key.
        frac_for_generation (float): Fraction of data to use overall.
            Defaults to 1.
        fm_encoder_name (str): Name of foundation model encoder.
            Defaults to 'gf-6L-30M-i2048'.
        seed (int): Random seed for data splitting. Defaults to 0.
        model_input_size (int): Maximum input sequence length for padding.
            Required.
        output_dir (str): Output directory path. Defaults to './'.
        frac_for_training (float): Fraction of training split to use.
            Defaults to 1.
        data_split_to_pass_to_test_step (str): Which split to use in
            test_dataloader. Options: 'train', 'val', 'test'.
            Defaults to 'val'. This is used for applying the test step on
            a different split after model training.
    """

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
        seed=0,
        model_input_size=None,
        output_dir='./',
        # development only:
        frac_for_training=1,
        data_split_to_pass_to_test_step='val',
    ):
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
        # train_size = 100
        # val_size = 100
        # test_size = 100

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
            drop_last=True,
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
