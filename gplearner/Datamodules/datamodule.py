import pickle
import random
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from datasets import load_from_disk
from geneformer.in_silico_perturber import pad_tensor_list
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningDataModule
from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split,
)

random.seed(0)


class txDataset(Dataset):
    def __init__(
        self,
        folder='./data/tokenized.dataset',
        adata=None,
        transform_adata=True,
    ):
        """Create a dataset from a directory with a tokenized Geneformer dataset

        Args:
            folder (str): Folder containing tokenized scRNA-seq dataset
            adata (optional):
                path to anndata object

        """
        self.gdata = load_from_disk(folder)

        if adata:
            # calculate size factor on raw data:
            self.size_factor = np.ravel(adata.X.sum(axis=1))

            if transform_adata:
                print('Normalizing and log-transforming adata')
                print(
                    'Before transformation adata.X min-max :'
                    f'{adata.X.min()} - {adata.X.max()}'
                )
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                print(
                    'After transformation adata.X min-max :'
                    f'{adata.X.min()} - {adata.X.max()}'
                )

            self.adata = adata[self.gdata['idx']]

            # check matching between tokenized data and anndata object:
            if len(self.adata) != len(self.gdata):
                print('adata', len(self.adata))
                print('tokenized data', len(self.gdata))
                raise ValueError(
                    'Number of cells in adata and tokenized dataset do not match'
                )

            # check if the index is the same
            if not all(self.adata.obs_names == self.gdata['idx']):
                print('adata', self.adata.obs_names[:5])
                print('tk data', self.gdata['idx'][:5])
                print(len(set(self.adata.obs_names) - set(self.gdata['idx'])))
                raise ValueError('Index of adata and tokenized data do not match')
        else:
            self.adata = None

        # Metadata to keep track of
        # (we assume filtering of obs columns happens at
        # tokenization stage so now we want to keep everything)
        self.metadata = [
            c for c in self.gdata.column_names if c not in ['input_ids', 'length']
        ]

    def __len__(self):
        return len(self.gdata)

    def __getitem__(self, ind):
        return {
            'gdata': self.gdata[ind],
            'adata': self.adata[ind] if self.adata else None,
            'size_factor': self.size_factor[ind] if self.adata else None,
        }


class txDataModule(LightningDataModule):
    def __init__(
        self,
        folder='./data/tokenized.dataset',
        adata=None,
        transform_adata=True,
        batch_size=3,
        num_workers=4,
        shuffle=False,
        # development only:
        frac_for_training=1,
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
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        token_dictionary_file = TOKEN_DICTIONARY_FILE
        self.frac_for_training = frac_for_training

        with open(token_dictionary_file, 'rb') as f:
            self.gene_token_dict = pickle.load(f)

        self.pad_token_id = self.gene_token_dict.get('<pad>')
        self.max_len = 2048
        self.adata = adata
        self.transform_adata = transform_adata

    def count_unique_classes(self, supervised_labels):
        """
        Count the number of categories in class for supervised learning

        Args:
            supervised_labels (list): List of classes to count

        Returns:
            dict: Dictionary with class names as keys and the number of samples
                for each class as values
        """
        self.setup()

        if isinstance(supervised_labels, str):
            supervised_labels = [supervised_labels]
        values = {}

        for c in supervised_labels:
            column_values = [
                self.train_dataset[i][c] for i in range(len(self.train_dataset))
            ]
            values[c] = len(set(column_values))

        return values

    def prepare_data(self):
        # Check if the folder path exists
        folder_path = Path(self.folder)
        assert folder_path.exists(), 'tokenized folder does not exist'

    def setup(self, stage=None):
        self.dataset = txDataset(
            self.folder,
            adata=self.adata,
            transform_adata=self.transform_adata,
        )
        self.metadata = self.dataset.metadata

        # Calculate lengths for train, validation, and test sets
        dataset_size = len(self.dataset)
        train_size = int(
            0.8 * dataset_size * self.frac_for_training
        )  # 80% for training
        print(f'Training on {train_size} samples')
        self.train_size = train_size

        val_size = int(0.1 * dataset_size)  # 10% for validation
        self.val_size = val_size

        test_size = (
            dataset_size - int(0.8 * dataset_size) - val_size
        )  # Remaining for test

        if test_size > 50_000:
            test_size = 50_000
        self.test_size = test_size
        print(f'Testing on {test_size} samples')

        discard = dataset_size - train_size - val_size - test_size

        # Assign Train/val split(s) for use in Dataloaders
        self.train_dataset, self.val_dataset, self.test_dataset, _ = random_split(
            self.dataset,
            [train_size, val_size, test_size, discard],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
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
        model_input_size = 2048
        input_batch_id = [torch.tensor(d['gdata']['input_ids']) for d in batch]
        length = torch.stack([torch.tensor(d['gdata']['length']) for d in batch])

        input_batch_id = pad_tensor_list(
            input_batch_id, 2048, self.pad_token_id, model_input_size
        )

        output_dict = {
            'input_ids': input_batch_id.clone().detach(),
            'length': length.clone().detach(),
        }

        # Keep track of metadata
        for m in self.metadata:
            if m.endswith('_id'):
                output_dict[m] = torch.tensor(
                    [d['gdata'][m] for d in batch], dtype=torch.long
                )
            else:
                output_dict[m] = [d['gdata'][m] for d in batch]

        # FOR ANNDATA
        if self.adata is not None:
            counts = [torch.tensor(d['adata'].X.toarray()) for d in batch]
            counts = torch.cat(counts, dim=0)
            output_dict['counts'] = counts
            output_dict['size_factor'] = [d['size_factor'] for d in batch]

        return output_dict
