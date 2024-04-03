import os
import pickle
import random
from pathlib import Path

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


class AnnDataset(Dataset):
    def __init__(
        self,
        path='/path/to/adata.h5ad',
    ):
        """Create a dataset from an anndata object

        Args:
            folder (str): path to h5ad file

        """

        # Load the data
        if path.endswith('.h5ad'):
            adata = sc.read_h5ad(path)
        elif path.endswith('.loom'):
            adata = sc.read_loom(path)

        self.adata = adata

        if 'batch_key' in adata.obs.columns:
            n_condition_combined = adata.obs['batch_key'].nunique()
        else:
            raise ValueError(
                'No batch_key found'
                'for ZINB or NB reconstruction loss'
                'Please provide batch_key in adata.obs'
                'by passing batch_keys argument to preprocess function'
            )

        self.n_condition_combined = n_condition_combined

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        adata_tensor = torch.tensor(self.adata.X[idx, :], dtype=torch.float32)
        obs = self.adata.obs.iloc[idx, :]
        idx = obs['idx']

        # obs = self.adata.obs.iloc[idx, :]
        # var = self.adata.var.iloc[idx, :]

        output = {
            'X': adata_tensor,
            'idx': idx,
            #  "obs" : obs,
            #  "var" : var,
            'size_factor': adata_tensor.sum(axis=-1),
        }
        return output

    def get_n_genes(self):
        return self.adata.shape[1]


class tkDataset(Dataset):
    def __init__(
        self,
        folder='./data/tokenized.dataset',
    ):
        """Create a dataset from a directory with a tokenized Geneformer dataset

        Args:
            folder (str): Folder containing tokenized scRNA-seq dataset
            adata (optional):
                path to anndata object

        """
        self.gdata = load_from_disk(folder)

        # Metadata to keep track of
        # (we assume filtering of obs columns happens at
        # tokenization stage so now we want to keep everything)
        self.metadata = [
            c for c in self.gdata.column_names if c not in ['input_ids', 'length']
        ]

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


class EmbDataset(Dataset):
    def __init__(self, folder_path, data_type):
        self.data_type = data_type
        if self.data_type == 'dataset':
            self.emb = load_from_disk(folder_path)
        elif self.data_type == 'h5ad':
            self.emb = sc.read_h5ad(folder_path)
        else:
            raise ValueError('data_type should be either dataset or h5ad')

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


#####################
# Datamodules
#####################


class txDataModule(LightningDataModule):
    def __init__(
        self,
        folder='./data/tokenized.dataset',
        adata_path=None,  # should be h5ad object that matches tokenized dataset exactly
        batch_size=3,
        num_workers=1,
        shuffle=False,
        # development only:
        frac_for_training=1,
        data_split_to_pass_to_val_step='val',
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
        token_dictionary_file = TOKEN_DICTIONARY_FILE
        self.frac_for_training = frac_for_training
        self.data_for_validation_step = data_split_to_pass_to_val_step

        with open(token_dictionary_file, 'rb') as f:
            self.gene_token_dict = pickle.load(f)

        self.pad_token_id = self.gene_token_dict.get('<pad>')
        self.max_len = 2048

    def prepare_data(self):
        # Check if the folder path exists
        folder_path = Path(self.folder)
        assert folder_path.exists(), 'tokenized folder does not exist'

        if self.adata_path is not None:
            adata_path = Path(self.adata_path)
            assert adata_path.exists(), 'adata path does not exist'

    def setup(self, stage=None):
        # Load the tokenized dataset
        tokenized_dataset = tkDataset(self.folder)

        # Optionally load anndata object
        if self.adata_path is not None:
            anndata_dataset = AnnDataset(self.adata_path)

            if len(tokenized_dataset) != len(anndata_dataset):
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

        # # FOR DEBUGGING
        # train_size = 10
        # val_size = 10
        # test_size = 10

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
        if self.data_for_validation_step == 'train':
            return DataLoader(
                self.train_dataset,
                collate_fn=self.custom_collate,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif self.data_for_validation_step == 'test':
            return DataLoader(
                self.test_dataset,
                collate_fn=self.custom_collate,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        else:
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
        # Step 1 : tokenized dataset
        tokenized_batch = [d['tk'] for d in batch]

        model_input_size = 2048
        input_batch_id = [torch.tensor(d['input_ids']) for d in tokenized_batch]
        length = torch.stack([torch.tensor(d['length']) for d in tokenized_batch])

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
                    [d[m] for d in tokenized_batch], dtype=torch.long
                )
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

        return output_dict


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
    ):
        super().__init__()
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.emb_to_keep = emb_label
        self.meta_labels = meta_labels
        self.data_type = data_type
        self.continuous_cov = continuous_cov

    def prepare_data(self):
        folder_path = Path(self.folder_path)
        assert folder_path.exists(), 'folder path does not exist'

    def setup(self, stage=None):
        tag = '.h5ad' if self.data_type == 'h5ad' else ''

        self.train_dataset = EmbDataset(
            os.path.join(self.folder_path, f'train_set{tag}'), data_type=self.data_type
        )
        self.val_dataset = EmbDataset(
            os.path.join(self.folder_path, f'val_set{tag}'), data_type=self.data_type
        )
        self.test_dataset = EmbDataset(
            os.path.join(self.folder_path, f'test_set{tag}'), data_type=self.data_type
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
        if self.data_type == 'dataset':
            # Step 1 : get relevant embeddings
            emb = [torch.tensor(d[self.emb_to_keep]) for d in batch]

            output_dict = {
                self.emb_to_keep: torch.stack(emb),
            }

            # Step 2: get metadata
            for m in self.meta_labels:
                if m.endswith('_id'):
                    output_dict[m] = torch.tensor(
                        [d[m] for d in batch], dtype=torch.long
                    )
                elif m in self.continous_cov:
                    output_dict[m] = torch.tensor([d[m] for d in batch])
                else:
                    output_dict[m] = [d[m] for d in batch]

        elif self.data_type == 'h5ad':
            emb = [torch.tensor(d['X']) for d in batch]

            # only keep embedding of interest
            var = batch[0]['var']
            emb_idx = var.index.get_loc(self.emb_to_keep)
            emb = emb[:, emb_idx]

            output_dict = {
                self.emb_to_keep: emb,
            }

            # get metadata
            for m in self.meta_labels:
                if m.endswith('_id'):
                    output_dict[m] = torch.tensor(
                        [d['obs'][m] for d in batch], dtype=torch.long
                    )
                else:
                    output_dict[m] = [d['obs'][m] for d in batch]

        return output_dict
