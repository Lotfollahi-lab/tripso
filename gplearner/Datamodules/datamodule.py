import os
import pickle
import random
from collections import Counter
from pathlib import Path
from scanpy import read as sc_read
import numpy as np
import os
import json
from pandas import read_csv

import numpy as np
import scanpy as sc
import torch
from datasets import load_from_disk
from geneformer.perturber_utils import pad_tensor_list
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningDataModule
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
    random_split,
    SequentialSampler
)

from scgpt.data_collator import DataCollator
from scgpt.tokenizer import GeneVocab

from scanpy import read as sc_read
import os
import json
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
        label_key=None,
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

        if label_key is not None:
            self.labels = np.array(self.gdata[label_key])
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
    def __init__(self, folder_path, data_type, label_key=None):
        self.data_type = data_type
        if self.data_type == 'dataset':
            self.emb = load_from_disk(folder_path)
        elif self.data_type == 'h5ad':
            self.emb = sc.read_h5ad(folder_path)
        else:
            raise ValueError('data_type should be either dataset or h5ad')

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
        if self.tk_dataset.labels is None:
            raise ValueError('Labels are not available.')

        # If subsample_indices is provided, use it to filter labels
        if subsample_indices is not None:
            labels = self.emb.labels[subsample_indices]
        else:
            labels = self.emb.labels

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
        num_workers=1,
        shuffle=False,
        use_weighted_sampler=False,
        label_key=None,
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
        self.label_key = label_key

        with open(token_dictionary_file, 'rb') as f:
            self.gene_token_dict = pickle.load(f)

        self.pad_token_id = self.gene_token_dict.get('<pad>')
        self.max_len = 2048

        self.use_weighted_sampler = use_weighted_sampler

    def prepare_data(self):
        # Check if the folder path exists
        folder_path = Path(self.folder)
        assert folder_path.exists(), 'tokenized folder does not exist'

        if self.adata_path is not None:
            adata_path = Path(self.adata_path)
            assert adata_path.exists(), 'adata path does not exist'

    def setup(self, stage=None):
        # Load the tokenized dataset
        tokenized_dataset = tkDataset(self.folder, label_key=self.label_key)

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
        if self.use_weighted_sampler:
            sampler = WeightedRandomSampler(
                weights=self.train_dataset.dataset.get_label_weights(
                    subsample_indices=self.train_dataset.indices
                ),
                num_samples=len(self.train_dataset),
                replacement=True,
                generator=torch.Generator().manual_seed(42),
            )

            dataloader = DataLoader(
                self.train_dataset,
                collate_fn=self.custom_collate,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        else:
            dataloader = DataLoader(
                self.train_dataset,
                collate_fn=self.custom_collate,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

        return dataloader

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
        use_weighted_sampler=False,
    ):
        super().__init__()
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.emb_to_keep = emb_label
        self.meta_labels = meta_labels
        self.data_type = data_type
        self.continuous_cov = continuous_cov
        self.use_weighted_sampler = use_weighted_sampler

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
        if self.use_weighted_sampler:
            sampler = WeightedRandomSampler(
                weights=self.train_dataset.dataset.get_label_weights(
                    subsample_indices=self.train_dataset.indices
                ),
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
                elif m in self.continuous_cov:
                    output_dict[m] = torch.tensor([d[m] for d in batch])
                else:
                    output_dict[m] = [d[m] for d in batch]

        elif self.data_type == 'h5ad':
            # only keep embedding of interest
            var = batch[0]['var']
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
                if m.endswith('_id'):
                    output_dict[m] = torch.tensor(
                        [d['obs'][m] for d in batch], dtype=torch.long
                    )
                else:
                    output_dict[m] = [d['obs'][m] for d in batch]

        return output_dict

class scgptDataset(Dataset):
    def __init__(self, count_matrix, gene_ids, vocab, model_configs, batch_ids=None):
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.batch_ids = batch_ids
        self.vocab = vocab
        self.model_configs = model_configs

    def __len__(self):
        return len(self.count_matrix)

    def __getitem__(self, idx):
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]
        # append <cls> token at the beginning
        genes = np.insert(genes, 0, self.vocab["<cls>"])
        values = np.insert(values, 0, self.model_configs["pad_value"])
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).float()
        output = {
            "id": idx,
            "genes": genes,
            "expressions": values,
        }
        if self.batch_ids is not None:
            output["batch_labels"] = self.batch_ids[idx]
        return output

class scgptDataModule(LightningDataModule):
    def __init__(
        self,
        adata_path='/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/gplearner_reproducibility/24-04-03_synthetic_clean/data/input_h5ad/24-04-03_synthetic_clean.h5ad',
        batch_size=3,
        num_workers=1,
        shuffle=False,
        input_emb_style='continuous',
        scgpt_mod='scGPT_human',
        max_length=10000,
        n_bins=51,
        # development only:
        frac_for_training=1,
    ):
        """Create a datamodule from scgpt dataset

        Args:
            adata (AnnData): Anndata object containing gene expression
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader.
                Defaults to 0.
            shuffle (bool, optional): Whether or not to have shuffling behavior
                during sampling. Defaults to False.
            frac_for_training (float, optional): The fraction of the dataset to use
                for training. Defaults to 1.
        """
        super().__init__()
        self.adata = sc_read(adata_path)
        self.max_length = max_length
        self.n_bins = n_bins
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.frac_for_training = frac_for_training

        if input_emb_style == "category":
            self.mask_value = self.n_bins + 1
            self.pad_value = self.n_bins  # for padding gene expr values
            self.n_input_bins = self.n_bins + 2
        else:
            self.mask_value = -1
            self.pad_value = -2
            self.n_input_bins = self.n_bins

        self.vocab_file = f"/lustre/scratch126/cellgen/team205/ha11/scGPT/{scgpt_mod}/vocab.json"
        self.vocab = GeneVocab.from_file(self.vocab_file)
        with open(f'/lustre/scratch126/cellgen/team205/ha11/scGPT/{scgpt_mod}/args.json', "r") as f:
            self.model_configs = json.load(f)
        assert self.model_configs["pad_value"] == self.pad_value
        if 'gene_name' not in self.adata.var.columns:
            if 'ensembl_id' not in self.adata.var.columns:
                raise Exception("Either gene_name or ensembl_id should be present in adata.var")
            else:
                self.extract_gene_names()
        self.metadata = [c for c in self.adata.obs.columns]

    def extract_gene_names(self):
        df = read_csv('/lustre/scratch126/cellgen/team205/ha11/scGPT/gene_info.csv', index_col=0)
        ids = df['feature_id'].tolist()
        names = df['feature_name'].tolist()
        name_dictionary = {k:v for k, v in zip(ids, names)}
        self.adata.var['gene_name'] = self.adata.var['ensembl_id'].map(name_dictionary)
        self.adata = self.adata[:, self.adata.var_names.isin(self.adata.var.dropna(how='any').index)]
        
        pad_token = "<pad>"
        special_tokens = [pad_token, "<cls>", "<eoc>"]
        for s in special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)
        
        self.adata.var["id_in_vocab"] = [1 if gene in self.vocab else -1 for gene in self.adata.var["gene_name"]]
        self.gene_ids_in_vocab = np.array(self.adata.var["id_in_vocab"])
        self.vocab.set_default_index(self.vocab["<pad>"])

        self.genes = self.adata.var["gene_name"].tolist()
        self.gene_ids = np.array(self.vocab(self.genes), dtype=int)
        if self.gene_ids is None:
            self.gene_ids = np.array(self.adata.var["id_in_vocab"])
            assert np.all(gene_ids >= 0)

    def setup(self, stage=None):
        count_matrix = self.adata.X
        count_matrix = (count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A)
        self.dataset = scgptDataset(count_matrix, self.gene_ids, self.vocab, self.model_configs)

        # Calculate lengths for train, validation, and test sets
        dataset_size = len(self.dataset)
        train_size = int(
            0.8 * dataset_size * self.frac_for_training
        )  # 80% for training
        print(f'Training on {train_size} samples')
        val_size = int(0.1 * dataset_size)  # 10% for validation
        test_size = (
            dataset_size - int(0.8 * dataset_size) - val_size
        )  # Remaining for test
        if test_size > 60_000:
            test_size = 50_000
        print(f'Testing on {test_size} samples')
        discard = dataset_size - train_size - val_size - test_size

        # Assign Train/val split(s) for use in Dataloaders
        self.train_dataset, self.val_dataset, self.test_dataset, _ = random_split(
            self.dataset,
            [train_size, val_size, test_size, discard],
            generator=torch.Generator().manual_seed(42),
        )
        self.collator = DataCollator(
            do_padding=True,
            pad_token_id=self.vocab[self.model_configs["pad_token"]],
            pad_value=self.pad_value,
            do_mlm=False,
            do_binning=True,
            max_length=self.max_length,
            sampling=True,
            keep_first_n_tokens=1,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            # shuffle=True,
            sampler=SequentialSampler(self.train_dataset),
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), self.batch_size),
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            # shuffle=True,
            sampler=SequentialSampler(self.val_dataset),
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), self.batch_size),
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            # shuffle=True,
            sampler=SequentialSampler(self.test_dataset),
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), self.batch_size),
            pin_memory=True
        )