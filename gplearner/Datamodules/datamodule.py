import pickle
import random
from pathlib import Path
from scanpy import read as sc_read
import numpy as np
import os
import json
from pandas import read_csv

import torch
from datasets import load_from_disk
from geneformer.perturber_utils import pad_tensor_list
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningDataModule
from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split,
    SequentialSampler
)

from scgpt.data_collator import DataCollator
from scgpt.tokenizer import GeneVocab

from scanpy import read as sc_read
import os
import json
random.seed(0)


class txDataset(Dataset):
    def __init__(
        self,
        folder='./data/tokenized.dataset',
    ):
        """Create a dataset from a directory with a tokenized Geneformer dataset

        Args:
            folder (str): Folder containing tokenized scRNA-seq dataset

        """
        self.gdata = load_from_disk(folder)

        # self.num_classes = len(set(self.gdata['label']))
        # self.num_envs = len(set(self.gdata['env']))

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


class txDataModule(LightningDataModule):
    def __init__(
        self,
        folder='./data/tokenized.dataset',
        batch_size=3,
        num_workers=1,
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
        self.dataset = txDataset(self.folder)
        self.metadata = self.dataset.metadata

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
        # if memory issues -> don't hardcode 2048 and pad to max length of batch
        model_input_size = 2048
        input_batch_id = [torch.tensor(d['input_ids']) for d in batch]
        length = torch.stack([torch.tensor(d['length']) for d in batch])

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
                output_dict[m] = torch.tensor([d[m] for d in batch], dtype=torch.long)
            else:
                output_dict[m] = [d[m] for d in batch]

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
        adata_path='/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/gplearner_reproducibility/24-04-03_synthetic_clean/data/input_h5ad/24-04-03_synthetic_clean_hvg.h5ad',
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