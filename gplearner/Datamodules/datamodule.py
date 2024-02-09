import pickle
import random
from pathlib import Path

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
    ):
        """Create a dataset from a directory with a tokenized Geneformer dataset

        Args:
            folder (str): Folder containing tokenized scRNA-seq dataset

        """
        self.gdata = load_from_disk(folder)

        self.num_classes = len(set(self.gdata['label']))
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
            shuffle=True,
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
            if m == 'label':
                output_dict[m] = torch.tensor([d[m] for d in batch], dtype=torch.long)
            elif m == 'env':
                output_dict[m] = torch.tensor([d[m] for d in batch], dtype=torch.long)
            else:
                output_dict[m] = [d[m] for d in batch]

        return output_dict
