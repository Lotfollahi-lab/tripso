import os
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.ops.adam import DeepSpeedCPUAdam
from scipy.sparse import vstack
from torch import optim

from gplearner.Utils.utils import (
    CosineLRwithWarmUp,
    ensembl_to_name,
    token_to_gene,
)


class scGPL(pl.LightningModule):
    """
    Description:
    ------------
    Trainer for gpTransformer model with tokenized scRNA-seq dataset as input.
    This module encompasses the following steps:
    1. Initialise MLP model
    2. Define training, validation and test step
    3. Define optimizer
    4. Define loss function
    5. Define metrics

    Parameters:
    -----------
    model: gpTransformer
        instantiated class of gpTransformer model
        (either Base, Supervised or Unsupervised)

    model_type:
        Base, Supervised or Unsupervised (affects loss function)

    Returns:
    --------

    """

    def __init__(
        self,
        model: nn.Module = None,
        model_type: str = 'Base',
        output_dir: str = '/path/to/output',
        use_gp_similarity_loss: bool = False,
        lambda_gp_similarity=1e-2,
        lr: float = 1e-3,
        weight_decay: float = 0,
        optimizer: Union[
            optim.Adam, optim.SGD, optim.AdamW, DeepSpeedCPUAdam
        ] = optim.AdamW,
        lr_scheduler='ReduceLROnPlateau',
        total_epochs: int = 100,
        return_gene_embeddings: bool = False,
        tokens_to_keep: Optional[List] = None,
        gene_file_tag: Optional[str] = None,
        return_attention: bool = False,
        gp: Optional[str] = None,
    ) -> None:
        super().__init__()
        # save hyperparameters
        self.save_hyperparameters()

        # setup model
        self.model = model
        self.model_type = model_type
        self.use_gp_similarity_loss = use_gp_similarity_loss
        self.lambda_gp_similarity = lambda_gp_similarity

        if model_type == 'Supervised':
            raise NotImplementedError(
                'Trainer for supervised model not yet implemented'
            )

        # configuring optimizers
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.total_epochs = total_epochs
        self.weight_decay = weight_decay
        self.optimizer_class = optimizer

        # Initialise list to append loss and accuracy
        for stage in ['train', 'val', 'test']:
            setattr(self, f'{stage}_loss_per_gp', {})
            setattr(self, f'{stage}_mgm_gene_pred', {})
            setattr(self, f'{stage}_mgm_gene_true', {})

            setattr(self, f'{stage}_loss_cell', [])
            setattr(self, f'{stage}_mgm_gp_true', [])
            setattr(self, f'{stage}_mgm_gp_pred', [])

            setattr(self, f'{stage}_gp_similarity_loss', [])
            setattr(self, f'{stage}_gp_attn_matrix', [])

            setattr(self, f'{stage}_loss', [])

        # For output - cells
        self.gp_cls: List[float] = []
        self.cell_metadata: Dict[str, Union[str, float]] = {}
        # For output - genes
        self.x_scgpl: List[float] = []
        self.tokens_scgpl: List[float] = []
        self.gp_labels: List[str] = []
        # For output - attention
        self.attn_scores: List[float] = []

        self.output_dir = output_dir

        # for test step
        self.return_gene_embeddings = return_gene_embeddings
        self.tokens_to_keep = tokens_to_keep
        self.gene_file_tag = gene_file_tag
        self.return_attention = return_attention
        self.gp = gp

    def forward(self, x):
        out = self.model(
            x,
            return_gene_embeddings=self.return_gene_embeddings,
            tokens_to_keep=self.tokens_to_keep,
        )
        return out

    def training_step(self, batch, batch_idx):
        loss_output = self.compute_loss(batch)
        loss_per_gp = loss_output['loss_per_gp']
        loss = loss_output['total_loss']

        if len(self.train_loss_per_gp) == 0:
            for gp in self.model.gp_inputs:
                self.train_loss_per_gp[gp] = loss_per_gp[gp].unsqueeze(0)
                self.log(
                    f'train/{gp}_MGM_loss',
                    loss_per_gp[gp],
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    sync_dist=True,
                )

        else:
            for gp in self.model.gp_inputs:
                self.train_loss_per_gp[gp] = torch.cat(
                    [self.train_loss_per_gp[gp], loss_per_gp[gp].unsqueeze(0)], dim=0
                )
                self.log(
                    f'train/{gp}_MGM_loss',
                    loss_per_gp[gp],
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    sync_dist=True,
                )

        self.train_loss.append(loss)
        self.log(
            'train/loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.model_type == 'Unsupervised':
            loss_cell = loss_output['loss_cell']
            self.train_loss_cell.append(loss_cell)
            self.log(
                'train/loss_cell',
                loss_cell,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

            if self.use_gp_similarity_loss:
                gp_similarity_loss = loss_output['gp_similarity_loss']
                self.train_gp_similarity_loss.append(gp_similarity_loss)
                self.train_gp_attn_matrix.append(loss_output['gp_attn_matrix'])

        return loss

    def on_train_epoch_end(self):
        # reset step_output
        stage = 'train'
        setattr(self, f'{stage}_loss_per_gp', {})
        setattr(self, f'{stage}_mgm_gene_pred', {})
        setattr(self, f'{stage}_mgm_gene_true', {})

        setattr(self, f'{stage}_loss_cell', [])
        setattr(self, f'{stage}_mgm_gp_true', [])
        setattr(self, f'{stage}_mgm_gp_pred', [])

        setattr(self, f'{stage}_gp_similarity_loss', [])
        setattr(self, f'{stage}_gp_attn_matrix', [])

        setattr(self, f'{stage}_loss', [])

    def _test_step_cell(self, batch, batch_idx):
        output = self.forward(batch)
        self.gp_cls.append(output['z'])

        for k, v in batch.items():
            if k != 'input_ids':
                if k in self.cell_metadata:
                    self.cell_metadata[k].append(v)
                else:
                    self.cell_metadata[k] = [v]

    def _test_step_genes(self, batch, batch_idx):
        output = self.forward(batch)

        self.x_scgpl += output['x_scgpl']
        self.tokens_scgpl += output['tokens_scgpl']
        self.gp_labels += output['gp_labels']

    def _test_step_attn(self, batch, batch_idx):
        output = self.model.get_last_self_attn(batch, gp=self.gp)

        # store attention scores here
        self.attn_scores += output['attn']

        # and metadata for obs
        for k, v in batch.items():
            if k != 'input_ids':
                if k in self.cell_metadata:
                    self.cell_metadata[k].append(v)
                else:
                    self.cell_metadata[k] = [v]

    def test_step(
        self,
        batch,
        batch_idx,
    ):
        if self.return_gene_embeddings:
            self._test_step_genes(batch, batch_idx)
        elif self.return_attention:
            self._test_step_attn(batch, batch_idx)
        else:
            self._test_step_cell(batch, batch_idx)

    def _end_test_epoch_cell(self):
        gp_emb = torch.concat(self.gp_cls, dim=0).cpu().numpy()

        # make 2D for annData input
        gp_emb = gp_emb.reshape(
            (-1, len(self.model.gp_inputs) * self.model.gp_latent_size)
        )

        # convert to dataframe, first sending tensors back to cpu as numpy arrays
        meta_dict = self.cell_metadata
        for k, v in meta_dict.items():
            if isinstance(v[0], torch.Tensor):
                meta_dict[k] = torch.cat(v).cpu().numpy().tolist()
            else:
                # flatten list of lists
                meta_dict[k] = [item for sublist in v for item in sublist]

        meta = pd.DataFrame(meta_dict)

        adata = sc.AnnData(X=gp_emb, obs=meta)

        # Set the var_names attribute of the AnnData object to the GP names
        # + index for each of the positions in the GP embedding vector
        gp_labels = [
            f'{string}_{i}'
            for string in self.model.gp_inputs
            for i in range(1, self.model.gp_latent_size + 1)
        ]

        adata.var_names = gp_labels
        adata.var['gp_idx'] = adata.var_names

        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.umap(adata, min_dist=0.4)

        adata.write_h5ad(os.path.join(self.output_dir, 'adata_gp_embedding.h5ad'))

        # reset
        self.gp_cls = []
        self.cell_metadata = {}

    def _end_test_epoch_genes(self):
        # Concatenate tensors
        x_scgpl = torch.cat(self.x_scgpl, dim=0).cpu().numpy()
        tokens_scgpl = torch.cat(self.tokens_scgpl, dim=0).cpu().numpy()

        # flatten list
        gp_labels = np.array(
            [item for sublist in self.gp_labels for item in sublist]
        ).flatten()
        # gp_labels = [item for sublist in self.gp_labels for item in sublist]
        # gp_labels = torch.cat(self.gp_labels, dim=0).cpu().numpy()

        # Create anndata object for clustering and visualisation
        adata = sc.AnnData(X=x_scgpl)
        adata.obs['token'] = list(tokens_scgpl)
        adata.obs['GP'] = list(gp_labels)

        # Map gene names for interpretability
        adata.obs['ensembl'] = adata.obs['token'].map(token_to_gene)
        adata.obs['gene'] = adata.obs['ensembl'].map(ensembl_to_name)

        print('Writing anndata file to disk at')
        print(f'{self.output_dir}/adata_gene_embedding_{self.gene_file_tag}.h5ad')

        adata.write_h5ad(
            os.path.join(
                self.output_dir, f'adata_gene_embedding_{self.gene_file_tag}.h5ad'
            )
        )

        # Reset
        self.x_scgpl = []
        self.tokens_scgpl = []
        self.gp_labels = []

    def _end_test_epoch_attn(self):
        attn = vstack(self.attn_scores)

        # convert to dataframe, first sending tensors back to cpu as numpy arrays
        meta_dict = self.cell_metadata
        for k, v in meta_dict.items():
            if isinstance(v[0], torch.Tensor):
                meta_dict[k] = torch.cat(v).cpu().numpy().tolist()
            else:
                # flatten list of lists
                meta_dict[k] = [item for sublist in v for item in sublist]

        meta = pd.DataFrame(meta_dict)
        adata = sc.AnnData(X=attn, obs=meta)

        # Set the var_names attribute of the AnnData object to the gp tokens
        gp_idx = self.model.gp_inputs.index(self.gp)
        tokens = pd.Series(
            getattr(self.model.multi_gp_encoder, f'gp{gp_idx}_tokens_encoded').keys()
        )
        ensembl_ids = tokens.map(token_to_gene)
        gene_names = ensembl_ids.map(ensembl_to_name)
        adata.var_names = ['cls'] + list(ensembl_ids)
        adata.var['token'] = ['cls'] + pd.Series(list(tokens), dtype=str).tolist()
        adata.var['ensembl'] = ['cls'] + list(ensembl_ids)
        adata.var['gene'] = ['cls'] + list(gene_names)

        adata.write_h5ad(
            os.path.join(self.output_dir, f'adata_{self.gp}_attn_scores.h5ad')
        )

        # reset
        self.attn_scores = []

    def on_test_epoch_end(self):
        if self.return_gene_embeddings:
            self._end_test_epoch_genes()
        elif self.return_attention:
            self._end_test_epoch_attn()
        else:
            self._end_test_epoch_cell()

    def compute_loss(self, batch):
        output = self.forward(batch)

        # calculate MLM loss for each GP
        gp_loss_dict = {}

        for i in range(len(self.model.gp_inputs)):
            # Loss

            loss_i = F.cross_entropy(
                output['logits_lm_list'][i].reshape(
                    -1, output['logits_lm_list'][i].shape[-1]
                ),
                output['gene_labels_list'][i].reshape(-1),
            )

            if torch.isnan(loss_i):
                # usually happens if all labels are masked
                print(f'Loss is NaN in {self.model.gp_inputs[i]}')
                print('Predictions:')
                print(output['logits_lm_list'][i])
                print('')
                print('True labels:')
                print(output['gene_labels_list'][i])
                print('')
                print('Number of NaNs in predictions:')
                print(torch.isnan(output['logits_lm_list'][i]).sum())
                print('')
                print('Number of NaNs in true labels:')
                print(torch.isnan(output['gene_labels_list'][i]).sum())
                gp_loss_dict[self.model.gp_inputs[i]] = torch.tensor(0).to(
                    loss_i.device
                )

            else:
                gp_loss_dict[self.model.gp_inputs[i]] = loss_i

        # compute total loss
        tensor_list = list(gp_loss_dict.values())

        loss = torch.sum(torch.stack(tensor_list))

        # package outputs to return flexible number of objects
        holder = {
            'loss_per_gp': gp_loss_dict,
            'total_loss': loss,
        }

        return holder

    def configure_optimizers(self):
        # Define optimizer and may be consider weight decay
        # to improve generalization L2 regularization

        # add custom learning rate for cell_token_learner if exists:
        params = list(self.model.named_parameters())

        def add_custom_lr(n):
            return 'cell_token_learner' in n

        grouped_parameters = [
            {'params': [p for n, p in params if not add_custom_lr(n)], 'lr': self.lr},
            {
                'params': [p for n, p in params if add_custom_lr(n)],
                'lr': self.lr * 1e-2,
            },
        ]

        optimizer = self.optimizer_class(
            grouped_parameters, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.lr_scheduler == 'ReduceLROnPlateau':
            print('Using ReduceLROnPlateau scheduler')
            LRscheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=2,  # default 10
                factor=0.1,  # default
                verbose=False,
                min_lr=1e-6,  # from dino,
                threshold=0.01,  # default 1e-4
            )

        elif self.lr_scheduler == 'CosineLRwithWarmUp':
            print('Using CosineAnnealingLR scheduler')
            LRscheduler = CosineLRwithWarmUp(
                optimizer,
                warmup_epochs=5,  # 10 warmup epochs in dino
                total_epochs=self.total_epochs,  # 100 epochs in dino
                eta_min=1e-6,  # from dino
            )
        else:
            raise NotImplementedError(
                'lr_scheduler must be either ReduceLROnPlateau or CosineLRwithWarmUp'
            )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': LRscheduler,
                # "monitor": "val/loss",
                'monitor': 'train/loss',
                'frequency': 1,
                'interval': 'epoch',
                'strict': True,
                'name': None,
            },
        }


if __name__ == '__main__':
    from gplearner.Datamodules.datamodule import txDataModule
    from gplearner.Models.gp_model import gpTransformerBase

    os.chdir(
        '/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium'
        '/scgpl_reproducibility/other/debugging'
    )
    dataset_path = '/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/'
    'scgpl_reproducibility/examples/dummy/data/input_dataset'
    gpdb_path = '/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium'
    '/scgpl_reproducibility/examples/dummy/gpdb.csv'

    gpdb = pd.read_csv(gpdb_path)
    txdata = txDataModule(folder=dataset_path, batch_size=128)

    model = gpTransformerBase(
        database=gpdb,
        gp_latent_size=256,
    )

    gp_transformer = scGPL(
        model,
        model_type='Base',
        lr=1e-3,
        total_epochs=1,
        output_dir='TEST',
    )

    trainer = pl.Trainer(max_epochs=1, devices=-1, accelerator='auto', precision=16)

    print('Running test dataloader')
    trainer.test(gp_transformer, txdata)
