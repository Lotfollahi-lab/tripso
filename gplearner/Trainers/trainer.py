import os
from typing import (
    Dict,
    List,
    Union,
)

import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch import optim

from ..Utils.utils import CosineLRwithWarmUp


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

        self.gp_cls: List[float] = []
        self.cell_metadata: Dict[str, Union[str, float]] = {}

        self.output_dir = output_dir

    def forward(self, x):
        out = self.model(x)
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

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        self.gp_cls.append(output['z'])

        for k, v in batch:
            if k != 'input_ids':
                if k in self.cell_metadata:
                    self.cell_metadata[k].append(v)
                else:
                    self.cell_metadata[k] = [v]

    def on_test_epoch_end(self):
        gp_emb = torch.concat(self.gp_cls, dim=0).cpu().numpy()

        meta = pd.DataFrame(self.cell_metadata)

        adata = sc.AnnData(X=gp_emb, obs=meta)

        # Set the var_names attribute of the AnnData object to the GP names
        # + index for each of the positions in the GP embedding vector
        gp_labels = [
            f'{string}_{i}'
            for string in self.model.gp_inputs
            for i in range(1, self.model.gp_latent_size + 1)
        ]  # + [f"remaining_var_{i}" for i in range(1, unexp_rep_size + 1)]

        adata.var_names = gp_labels
        adata.var['gp_idx'] = adata.var_names

        adata.write_h5ad(os.path.join(self.output_dir, 'adata_gp_embedding.h5ad'))

        # reset
        self.gp_cls = []
        self.cell_metadata = {}

    def compute_loss(self, batch):
        output = self.forward(batch)

        # calculate MLM loss for each GP
        gp_loss_dict = {}
        mgm_gene_pred = {}
        mgm_gene_true = {}

        for i in range(len(self.model.gp_inputs)):
            # Loss

            loss_i = F.cross_entropy(
                output['logits_lm_list'][i].reshape(
                    -1, output['logits_lm_list'][i].shape[-1]
                ),
                output['gene_labels_list'][i].reshape(-1),
            )

            if torch.isnan(loss_i):
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

            gp_loss_dict[self.model.gp_inputs[i]] = loss_i

            # True/predicted tokens
            pred_i = output['logits_lm_list'][i].argmax(-1)

            # Ignore -100 masked tokens
            mask = (
                output['gene_labels_list'][i] != -100
            )  # Create a mask to ignore -100 values

            mgm_gene_pred[self.model.gp_inputs[i]] = pred_i[mask]
            mgm_gene_true[self.model.gp_inputs[i]] = output['gene_labels_list'][i][mask]

        # compute total loss
        tensor_list = list(gp_loss_dict.values())

        loss = torch.sum(torch.stack(tensor_list))

        # package outputs to return flexible number of objects
        holder = {
            'loss_per_gp': gp_loss_dict,
            'total_loss': loss,
            # "mgm_gene_pred" : mgm_gene_pred,
            # "mgm_gene_true" : mgm_gene_true
        }

        if self.model_type == 'Unsupervised':
            # calculate loss on GP tokens self-attention
            batch_size = output['z'].shape[0]
            gp_labels = (
                torch.tensor([i for i in range(len(self.model.gp_inputs))])
                .unsqueeze(0)
                .expand(batch_size, len(self.model.gp_inputs))
                .to(output['z'].device)
            )
            loss_cell = F.cross_entropy(
                output['logits_gp'], gp_labels
            )  # masked GPs are set to -100 and ignored by cross entropy loss
            holder['loss_cell'] = loss_cell

            # # accuracy for GP tokens self-attention
            # pred_gp = output["logits_gp"].argmax(-1)

            # # Ignore masked values
            # mask = (gp_labels != -100)

            # # output for accuracy calculation on epoch end
            # holder["mgm_gp_pred"] = pred_gp[mask]
            # holder["mgm_gp_true"] = gp_labels[mask]

            # finally calculate similarity betweeen final self attention block
            # and ground truth similarity matrix
            # expand true similarity matrix along batch dimension:
            if self.use_gp_similarity_loss:
                attention_matrix = output['attention_matrix'].float()
                true_similarity = (
                    torch.tensor(self.model.gp_similarity_matrix)
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                    .to(output['z'].device)
                    .float()
                )
                gp_similarity_loss = F.mse_loss(attention_matrix, true_similarity)

                holder['gp_similarity_loss'] = gp_similarity_loss
                holder['gp_attn_matrix'] = attention_matrix

            # calculate full loss term
            # individual_gp_loss = loss / len(self.model.gp_inputs)
            individual_gp_loss = loss

            if self.use_gp_similarity_loss:
                loss = (
                    individual_gp_loss
                    + loss_cell
                    + self.lambda_gp_similarity * gp_similarity_loss
                )
            else:
                loss = individual_gp_loss + loss_cell

            holder['total_loss'] = loss

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
