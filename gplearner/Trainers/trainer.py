import os
import warnings
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
from datasets import Dataset, concatenate_datasets

# from deepspeed.ops.adam import DeepSpeedCPUAdam
from scipy.sparse import vstack
from sklearn.metrics import classification_report
from torch import optim
from torchmetrics import MeanSquaredError, PearsonCorrCoef
from torchmetrics.functional import pairwise_cosine_similarity

from ..Metrics.metrics import evaluate_emd, evaluate_mmd
from ..Models.gp_model import EmbEvaluatorHead
from ..Utils.losses import (
    mse_loss,
    nb,
    zinb,
)
from ..Utils.utils import (
    CosineLRwithWarmUp,
    ensembl_to_name,
    one_hot_encoder,
    token_to_gene,
    wrangle_classification_report,
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
        Mean : learn GP representations by averaging representations
        Base : use individual transformer blocks to learn GP representations
        Global : additionally learn a global cell token

    model_loss:
        loss function for learning global cell token
        only supervised implemented for now

    lambda_clf_loss:
        dictionary of form {label_name : value} weighting classification loss

    output_dir:
        path to save outputs

    use_gp_similarity_loss:
        whether to use GP similarity loss

    lambda_gp_similarity:
        weight for GP similarity loss

    gp_similarity:
        path to file containing GP similarity matrix

    lr:
        learning rate

    return_classification_report:
        whether to return classification report at test time
        set to False if not using supervised learning
        or if test data does not contain true labels

    Returns:
    --------

    """

    def __init__(
        self,
        model: nn.Module = None,
        model_type: str = 'Base',
        global_loss: str = 'supervised',
        lambda_clf_loss=1,
        output_dir: str = '/path/to/output',
        use_gp_similarity_loss: bool = False,
        lambda_gp_similarity=1e-2,
        gp_similarity: Optional[str] = None,
        lr: float = 1e-3,
        weight_decay: float = 0,
        optimizer: Union[
            optim.Adam,
            optim.SGD,
            optim.AdamW,
            # DeepSpeedCPUAdam
        ] = optim.AdamW,
        lr_scheduler='ReduceLROnPlateau',
        total_epochs: int = 100,
        return_gene_embeddings: bool = False,
        tokens_to_keep: Optional[List] = None,
        gene_file_tag: Optional[str] = None,
        return_attention: bool = False,
        gp: Optional[str] = None,
        return_classification_report: bool = False,
        total_n_genes: int = 20_000,
        n_condition_combined: int = 1,  # number of batches for zinb and nb
        test_random_baseline: bool = False,
        finetune_lr: float = 1e-5,
        use_finetune_lr: bool = False,
        save_emb: bool = False,
        split_label: str = 'train',
    ) -> None:
        super().__init__()
        # save hyperparameters
        self.save_hyperparameters()

        # setup model
        self.model = model
        self.model_type = model_type
        self.global_loss = global_loss
        self.return_classification_report = return_classification_report
        self.test_random_baseline = test_random_baseline

        if use_gp_similarity_loss and gp_similarity is None:
            raise ValueError(
                'If use_gp_similarity_loss is True, gp_similarity_file must be provided'
            )

        self.use_gp_similarity_loss = use_gp_similarity_loss
        self.gp_similarity = gp_similarity

        self.lambda_gp_similarity = lambda_gp_similarity

        if (self.global_loss == 'supervised') & (self.model_type == 'Global'):
            if isinstance(lambda_clf_loss, int) or isinstance(lambda_clf_loss, float):
                self.lambda_clf_loss = {t: 1 for t in self.model.supervised_tasks}
            elif isinstance(lambda_clf_loss, dict):
                self.lambda_clf_loss = lambda_clf_loss
            else:
                raise ValueError(
                    'Please provide dictionary with task names as keys'
                    'classification loss weights as values'
                    'e.g. {task1: 1, task2: 0.5}'
                    'or a single float value for all tasks'
                )

        if (self.model_type == 'Global') & (self.global_loss == 'reconstruction'):
            self.reconstruction_loss = self.model.reconstruction_loss
            if self.reconstruction_loss in ['nb', 'zinb']:
                self.n_conditions_combined = n_condition_combined
                self.theta = torch.nn.Parameter(
                    torch.randn(total_n_genes, self.n_conditions_combined)
                )
            else:
                self.theta = None

            self.metric = nn.ModuleDict(
                {
                    'mse': MeanSquaredError(),
                }
            )

        # configuring optimizers
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.total_epochs = total_epochs
        self.weight_decay = weight_decay
        self.optimizer_class = optimizer
        self.finetune_lr = finetune_lr
        self.use_finetune_lr = use_finetune_lr
        self.save_emb = save_emb
        self.split_label = split_label

        # Initialise list to append loss and accuracy
        for stage in ['train', 'val', 'test']:
            setattr(self, f'{stage}_loss_per_gp', {})
            setattr(self, f'{stage}_mgm_gene_pred', {})
            setattr(self, f'{stage}_mgm_gene_true', {})

            setattr(self, f'{stage}_gp_similarity_loss', [])

            # For learning global cell token
            if self.model_type == 'Global':
                if self.global_loss == 'supervised':
                    setattr(self, f'{stage}_clf_pred', {})
                    setattr(self, f'{stage}_clf_true', {})

                    for t in self.model.supervised_tasks:
                        setattr(self, f'{stage}_{t}_loss', [])
                        getattr(self, f'{stage}_clf_pred')[t] = []
                        getattr(self, f'{stage}_clf_true')[t] = []

                if self.global_loss == 'reconstruction':
                    setattr(self, f'{stage}_true_counts_list', [])
                    setattr(self, f'{stage}_pred_counts_list', [])

            setattr(self, f'{stage}_loss', [])

        # For output - cells
        self.gp_cls: List[float] = []
        self.cell_metadata: Dict[str, Union[str, float]] = {}
        self.cell_token: List[float] = []
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

        # for saving embeddings
        self.emb_dataset = None

    def forward(self, x):
        out = self.model(
            x,
            return_gene_embeddings=self.return_gene_embeddings,
            tokens_to_keep=self.tokens_to_keep,
        )

        return out

    def training_step(self, batch, batch_idx):
        loss_output = self.compute_loss(batch)

        # exit function if we've already saved embeddings
        if loss_output is None:
            return None

        loss_per_gp = loss_output['loss_per_gp']
        loss = loss_output['total_loss']

        if len(self.train_loss_per_gp) == 0:
            for i, gp in enumerate(self.model.gp_inputs):
                # only log if requires_grad = True
                if (
                    self.model.multi_gp_encoder.encoder[i]
                    .blocks[0]
                    .attn.qkv.weight.requires_grad
                ):
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
            for i, gp in enumerate(self.model.gp_inputs):
                if (
                    self.model.multi_gp_encoder.encoder[i]
                    .blocks[0]
                    .attn.qkv.weight.requires_grad
                ):
                    self.train_loss_per_gp[gp] = torch.cat(
                        [self.train_loss_per_gp[gp], loss_per_gp[gp].unsqueeze(0)],
                        dim=0,
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

        if self.model_type == 'Global':
            if self.global_loss == 'supervised':
                for t in self.model.supervised_tasks:
                    clf_loss = loss_output['loss_clf'][t]
                    setattr(self, f'train_{t}_loss', clf_loss)
                    self.log(
                        f'train/{t}_loss',
                        clf_loss,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                        sync_dist=True,
                    )
            elif self.global_loss == 'mse':
                embedding_mse_loss = loss_output['embedding_mse_loss']
                self.log(
                    'train/embedding_mse_loss',
                    embedding_mse_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )
            elif self.global_loss == 'masking':
                cell_masking_loss = loss_output['cell_masking_loss']
                self.log(
                    'train/cell_masking_loss',
                    cell_masking_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

            elif self.global_loss == 'reconstruction':
                reconstruction_loss = loss_output['reconstruction_loss']
                self.log(
                    f'train/{self.model.reconstruction_loss}_loss',
                    reconstruction_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

        if self.use_gp_similarity_loss:
            gp_similarity_loss = loss_output['gp_similarity_loss']
            self.train_gp_similarity_loss.append(gp_similarity_loss)
            self.log(
                'train/gp_similarity_loss',
                gp_similarity_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        return loss

    def on_train_epoch_end(self):
        # reset step_output
        stage = 'train'
        setattr(self, f'{stage}_loss_per_gp', {})
        setattr(self, f'{stage}_mgm_gene_pred', {})
        setattr(self, f'{stage}_mgm_gene_true', {})

        setattr(self, f'{stage}_gp_similarity_loss', [])

        if self.model_type == 'Global':
            if self.global_loss == 'supervised':
                for t in self.model.supervised_tasks:
                    setattr(self, f'{stage}_{t}_loss', [])

            if self.global_loss == 'reconstruction':
                # return Pearson correlation coefficient
                true_counts = torch.cat(self.train_true_counts_list).float()
                pred_counts = torch.cat(self.train_pred_counts_list)

                # Pearson correlation coefficient
                self.metric['pearson_train'] = PearsonCorrCoef(
                    num_outputs=true_counts.shape[0]
                ).to(true_counts.device)

                pearson = self.metric['pearson_train'](pred_counts.T, true_counts.T)
                mean_pearson = torch.mean(pearson)
                self.log(
                    'train/pearson',
                    mean_pearson,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                mse = self.metric['mse'](pred_counts, true_counts)
                mean_mse = torch.mean(mse)
                self.log(
                    'train/mse',
                    mean_mse,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        # empty lists
        self.train_true_counts_list = []
        self.train_pred_counts_list = []

        setattr(self, f'{stage}_loss', [])

    def validation_step(self, batch, batch_idx):
        # Optionally save embeddings
        if self.save_emb:
            output = self.forward(batch)

            emb_dict = {}

            for i, gp in enumerate(self.model.gp_inputs):
                emb_dict[gp] = output['z'][:, i, :].detach().cpu()
                emb_dict[f'{gp}_num_genes'] = (
                    output['num_genes_per_cell_list'][i].cpu().numpy().T
                )

            if self.model_type == 'Global':
                emb_dict['cell_token'] = output['cell_token'].detach().cpu()

            # metadata
            for k, v in batch.items():
                if k != 'input_ids':
                    emb_dict[k] = v

            emb = Dataset.from_dict(emb_dict)

            if self.emb_dataset is None:
                self.emb_dataset = emb
            else:
                self.emb_dataset = concatenate_datasets([self.emb_dataset, emb])

            return None

        if self.model_type == 'Global':
            if self.global_loss == 'supervised':
                output = self.forward(batch)

                # track true labels and predictions
                for t in self.model.supervised_tasks:
                    self.val_clf_pred[t] += output[f'logits_{t}']
                    self.val_clf_true[t] += batch[t]

            elif self.global_loss == 'reconstruction':
                output = self.forward(batch)

                if self.model.reconstruction_loss in ['mse']:
                    self.val_pred_counts_list.append(
                        output['count_output']['count_lognorm']
                    )
                    self.val_true_counts_list.append(batch['counts'])

                if self.model.reconstruction_loss in ['nb', 'zinb']:
                    self.val_pred_counts_list.append(
                        output['count_output']['count_mean']
                    )
                    self.val_true_counts_list.append(batch['counts'])

                if self.model.reconstruction_loss == 'binning':
                    self.val_pred_counts_list.append(output['count_output'])
                    self.val_true_counts_list.append(output['true_bins'])

    def on_validation_epoch_end(self):
        if self.save_emb:
            output_path = os.path.join(self.output_dir, 'embeddings')
            os.makedirs(output_path, exist_ok=True)
            output_name = os.path.join(output_path, f'{self.split_label}_set')
            self.emb_dataset.save_to_disk(output_name)
            self.emb_dataset = None
            return None

        if self.model_type == 'Global':
            if self.global_loss == 'supervised':
                for t in self.model.supervised_tasks:
                    # calculate accuracy
                    pred = torch.stack(self.val_clf_pred[t], dim=-1).T
                    true = torch.cat(
                        [torch.unsqueeze(tensor, 0) for tensor in self.val_clf_true[t]]
                    )
                    acc = (pred.argmax(dim=1) == true).float().mean()
                    self.log(
                        f'val/{t}_accuracy',
                        acc,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                        sync_dist=True,
                    )

                self.val_clf_pred = {t: [] for t in self.model.supervised_tasks}
                self.val_clf_true = {t: [] for t in self.model.supervised_tasks}

            elif self.global_loss == 'reconstruction':
                # return Pearson correlation coefficient
                true_counts = torch.cat(self.val_true_counts_list).float()
                pred_counts = torch.cat(self.val_pred_counts_list)

                self.metric['pearson_val'] = PearsonCorrCoef(
                    num_outputs=true_counts.shape[0]
                ).to(true_counts.device)

                pearson = self.metric['pearson_val'](pred_counts.T, true_counts.T)

                mean_pearson = torch.mean(pearson)
                self.log(
                    'val/pearson',
                    mean_pearson,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

                mse = self.metric['mse'](pred_counts, true_counts)
                mean_mse = torch.mean(mse)
                self.log('val/mse', mean_mse, on_epoch=True, prog_bar=True, logger=True)

                # reset lists
                self.val_true_counts_list = []
                self.val_pred_counts_list = []

    def _test_step_cell(self, batch, batch_idx):
        output = self.forward(batch)

        self.gp_cls.append(output['z'])

        for k, v in batch.items():
            if k != 'input_ids':
                if k in self.cell_metadata:
                    self.cell_metadata[k].append(v)
                else:
                    self.cell_metadata[k] = [v]

        # option to store cell type predictions
        if self.model_type == 'Global':
            self.cell_token.append(output['cell_token'])

            if self.global_loss == 'supervised':
                for t in self.model.supervised_tasks:
                    self.test_clf_pred[t].append(output[f'logits_{t}'])

            if self.test_random_baseline:
                # store predicted counts and true counts
                self.test_true_counts_list.append(batch['counts'])
                if self.model.reconstruction_loss in ['mse']:
                    self.test_pred_counts_list.append(
                        output['count_output']['count_lognorm']
                    )
                if self.model.reconstruction_loss in ['nb', 'zinb']:
                    self.test_pred_counts_list.append(
                        output['count_output']['count_mean']
                    )

    def _test_step_genes(self, batch, batch_idx):
        output = self.forward(batch)

        self.x_scgpl += output['x_scgpl']
        self.tokens_scgpl += output['tokens_scgpl']
        self.gp_labels += output['gp_labels']

    def _test_step_attn(self, batch, batch_idx):
        if self.gp == 'cell_token':
            output = self.model.get_cell_token_attention(batch)
        else:
            output = self.model.get_last_self_attn(batch, gp=self.gp)

        # store attention scores here
        self.attn_scores.append(output['attn'])

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
        # optionally check count reconstruction metrics
        if self.test_random_baseline:
            # return Pearson correlation coefficient
            true_counts = torch.cat(self.test_true_counts_list).float()
            pred_counts = torch.cat(self.test_pred_counts_list)

            print('True counts max value:', true_counts.max())
            print('Predicted counts max value:', pred_counts.max())

            # shuffle the counts
            true_counts_shuffled = true_counts[torch.randperm(true_counts.size(0))]

            # Pearson
            self.metric['pearson_val'] = PearsonCorrCoef(
                num_outputs=true_counts.shape[0]
            ).to(true_counts.device)

            pearson = self.metric['pearson_val'](pred_counts.T, true_counts.T)
            mean_pearson = torch.mean(pearson)

            pearson_shuffled = self.metric['pearson_val'](
                pred_counts.T, true_counts_shuffled.T
            )
            mean_pearson_shuffled = torch.mean(pearson_shuffled)

            # Pearson correlation for non zero genes
            n_cells, n_genes = pred_counts.shape
            mean_pearson_non_zero = []

            for cell_idx in range(n_cells):
                # For each cell, identify non-zero genes
                non_zero_genes = true_counts[cell_idx, :] > 0

                # Filter out zero-expression genes for this cell
                # in both pred and true counts
                pred_non_zero = pred_counts[cell_idx, non_zero_genes]
                true_non_zero = true_counts[cell_idx, non_zero_genes]

                if (
                    len(pred_non_zero) > 1
                ):  # Ensure there's more than one gene to calculate Pearson correlation
                    # Calculate Pearson correlation for the non-zero genes in this cell
                    pearson_corr = torch.corrcoef(
                        torch.stack((pred_non_zero, true_non_zero))
                    )[0, 1]
                    mean_pearson_non_zero.append(pearson_corr)

            # Compute the mean Pearson correlation across all cells
            mean_pearson_non_zero = torch.tensor(mean_pearson_non_zero).mean()

            # MSE
            mse = self.metric['mse'](pred_counts, true_counts)
            mean_mse = torch.mean(mse)

            mse_shuffled = self.metric['mse'](pred_counts, true_counts_shuffled)
            mean_mse_shuffled = torch.mean(mse_shuffled)

            # set up anndata object for subsetting by condition
            meta_dict = self.cell_metadata

            meta_dict.pop('counts', None)
            meta_dict.pop('size_factor', None)

            for k, v in meta_dict.items():
                if isinstance(v[0], torch.Tensor):
                    meta_dict[k] = torch.cat(v).cpu().numpy().tolist()
                else:
                    # flatten list of lists
                    meta_dict[k] = [item for sublist in v for item in sublist]

            meta = pd.DataFrame(meta_dict)

            if 'batch_key' not in meta.columns:
                meta['batch_key'] = 'single_condition'

            adata_true = sc.AnnData(X=true_counts.cpu().numpy(), obs=meta)
            adata_pred = sc.AnnData(X=pred_counts.cpu().numpy(), obs=meta)

            mmd = evaluate_mmd(adata_true, adata_pred, condition_key='batch_key')

            mmd.to_csv(os.path.join(self.output_dir, 'global_recon_mmd.csv'))

            emd = evaluate_emd(adata_true, adata_pred, condition_key='batch_key')
            emd.to_csv(os.path.join(self.output_dir, 'global_recon_emd.csv'))

            # count zero values in true and predicted
            true_zeros = torch.sum(true_counts == 0).item()
            pred_zeros = torch.sum(pred_counts == 0).item()
            true_prop_zeros = true_zeros / true_counts.numel()
            pred_prop_zeros = pred_zeros / pred_counts.numel()

            # write to disk
            metrics_df = pd.DataFrame(
                {
                    'metric': [
                        'pearson',
                        'pearson_shuffled',
                        'pearson_non_zero',
                        'mse',
                        'mse_shuffled',
                        'true_zeros',
                        'pred_zeros',
                        'true_prop_zeros',
                        'pred_prop_zeros',
                        'max true counts',
                        'max pred counts',
                    ],
                    'value': [
                        mean_pearson.item(),
                        mean_pearson_shuffled.item(),
                        mean_pearson_non_zero.item(),
                        mean_mse.item(),
                        mean_mse_shuffled.item(),
                        true_zeros,
                        pred_zeros,
                        true_prop_zeros,
                        pred_prop_zeros,
                        true_counts.max().item(),
                        pred_counts.max().item(),
                    ],
                }
            )

            metrics_df.to_csv(
                os.path.join(self.output_dir, 'random_baseline_metrics.csv'),
                index=False,
            )

            return metrics_df

        # Main function: saving GP embeddings

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

        # get cell type predictions
        if self.model_type == 'Global':
            cell_token = torch.cat(self.cell_token).cpu().numpy()

            if self.global_loss == 'supervised':
                for t in self.model.supervised_tasks:
                    logits = torch.cat(self.test_clf_pred[t])
                    predicted_classes = torch.argmax(logits, dim=1)
                    meta_dict[f'{t}_pred_encoded'] = predicted_classes.cpu().numpy()

                    if self.return_classification_report:
                        true_classes = np.array(self.cell_metadata[t])
                        predicted_classes = np.array(meta_dict[f'{t}_pred_encoded'])
                        report = classification_report(
                            true_classes, predicted_classes, output_dict=True
                        )
                        output_df = wrangle_classification_report(report)
                        output_df.to_csv(
                            os.path.join(
                                self.output_dir, f'{t}_classification_report.csv'
                            ),
                            index=False,
                        )

        meta = pd.DataFrame(meta_dict)

        # add non encoded string version of predicted labels
        if self.model_type == 'Global':
            if self.global_loss == 'supervised':
                for t in self.model.supervised_tasks:
                    conversion = meta[[t, f'{t}_pred_encoded']].drop_duplicates()
                    conversion = {
                        k: v
                        for k, v in zip(conversion[f'{t}_pred_encoded'], conversion[t])
                    }
                    meta[f'{t}_pred'] = meta[f'{t}_pred_encoded'].map(conversion)

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

        # change back missing values
        for column in adata.obs.columns:
            adata.obs[column] = np.where(
                adata.obs[column] == ' ', np.nan, adata.obs[column]
            )

        adata.write_h5ad(os.path.join(self.output_dir, 'adata_gp_embedding.h5ad'))

        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.umap(adata, min_dist=0.4)

        adata.write_h5ad(os.path.join(self.output_dir, 'adata_gp_embedding.h5ad'))

        if self.model_type == 'Global':
            bdata = sc.AnnData(X=cell_token, obs=meta)
            sc.pp.neighbors(bdata, use_rep='X')
            sc.tl.umap(bdata, min_dist=0.4)

            for column in bdata.obs.columns:
                bdata.obs[column] = np.where(
                    bdata.obs[column] == ' ', np.nan, bdata.obs[column]
                )

            bdata.write_h5ad(os.path.join(self.output_dir, 'adata_cell_embedding.h5ad'))

        # reset
        self.gp_cls = []
        self.cell_metadata = {}
        self.cell_token = []

    def _end_test_epoch_genes(self):
        # Concatenate tensors
        x_scgpl = torch.cat(self.x_scgpl, dim=0).cpu().numpy()
        tokens_scgpl = torch.cat(self.tokens_scgpl, dim=0).cpu().numpy()

        # flatten list
        gp_labels = np.array(
            [item for sublist in self.gp_labels for item in sublist]
        ).flatten()

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
        if self.gp != 'cell_token':
            gp_idx = self.model.gp_inputs.index(self.gp)
            tokens = pd.Series(
                getattr(
                    self.model.multi_gp_encoder, f'gp{gp_idx}_tokens_encoded'
                ).keys()
            )
            ensembl_ids = tokens.map(token_to_gene)
            gene_names = ensembl_ids.map(ensembl_to_name)

        if self.gp == 'cell_token':
            if self.model_type == 'Global':
                adata.var_names = ['cls'] + list(self.model.gp_inputs)
            else:
                adata.var_names = list(self.model.gp_inputs)
        else:
            if self.model_type == 'Mean':
                adata.var_names = list(ensembl_ids)
                adata.var['token'] = pd.Series(list(tokens), dtype=str).tolist()
                adata.var['ensembl'] = list(ensembl_ids)
                adata.var['gene'] = list(gene_names)
            else:
                adata.var_names = ['cls'] + list(ensembl_ids)
                adata.var['token'] = ['cls'] + pd.Series(
                    list(tokens), dtype=str
                ).tolist()
                adata.var['ensembl'] = ['cls'] + list(ensembl_ids)
                adata.var['gene'] = ['cls'] + list(gene_names)

        warnings.warn('Converting X array to dense format for writing to disk')
        adata.X = adata.X.toarray()

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
        loss = 0

        # import time
        # t0 = time.time()

        for i in range(len(self.model.gp_inputs)):
            # Loss
            if (
                self.model.multi_gp_encoder.encoder[i]
                .blocks[0]
                .attn.qkv.weight.requires_grad
            ):
                loss_i = F.cross_entropy(
                    output['logits_lm_list'][i].reshape(
                        -1, output['logits_lm_list'][i].shape[-1]
                    ),
                    output['gene_labels_list'][i].reshape(-1),
                )

                # if torch.isnan(loss_i):
                #     # usually happens if all labels are masked
                #     print(f'Loss is NaN in {self.model.gp_inputs[i]}')
                #     print('Predictions:')
                #     print(output['logits_lm_list'][i])
                #     print('')
                #     print('True labels:')
                #     print(output['gene_labels_list'][i])
                #     print('')
                #     print('Number of NaNs in predictions:')
                #     print(torch.isnan(output['logits_lm_list'][i]).sum())
                #     print('')
                #     print('Number of NaNs in true labels:')
                #     print(torch.isnan(output['gene_labels_list'][i]).sum())
                #     gp_loss_dict[self.model.gp_inputs[i]] = (
                #         torch.tensor(0).to(loss_i.device).float()
                #     )

                # else:
                gp_loss_dict[self.model.gp_inputs[i]] = loss_i
                loss += loss_i

                # t2 = time.time()
                # print(f'Time taken after checking nan', t2 - t0)

            else:
                gp_loss_dict[self.model.gp_inputs[i]] = (
                    torch.tensor(0).to(output['logits_lm_list'][i].device).float()
                )

            # compute total loss
        #     tensor_list = list(gp_loss_dict.values())

        # loss = torch.sum(torch.stack(tensor_list))
        # t3 = time.time()
        # print(f'Time taken after loss calculation: {t3 - t0}')

        # package outputs to return flexible number of objects
        holder = {
            'loss_per_gp': gp_loss_dict,
        }

        if self.use_gp_similarity_loss:
            gp_similarity_loss = self.compute_gp_similarity_loss(output['z'])
            loss += self.lambda_gp_similarity * gp_similarity_loss
            holder['gp_similarity_loss'] = gp_similarity_loss

        if self.model_type == 'Global':
            if self.global_loss == 'supervised':
                clf_loss_dict = {}

                for t in self.model.supervised_tasks:
                    clf_loss = self.compute_clf_loss(output[f'logits_{t}'], batch[t])
                    clf_loss_dict[t] = clf_loss
                    loss += self.lambda_clf_loss[t] * clf_loss

                holder['loss_clf'] = clf_loss_dict

            elif self.global_loss == 'mse':
                embedding_mse_loss = F.mse_loss(output['cell_token'], output['gf_emb'])
                holder['embedding_mse_loss'] = embedding_mse_loss
                loss += embedding_mse_loss

            elif self.global_loss == 'masking':
                cell_masking_loss = F.cross_entropy(
                    output['gp_logits_lm'].reshape(
                        -1, output['gp_logits_lm'].shape[-1]
                    ),
                    output['gp_labels'].reshape(-1),
                )

                holder['cell_masking_loss'] = cell_masking_loss
                loss += cell_masking_loss

            elif self.global_loss == 'reconstruction':
                reconstruction_loss = self.compute_count_loss(output, batch)
                holder['reconstruction_loss'] = reconstruction_loss
                loss += reconstruction_loss

                if self.model.reconstruction_loss in ['mse']:
                    self.train_pred_counts_list.append(
                        output['count_output']['count_lognorm']
                    )
                    self.train_true_counts_list.append(batch['counts'])

                if self.model.reconstruction_loss in ['nb', 'zinb']:
                    self.train_pred_counts_list.append(
                        output['count_output']['count_mean']
                    )
                    self.train_true_counts_list.append(batch['counts'])

                if self.model.reconstruction_loss in ['binning']:
                    self.train_pred_counts_list.append(output['count_output'])
                    self.train_true_counts_list.append(output['true_bins'])

        # t5 = time.time()
        # print('Skipped global stuff', t5-t3)
        holder['total_loss'] = loss

        # t6 = time.time()
        # print(f'Time taken for loss calculation: {t6 - t0}')

        return holder

    def compute_gp_similarity_loss(self, z):
        # calculate pairwise cosine similarity
        cs = []
        for i in range(z.shape[0]):
            c = pairwise_cosine_similarity(z[i, :])
            cs.append(c)
        gp_cosine_similarity = torch.stack(cs)

        gp_similarity = torch.tensor(self.gp_similarity).to(gp_cosine_similarity.device)

        # compute loss
        gp_similarity_loss = F.mse_loss(gp_cosine_similarity, gp_similarity)

        return gp_similarity_loss

    def compute_clf_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def compute_count_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ):
        true_counts = batch['counts']
        batch_size_factor = torch.tensor(batch['size_factor']).to(true_counts.device)

        if self.model.reconstruction_loss == 'mse':
            loss = (
                mse_loss(outputs['count_output']['count_lognorm'], true_counts)
                .sum(dim=-1)
                .mean()
                .float()
            )
            return loss

        elif self.model.reconstruction_loss == 'zinb':
            dec_mean_gamma, dec_dropout = (
                outputs['count_output']['count_mean'],
                outputs['count_output']['count_dropout'],
            )
            size_factor_view = batch_size_factor.unsqueeze(1).expand(
                dec_mean_gamma.size(0), dec_mean_gamma.size(1)
            )
            dec_mean = dec_mean_gamma * size_factor_view

            dispersion = F.linear(
                one_hot_encoder(batch['batch_key_id'], self.n_conditions_combined),
                self.theta,
            )
            dispersion = torch.exp(dispersion)
            loss = (
                -zinb(x=true_counts, mu=dec_mean, theta=dispersion, pi=dec_dropout)
                .sum(dim=-1)
                .mean()
            )
            return loss

        elif self.model.reconstruction_loss == 'nb':
            dec_mean_gamma = outputs['count_output']['count_mean']
            size_factor_view = batch_size_factor.unsqueeze(1).expand(
                dec_mean_gamma.size(0), dec_mean_gamma.size(1)
            )
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(
                one_hot_encoder(batch['batch_key_id'], self.n_conditions_combined),
                self.theta,
            )
            dispersion = torch.exp(dispersion)
            loss = -nb(x=true_counts, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()
            return loss

        elif self.reconstruction_loss == 'binning':
            pred = outputs['count_output']
            true = outputs['true_bins'].float()

            # calcualte mse loss
            loss = F.mse_loss(pred, true)

            return loss

        else:
            raise ValueError(
                'Reconstruction loss not supported' 'Please choose from mse, nb or zinb'
            )

    def configure_optimizers(self):
        # Define optimizer and may be consider weight decay
        # to improve generalization L2 regularization

        # add custom learning rate for cell_token_learner if exists:
        params = list(self.model.named_parameters())

        def add_custom_lr(n):
            return 'multi_gp_encoder' in n

        if self.use_finetune_lr:
            grouped_parameters = [
                {
                    'params': [p for n, p in params if add_custom_lr(n)],
                    'lr': self.lr,
                },
                {
                    'params': [p for n, p in params if not add_custom_lr(n)],
                    'lr': self.finetune_lr,
                },
            ]
        else:
            grouped_parameters = [{'params': [p for n, p in params], 'lr': self.lr}]

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


########################################
# For evaluating learned embeddings
########################################


class EmbEvaluator(pl.LightningModule):
    def __init__(
        self, n_classes, emb_dim, task, lr, emb_label, y_label, output_dir, filter_tag
    ):
        super().__init__()
        self.save_hyperparameters()

        self.evaluator_head = EmbEvaluatorHead(emb_dim, n_classes)
        self.emb_label = emb_label

        if task == 'classification':
            # add id tag for encoded covariate
            if not y_label.endswith('_id'):
                y_label = f'{y_label}_id'
        self.y_label = y_label
        self.task = task
        self.output_dir = output_dir
        self.filter_tag = filter_tag

        if task == 'classification':
            self.loss_fn = nn.CrossEntropyLoss()
        elif task == 'regression':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError('Task must be either classification or regression')

        self.lr = lr

        # for tracking
        self.y_unencoded = []
        for stage in ['train', 'val', 'test']:
            setattr(self, f'{stage}_pred', [])
            setattr(self, f'{stage}_true', [])

    def forward(self, x):
        return self.evaluator_head(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.emb_label]
        y = batch[self.y_label]

        y_out = self.evaluator_head(x)

        loss = self.loss_fn(y_out, y)

        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.train_pred.append(y_out)
        self.train_true.append(y)

        return loss

    def on_train_epoch_end(self):
        # calculate accuracy
        if self.task == 'classification':
            pred = torch.cat(self.train_pred)
            true = torch.cat(self.train_true)
            acc = (pred.argmax(dim=1) == true).float().mean()
            self.log(
                'train_accuracy',
                acc,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        elif self.task == 'regression':
            pred = torch.cat(self.train_pred)
            true = torch.cat(self.train_true)
            mse = self.loss_fn(pred, true)
            self.log('train_mse', mse, on_epoch=True, prog_bar=True, logger=True)

            # calculate pearson correlation
            pearson = torch.corrcoef(pred, true)[0, 1]
            self.log(
                'train_pearson', pearson, on_epoch=True, prog_bar=True, logger=True
            )

        # reset
        self.train_pred = []
        self.train_true = []

    def validation_step(self, batch, batch_idx):
        x = batch[self.emb_label]

        y = batch[self.y_label]

        y_out = self.evaluator_head(x)

        loss = self.loss_fn(y_out, y)
        self.log(
            'val_loss',
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.val_pred.append(y_out)
        self.val_true.append(y)

        return loss

    def on_validation_epoch_end(self):
        # calculate accuracy
        if self.task == 'classification':
            pred = torch.cat(self.val_pred)
            true = torch.cat(self.val_true)
            acc = (pred.argmax(dim=1) == true).float().mean()
            self.log('val_accuracy', acc, on_epoch=True, prog_bar=True, logger=True)

        elif self.task == 'regression':
            pred = torch.cat(self.val_pred)
            true = torch.cat(self.val_true)
            mse = self.loss_fn(pred, true)
            self.log('val_mse', mse, on_epoch=True, prog_bar=True, logger=True)

            # calculate pearson correlation
            pearson = torch.corrcoef(pred, true)[0, 1]
            self.log('val_pearson', pearson)

        # reset
        self.val_pred = []
        self.val_true = []

    def test_step(self, batch, batch_idx):
        x = batch[self.emb_label]
        y = batch[self.y_label]

        label_name = self.y_label
        label_name = label_name.replace('_id', '')
        y_unencoded = batch[label_name]

        y_out = self.evaluator_head(x)

        self.test_pred.append(y_out)
        self.test_true.append(y)
        self.y_unencoded += y_unencoded

    def on_test_epoch_end(self):
        # calculate accuracy
        if self.task == 'classification':
            pred = torch.cat(self.test_pred)
            true = torch.cat(self.test_true)
            acc = (pred.argmax(dim=1) == true).float().mean()
            self.log('test_accuracy', acc)

            # output classification report
            true_classes = true.cpu().numpy()
            predicted_classes = pred.argmax(dim=1).cpu().numpy()
            report = classification_report(
                true_classes, predicted_classes, output_dict=True
            )

            output_df = wrangle_classification_report(report)

            # convert labels back to original strings
            original_labels = self.y_unencoded
            conversion_df = pd.DataFrame(
                {
                    'encoded': true_classes,
                    'original': original_labels,
                }
            ).drop_duplicates()

            conversion_dict = {
                str(k): v
                for k, v in zip(conversion_df['encoded'], conversion_df['original'])
            }

            output_df = output_df[
                ~output_df['output_class'].isin(['macro avg', 'weighted avg'])
            ]
            output_df['output_class'] = output_df['output_class'].astype(str)
            output_df['output_class'] = output_df['output_class'].map(conversion_dict)

            output_df.to_csv(
                os.path.join(
                    self.output_dir,
                    f'cell_metrics/{self.y_label}_from_{self.emb_label}'
                    f'{self.filter_tag}.csv',
                ),
                index=False,
            )

        elif self.task == 'regression':
            pred = torch.cat(self.test_pred)
            true = torch.cat(self.test_true)
            mse = self.loss_fn(pred, true)
            self.log('test_mse', mse)

            # calculate pearson correlation
            pearson = torch.corrcoef(pred, true)[0, 1]
            self.log('test_pearson', pearson)

            # output to csv
            df = pd.DataFrame(
                {
                    'MSE': [mse.item()],
                    'Pearson': [pearson.item()],
                    'Max true': [true.max().item()],
                    'Max pred': [pred.max().item()],
                    'Min true': [true.min().item()],
                    'Min pred': [pred.min().item()],
                }
            )

            df.to_csv(
                os.path.join(
                    self.output_dir,
                    f'cell_metrics/{self.y_label}_from_{self.emb_label}.csv',
                ),
                index=False,
            )

        # reset
        self.test_pred = []
        self.test_true = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


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
