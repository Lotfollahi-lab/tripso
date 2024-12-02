import os
import warnings
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report, roc_auc_score
from torch import optim
from torchmetrics import MeanSquaredError, PearsonCorrCoef

from ..Models.gp_model import EmbEvaluatorHead
from ..Utils.losses import compute_count_loss, compute_gp_similarity_loss
from ..Utils.utils import (
    CosineLRwithWarmUp,
    evaluate_gene_expr_reconstruction,
    get_gp_tokens,
    wrangle_classification_report,
)

# ------------------------------------------------------
# Base trainers
# ------------------------------------------------------


class gpBase(pl.LightningModule):
    """
    Description:
    ------------
    Trainer for gpTransformer model with tokenized scRNA-seq dataset as input.
    This module encompasses the following steps:
    1. Initialise model
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
        output_dir: str = '/path/to/output',
        # GP similarity -> force cosine similarity of <GP> towards
        # similarity (defined by GP overlap)
        use_gp_similarity_loss: bool = False,
        lambda_gp_similarity=1e-2,
        gp_similarity: Optional[str] = None,
        # Use GO -> regularize attention matrix towards GO importance
        use_go_similarity_loss: bool = False,
        lambda_go_similarity=1e-2,
        go_similarity: Optional[pd.DataFrame] = None,
        go_similarity_gp: Optional[str] = 'hvg',  # GP to apply GO similarity loss:
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
        genes_to_keep: Optional[List] = None,
        gene_dir_tag: Optional[str] = None,
        return_attention: bool = False,
        gp: Optional[str] = None,
        finetune_lr: Union[float, dict] = 1e-5,
        use_finetune_lr: bool = False,
        save_emb: bool = False,
        split_label: str = 'train',
        hparam_save: str = 'all',
        set_gpfinder_weight_decay: Optional[float] = None,
        calc_gp_loss: bool = True,
    ) -> None:
        super().__init__()
        # save hyperparameters
        if hparam_save == 'all':
            # important that this is default for model training
            self.save_hyperparameters()
        else:
            # ignore model to avoid yaml error
            self.save_hyperparameters(ignore=['model'])

        # setup model
        self.model = model
        self.model_type = 'Base'
        self.calc_gp_loss = calc_gp_loss

        if use_gp_similarity_loss and gp_similarity is None:
            raise ValueError(
                'If use_gp_similarity_loss is True, gp_similarity_file must be provided'
            )

        self.use_gp_similarity_loss = use_gp_similarity_loss
        self.gp_similarity = gp_similarity
        self.lambda_gp_similarity = lambda_gp_similarity
        self.go_similarity_gp = go_similarity_gp

        if use_go_similarity_loss and go_similarity is None:
            raise ValueError(
                'If use_go_similarity_loss is True, go_similarity_file must be provided'
            )

        self.use_go_similarity_loss = use_go_similarity_loss
        self.lambda_go_similarity = lambda_go_similarity
        self.go_similarity_gp = go_similarity_gp

        if go_similarity is not None:
            go_similarity_tensor = torch.tensor(go_similarity.values).float()
            self.register_buffer('go_similarity', go_similarity_tensor)

            self.go_genes = go_similarity.index

            # Check order of genes
            gp_index = self.model.gp_inputs.index(go_similarity_gp)
            gp_tokens = (
                (getattr(self.model.multi_gp_encoder, f'gp{gp_index}_tokens'))
                .cpu()
                .numpy()
            )

            # Convert go genes to tokens
            go_tokens = np.array(
                list(
                    get_gp_tokens(
                        pd.Series(go_similarity.index),
                        do_ensembl_conversion=self.model.do_ensembl_conversion,
                        gp_name=go_similarity_gp,
                        gene_token_path=self.model.gene_token_path,
                        gene_name_path=self.model.gene_name_path,
                    )
                )
            )

            # Check identical:
            assert np.all(gp_tokens == go_tokens), 'GO genes do not match GP tokens'

        else:
            self.go_similarity = go_similarity

        self.lambda_gp_similarity = lambda_gp_similarity

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
        self.set_gpfinder_weight_decay = set_gpfinder_weight_decay

        # Initialise list to append loss and accuracy
        for stage in ['train', 'val', 'test']:
            setattr(self, f'{stage}_loss_per_gp', {})
            setattr(self, f'{stage}_mgm_gene_pred', {})
            setattr(self, f'{stage}_mgm_gene_true', {})

            setattr(self, f'{stage}_gp_similarity_loss', [])

            setattr(self, f'{stage}_loss', [])

        self.output_dir = output_dir

        # For output - cells
        self.gp_cls: List[float] = []
        self.cell_metadata: Dict[str, Union[str, float]] = {}
        self.cell_token: List[float] = []

        # For output - attention
        self.attn_scores: List[float] = []

        # for test step
        self.return_gene_embeddings = return_gene_embeddings
        self.tokens_to_keep = tokens_to_keep
        self.genes_to_keep = genes_to_keep

        self.gene_dir_tag = gene_dir_tag
        self.return_attention = return_attention
        self.gp = gp

        # for saving embeddings
        self.emb_dataset = None
        self.gene_dataset = None
        self.token_dataset = None
        self.attn_adata_holder: List[ad.AnnData] = []

    def forward(self, x, masking):
        out = self.model(
            x,
            masking=masking,
            return_gene_embeddings=self.return_gene_embeddings,
            tokens_to_keep=self.tokens_to_keep,
            gp_of_interest=self.gp,
            return_attention=self.return_attention,
        )

        return out

    def log_gp_loss(self, loss_per_gp):
        for i, gp in enumerate(self.model.gp_inputs):
            # only log if requires_grad = True
            if (
                self.model.multi_gp_encoder.encoder[i]
                .blocks[0]
                .attn.qkv.weight.requires_grad
            ):
                # if True:
                self.log(
                    f'train/{gp}_MGM_loss',
                    loss_per_gp[gp],
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    sync_dist=True,
                )

    def training_step(self, batch, batch_idx):
        output = self.forward(batch, masking=True)

        loss_output = self.compute_gp_loss(batch, output)

        loss_per_gp = loss_output['loss_per_gp']
        loss = loss_output['total_loss']

        if self.calc_gp_loss:
            self.log_gp_loss(loss_per_gp)

        self.log(
            'train/loss',
            loss,
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
        setattr(self, f'{stage}_mgm_gene_pred', {})
        setattr(self, f'{stage}_mgm_gene_true', {})
        setattr(self, f'{stage}_gp_similarity_loss', [])
        setattr(self, f'{stage}_loss', [])

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch, masking=True)

        loss_output = self.compute_gp_loss(batch, output)

        loss = loss_output['total_loss']
        perp = torch.exp(loss)

        self.log(
            'val/loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            'val/perplexity',
            perp,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        if self.save_emb:
            output = self.forward(batch, masking=False)

            emb_dict = {}

            for i, gp in enumerate(self.model.gp_inputs):
                emb_dict[gp] = output['z'][:, i, :].detach().cpu()
                emb_dict[f'{gp}_num_genes'] = (
                    output['num_genes_per_cell_list'][i].cpu().numpy().T
                )

            if 'cell_token' in output:
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

        if self.return_gene_embeddings:
            output = self.forward(batch, masking=False)

            emb_dict = {}

            # Get embeddings of the relevant genes
            for i, gene in enumerate(self.tokens_to_keep):
                gene_name = self.genes_to_keep[i]

                emb_dict[gene_name] = output[gene].detach().cpu()
                emb_dict[f'{gene_name}_rank'] = output[f'{gene}_rank'].detach().cpu()

            # metadata
            for k, v in batch.items():
                if k != 'input_ids':
                    emb_dict[k] = v

            emb = Dataset.from_dict(emb_dict)

            if self.gene_dataset is None:
                self.gene_dataset = emb
            else:
                self.gene_dataset = concatenate_datasets([self.gene_dataset, emb])

            return None

        if self.return_attention:
            # returns a dictionary where each gene is a key
            if self.gp != 'cell_token':
                output = self.model.get_cls_attn(batch, self.gp)
            else:
                # for cell token (only implemented for global model)
                output = self.model.get_cell_token_attention(batch)

            # add metadata
            meta_dict = {}
            for k, v in batch.items():
                if k != 'input_ids':
                    if isinstance(v, torch.Tensor):
                        meta_dict[k] = v.cpu().numpy()
                    else:
                        meta_dict[k] = v

            adata = sc.AnnData(
                csr_matrix(pd.DataFrame(output).values),
                obs=pd.DataFrame(meta_dict),
                var=pd.DataFrame(index=list(output.keys())),
            )

            self.attn_adata_holder.append(adata)

            # if self.attn_adata_holder is None:
            #     self.attn_adata_holder = adata
            # else:
            #     self.attn_adata_holder = ad.concat([self.attn_adata_holder, adata])

            return None

    def on_test_epoch_end(self):
        if self.save_emb:
            output_path = os.path.join(self.output_dir, 'embeddings')
            os.makedirs(output_path, exist_ok=True)
            output_name = os.path.join(output_path, f'{self.split_label}_set')
            self.emb_dataset.save_to_disk(output_name)
            self.emb_dataset = None
            return None

        if self.return_gene_embeddings:
            output_path = os.path.join(self.output_dir, self.gene_dir_tag)
            os.makedirs(output_path, exist_ok=True)
            output_name = os.path.join(output_path, f'{self.split_label}_set')
            self.gene_dataset.save_to_disk(output_name)
            self.gene_dataset = None
            return None

        if self.return_attention:
            output_path = os.path.join(self.output_dir, 'attention')
            os.makedirs(output_path, exist_ok=True)
            adata = ad.concat(self.attn_adata_holder)
            adata.write_h5ad(
                os.path.join(
                    output_path, f'{self.gp}_attention_{self.split_label}_set.h5ad'
                )
            )

            return None

    def compute_gp_loss(self, batch, fw_pass_output):
        output = fw_pass_output

        # calculate MLM loss for each GP
        gp_loss_dict = {}
        loss = 0

        for i in range(len(self.model.gp_inputs)):
            # Loss
            if self.calc_gp_loss and (
                (
                    self.model.multi_gp_encoder.encoder[i]
                    .blocks[0]
                    .attn.qkv.weight.requires_grad
                )
            ):
                loss_i = F.cross_entropy(
                    output['logits_lm_list'][i].reshape(
                        -1, output['logits_lm_list'][i].shape[-1]
                    ),
                    output['gene_labels_list'][i].reshape(-1),
                )

                gp_loss_dict[self.model.gp_inputs[i]] = loss_i
                loss += loss_i

            else:
                gp_loss_dict[self.model.gp_inputs[i]] = (
                    torch.tensor(0).to(output['logits_lm_list'][i].device).float()
                )

        # package outputs to return flexible number of objects
        holder = {
            'loss_per_gp': gp_loss_dict,
        }

        if self.use_gp_similarity_loss:
            gp_similarity_loss = compute_gp_similarity_loss(
                output['z'], self.gp_similarity
            )
            loss += self.lambda_gp_similarity * gp_similarity_loss
            holder['gp_similarity_loss'] = gp_similarity_loss

        if self.use_go_similarity_loss:
            # only implemented for single GP for now
            # otherwise would need one matrix per GP
            gp_idx = self.model.gp_inputs.index(self.go_similarity_gp)
            output_attn = self.model.get_last_self_attn(batch, gp_idx)

            go_similarity_loss = F.mse_loss(output_attn['attn'], self.go_similarity)
            loss += self.lambda_go_similarity * go_similarity_loss
            holder['go_similarity_loss'] = go_similarity_loss

        holder['total_loss'] = loss

        return holder

    def configure_optimizers(self):
        params = list(self.model.named_parameters())

        def get_lr_for_param(name):
            """Determine the learning rate for a parameter based on its name."""
            if isinstance(self.finetune_lr, float):
                # Use finetune_lr for specific parameter names
                if 'multi_gp_encoder' in name or 'gf_wrapper' in name:
                    return self.finetune_lr
                else:
                    return self.lr
            elif isinstance(self.finetune_lr, dict):
                # Use the learning rate from the dict if a key matches part of the name
                for key, lr in self.finetune_lr.items():
                    if key in name:
                        return lr
                # Default to self.lr if no key matches
                return self.lr
            else:
                raise ValueError('finetune_lr must be either a float or a dict.')

        # Group parameters with their respective learning rates
        lr_to_params = {}
        for name, param in params:
            if not param.requires_grad:
                continue
            lr = get_lr_for_param(name)
            if lr not in lr_to_params:
                lr_to_params[lr] = []
            lr_to_params[lr].append(param)

        grouped_parameters = [
            {'params': param_list, 'lr': lr} for lr, param_list in lr_to_params.items()
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


class gpGlobal(gpBase):
    def __init__(
        self,
        global_loss: str = 'supervised',
        lambda_clf_loss=1,
        return_classification_report: bool = False,
        total_n_genes: int = 20_000,
        n_condition_combined: int = 1,  # number of batches for zinb and nb
        test_random_baseline: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = 'Global'
        self.global_loss = global_loss
        self.return_classification_report = return_classification_report
        self.total_n_genes = total_n_genes
        self.n_condition_combined = n_condition_combined
        self.test_random_baseline = test_random_baseline

        if self.global_loss == 'supervised':
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

        if self.global_loss == 'reconstruction':
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

        # For learning global cell token
        for stage in ['train', 'val', 'test']:
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

    def training_step(self, batch, batch_idx):
        output = self.forward(batch, masking=True)

        loss_base = self.compute_gp_loss(batch, output)

        if self.calc_gp_loss:
            self.log_gp_loss(loss_base['loss_per_gp'])

        if self.global_loss == 'supervised':
            clf_loss = self.compute_supervised_loss(output, batch, stage='train')
            loss = loss_base['total_loss'] + clf_loss['total_loss']

            # Log losses
            for t in self.model.supervised_tasks:
                self.log(
                    f'train/{t}_loss',
                    clf_loss[t],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

        elif self.global_loss == 'masking':
            cell_masking_loss = F.cross_entropy(
                output['gp_logits_lm'].reshape(-1, output['gp_logits_lm'].shape[-1]),
                output['gp_labels'].reshape(-1),
            )

            loss = loss_base['total_loss'] + cell_masking_loss

        elif self.global_loss == 'reconstruction':
            reconstruction_loss = self.compute_reconstruction_loss(
                batch, output, stage='train'
            )
            loss = loss_base['total_loss'] + reconstruction_loss

            # log loss
            self.log(
                f'train/{self.model.reconstruction_loss}_loss',
                reconstruction_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        self.log(
            'train/loss',
            loss,
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

        if self.global_loss == 'supervised':
            for t in self.model.supervised_tasks:
                # compute accuracy
                clf_pred = torch.cat(getattr(self, 'train_clf_pred')[t])
                clf_true = torch.cat(getattr(self, 'train_clf_true')[t])
                acc = torch.sum(clf_pred == clf_true).float() / clf_true.shape[0]

                self.log(
                    f'train/{t}_accuracy',
                    acc,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

                # empty lists
                getattr(self, 'train_clf_pred')[t] = []
                getattr(self, 'train_clf_true')[t] = []

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

            # TODO : properly sample the counts
            # mse = self.metric['mse'](pred_counts, true_counts)
            # mean_mse = torch.mean(mse)
            # self.log(
            #     'train/mse',
            #     mean_mse,
            #     on_epoch=True,
            #     prog_bar=True,
            #     logger=True,
            # )

        # empty lists
        self.train_true_counts_list = []
        self.train_pred_counts_list = []

        setattr(self, f'{stage}_loss', [])

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch, masking=True)

        loss_base = super().compute_gp_loss(batch, output)

        if self.global_loss == 'supervised':
            clf_loss = self.compute_supervised_loss(output, batch, stage='val')
            loss = loss_base['total_loss'] + clf_loss['total_loss']

        elif self.global_loss == 'masking':
            cell_masking_loss = F.cross_entropy(
                output['gp_logits_lm'].reshape(-1, output['gp_logits_lm'].shape[-1]),
                output['gp_labels'].reshape(-1),
            )

            loss = loss_base['total_loss'] + cell_masking_loss

        elif self.global_loss == 'reconstruction':
            reconstruction_loss = self.compute_reconstruction_loss(
                batch, output, stage='val'
            )
            loss = loss_base['total_loss'] + reconstruction_loss

            self.log(
                f'val/{self.model.reconstruction_loss}_loss',
                reconstruction_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        self.log(
            'val/loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        # reset step_output
        stage = 'val'

        if self.global_loss == 'supervised':
            acc_holder = []

            for t in self.model.supervised_tasks:
                # compute accuracy
                clf_pred = torch.cat(getattr(self, 'val_clf_pred')[t])
                clf_true = torch.cat(getattr(self, 'val_clf_true')[t])
                acc = torch.sum(clf_pred == clf_true).float() / clf_true.shape[0]

                acc_holder.append(acc)

                self.log(
                    f'val/{t}_accuracy',
                    acc,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    # sync_dist=True,
                )

                # empty lists
                getattr(self, 'val_clf_pred')[t] = []
                getattr(self, 'val_clf_true')[t] = []

            # Log accuracy across tasks for early stopping
            mean_acc = torch.mean(torch.tensor(acc_holder))
            self.log(
                'val/accuracy',
                mean_acc,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                # sync_dist=True,
            )

        if self.global_loss == 'reconstruction':
            # return Pearson correlation coefficient
            true_counts = torch.cat(self.val_true_counts_list).float()
            pred_counts = torch.cat(self.val_pred_counts_list)

            # Pearson correlation coefficient
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

            # TODO : properly sample the counts
            # mse = self.metric['mse'](pred_counts, true_counts)
            # mean_mse = torch.mean(mse)
            # self.log(
            #     'val/mse',
            #     mean_mse,
            #     on_epoch=True,
            #     prog_bar=True,
            #     logger=True,
            # )

        # empty lists
        self.val_true_counts_list = []
        self.val_pred_counts_list = []

        setattr(self, f'{stage}_loss', [])

    def test_step(self, batch, batch_idx):
        if self.save_emb or self.return_gene_embeddings or self.return_attention:
            super().test_step(batch, batch_idx)
            return None

        if self.global_loss == 'supervised':
            # track metadata for evaluation
            for k, v in batch.items():
                if k != 'input_ids':
                    if k in self.cell_metadata:
                        self.cell_metadata[k].append(v)
                    else:
                        self.cell_metadata[k] = [v]

            output = self.forward(batch, masking=False)

            for t in self.model.supervised_tasks:
                self.test_clf_pred[t].append(output[f'logits_{t}'])

        if self.test_random_baseline:
            # track metadata for evaluation
            for k, v in batch.items():
                if k != 'input_ids':
                    if k in self.cell_metadata:
                        self.cell_metadata[k].append(v)
                    else:
                        self.cell_metadata[k] = [v]

            output = self.forward(batch, masking=False)

            # store predicted counts and true counts
            self.test_true_counts_list.append(batch['counts'])
            if self.model.reconstruction_loss in ['mse']:
                self.test_pred_counts_list.append(
                    output['count_output']['count_lognorm']
                )
            if self.model.reconstruction_loss in ['nb', 'zinb']:
                self.test_pred_counts_list.append(output['count_output']['count_mean'])

    def on_test_epoch_end(self):
        if self.save_emb or self.return_gene_embeddings or self.return_attention:
            super().on_test_epoch_end()
            return None

        if self.test_random_baseline:
            true_counts = torch.cat(self.test_true_counts_list).float()
            pred_counts = torch.cat(self.test_pred_counts_list)

            # get metadata
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

            evaluate_gene_expr_reconstruction(
                true_counts,
                pred_counts,
                meta,
                self.output_dir,
            )

        if self.global_loss == 'supervised':
            # get metadata
            meta_dict = self.cell_metadata

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
                        os.path.join(self.output_dir, f'{t}_classification_report.csv'),
                        index=False,
                    )

        # reset
        self.gp_cls = []
        self.cell_metadata = {}
        self.cell_token = []

    def compute_supervised_loss(self, output, batch, stage):
        clf_loss_dict = {}
        loss = 0

        for t in self.model.supervised_tasks:
            clf_loss = F.cross_entropy(output[f'logits_{t}'], batch[t])
            clf_loss_dict[t] = clf_loss
            loss += self.lambda_clf_loss[t] * clf_loss

            # track for calculating accuracy
            getattr(self, f'{stage}_clf_pred')[t].append(
                torch.argmax(output[f'logits_{t}'], dim=1)
            )
            getattr(self, f'{stage}_clf_true')[t].append(batch[t])

        clf_loss_dict['total_loss'] = loss

        return clf_loss_dict

    def compute_reconstruction_loss(self, batch, output, stage):
        reconstruction_loss = compute_count_loss(
            output,
            batch,
            self.model.reconstruction_loss,
            self.theta,
            self.n_conditions_combined,
        )

        if self.model.reconstruction_loss in ['mse']:
            getattr(self, f'{stage}_pred_counts_list').append(
                output['count_output']['count_lognorm']
            )
            getattr(self, f'{stage}_true_counts_list').append(batch['counts'])

        if self.model.reconstruction_loss in ['nb', 'zinb']:
            getattr(self, f'{stage}_pred_counts_list').append(
                output['count_output']['count_mean']
            )
            getattr(self, f'{stage}_true_counts_list').append(batch['counts'])

        return reconstruction_loss


# ------------------------------------------------------
# Extra trainers
# ------------------------------------------------------


class gpPrototypes(gpGlobal):
    def __init__(
        self,
        lambda_prototype_loss: float = 10,  # 1e-2,
        prototype_labels_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.lambda_prototype_loss = lambda_prototype_loss
        self.prototype_labels_key = prototype_labels_key

    def training_step(self, batch, batch_idx):
        output = self.forward(batch, masking=True)

        # GP loss
        loss_base = self.compute_gp_loss(batch, output)

        # Reconstruction loss
        reconstruction_loss = self.compute_reconstruction_loss(
            batch, output, stage='train'
        )
        loss = loss_base['total_loss'] + reconstruction_loss

        self.log(
            f'train/{self.model.reconstruction_loss}_loss',
            reconstruction_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Prototype loss
        prototype_loss = self.compute_prototype_loss(
            output['cell_token'], batch[self.prototype_labels_key]
        )

        self.log(
            'train/prototype_loss',
            prototype_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss = loss + self.lambda_prototype_loss * prototype_loss

        self.log(
            'train/loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def compute_prototype_loss(self, Z, labels):
        # following DeepGSEA : prototype loss has 3 terms
        # p2p = loss for the pairwise prototype distance
        # where d_min is the minimum acceptable distance between prototypes
        # (in deep gsea defined as 1)
        # we use coarse cell type labels = use version with labels

        # Compute the pairwise distances between prototypes
        prototypes_flat = self.model.prototypes.view(-1, self.model.prototypes.size(-1))
        distances = torch.cdist(prototypes_flat, prototypes_flat)

        # Ensure we only consider each pair once by using
        # the upper triangular part of the distance matrix
        mask = torch.triu(torch.ones_like(distances), diagonal=1)
        masked_distances = distances * mask

        # Calculate the hinge loss
        distance_threshold = 1
        p2p_loss = torch.clamp(distance_threshold - masked_distances, min=0)
        p2p_loss = p2p_loss[mask.bool()].mean()

        # Step 2 : c2p = cell to prototype loss
        # encourages the model to minimize the distance from each cell
        # to the closest prototype with the same phenotype

        # Retrieve the prototype for each cell based on the labels
        prototypes_for_cells = self.model.prototypes[
            labels
        ]  # Shape: (batch_size, gp_latent_size)

        # Calculate the distances between each cell representation
        # and its corresponding prototype
        distances = torch.norm(Z - prototypes_for_cells, dim=1)  # Shape: (batch_size,)

        # Define the loss function to minimize
        # these distances (e.g., mean squared error)
        c2p_loss = distances.mean()

        # Finally, encourage the model to minimize the distance
        # from each prototype to the center of cells to which it is the closest
        num_prototypes = self.model.prototypes.size(0)
        gp_latent_size = self.model.prototypes.size(1)

        # Initialize a tensor to store
        # the sum of cell representations for each prototype
        prototype_sums = torch.zeros(num_prototypes, gp_latent_size, device=Z.device)
        # Initialize a tensor to store the count of cells assigned to each prototype
        prototype_counts = torch.zeros(num_prototypes, device=Z.device)

        # Accumulate sums and counts for each prototype based on cell labels
        for i in range(num_prototypes):
            mask = labels == i
            if mask.sum() > 0:
                prototype_sums[i] = Z[mask].sum(dim=0)
                prototype_counts[i] = mask.sum()

        # Compute the centers for each prototype
        prototype_centers = prototype_sums / prototype_counts.clamp(min=1).unsqueeze(1)

        # Compute the distances from each prototype to its center
        prototype_distances = torch.norm(
            self.model.prototypes - prototype_centers, dim=1
        )

        # Define the loss function to minimize these distances
        p2c_loss = prototype_distances.mean()

        # Combine the loss terms
        loss = p2p_loss + c2p_loss + p2c_loss

        return loss


# Prompt trainer not implemented
# --> see previous code for classification head
# specific for virtual tokens
# maybe need function to return_virtual_tokens too?


########################################
# For evaluating learned embeddings
########################################


class EmbEvaluator(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        emb_dim,
        task,
        lr,
        emb_label,
        y_label,
        output_dir,
        filter_tag,
        num_condition_cat=0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.evaluator_head = EmbEvaluatorHead(emb_dim, n_classes, num_condition_cat)
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
            on_step=False,
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

            output_df = output_df[
                ~output_df['output_class'].isin(['macro avg', 'weighted avg'])
            ]

            # ---------- Calculate ROC-AUC -----------------
            # count number of nan values
            pred_np = pred.cpu().numpy()
            idx = np.isnan(pred_np).any(axis=1)
            nans = idx.sum()
            if nans > 0:
                print('Number of nan values:', nans)

            # drop rows with nan
            warnings.warn(
                f'Dropping {len(idx)} nan values (out of {true_classes.shape[0]})'
                'for ROC-AUC calculation'
            )
            pred = pred[~idx]
            true_classes = true_classes[~idx]

            # Apply softmax with numerical stability
            max_pred = torch.max(pred, dim=-1, keepdim=True)[0]
            stabilized_pred = pred - max_pred
            class_proba = F.softmax(stabilized_pred, dim=-1).cpu().numpy()

            # check if multiclass
            if len(np.unique(true_classes)) > 2:
                roc_auc = roc_auc_score(
                    true_classes, class_proba, multi_class='ovr', average=None
                )
                roc_df = pd.DataFrame(
                    {
                        'class': np.unique(true_classes),
                        'roc_auc': roc_auc,
                    }
                )

                output_df['output_class'] = output_df['output_class'].astype(int)
                output_df = output_df.join(roc_df.set_index('class'), on='output_class')

            else:
                # Extract the probabilities for the positive class (class 1)
                y_score_positive_class = class_proba[:, 1]
                roc_auc = roc_auc_score(true_classes, y_score_positive_class)
                output_df['roc_auc'] = roc_auc

            # ---------- Wrangle output -----------------
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

            output_df['output_class'] = output_df['output_class'].astype(str)
            output_df['output_class'] = output_df['output_class'].map(conversion_dict)

            # check output directory exists
            if not os.path.exists(os.path.join(self.output_dir, 'cell_metrics')):
                os.makedirs(
                    os.path.join(self.output_dir, 'cell_metrics'), exist_ok=True
                )

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
        gp_latent_size=512,
    )

    gp_transformer = gpBase(
        model,
        lr=1e-3,
        total_epochs=1,
        output_dir='TEST',
    )

    trainer = pl.Trainer(max_epochs=1, devices=-1, accelerator='auto', precision=16)

    print('Running test dataloader')
    trainer.test(gp_transformer, txdata)
