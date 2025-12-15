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
from peft import LoraConfig, get_peft_model
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim
from torchmetrics import MeanSquaredError

from ..Models.gp_model import EmbEvaluatorHead, gpTransformerBase
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
        model: gpTransformerBase,
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
        lr: Union[float, Dict] = 1e-3,
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
        return_gene_cosim: Optional[str] = None,
        tokens_to_keep: Optional[List] = None,
        genes_to_keep: Optional[List] = None,
        token_to_gene_to_keep_dict: Optional[Dict] = None,
        gene_dir_tag: Optional[str] = None,
        return_attention: bool = False,
        return_mean_non_padding: bool = False,
        gp: Optional[str] = None,
        gp_for_downstream: Optional[str] = None,
        save_emb: bool = False,
        split_label: str = 'train',
        hparam_save: str = 'all',
        set_gpfinder_weight_decay: Optional[float] = None,
        calc_gp_loss: bool = True,
        calc_gene_loss: bool = False,
        warmup: Optional[int] = 0,
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
        self.calc_gene_loss = calc_gene_loss
        self.warmup = warmup

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
        self.save_emb = save_emb
        self.split_label = split_label
        self.set_gpfinder_weight_decay = set_gpfinder_weight_decay

        self.output_dir = output_dir

        # For output - cells
        self.gp_cls: List[float] = []
        self.cell_metadata: Dict[str, Union[str, float]] = {}
        self.cell_token: List[float] = []

        # For output - attention
        self.attn_scores: List[float] = []

        # for test step
        self.return_gene_embeddings = return_gene_embeddings
        self.return_gene_cosim = return_gene_cosim
        self.tokens_to_keep = tokens_to_keep
        self.token_to_gene_to_keep_dict = token_to_gene_to_keep_dict
        self.genes_to_keep = genes_to_keep
        self.return_mean_non_padding = return_mean_non_padding

        self.gene_dir_tag = gene_dir_tag
        self.return_attention = return_attention
        self.gp = gp
        self.gp_for_downstream = gp_for_downstream

        # for saving embeddings
        self.emb_dataset = None
        self.gene_dataset = None
        self.token_dataset = None
        self.attn_adata_holder: List[ad.AnnData] = []
        self.gene_to_gp_cosim: Dict[str, np.ndarray] = {}
        self.gene_meta_dict = None
        self.gene_to_gene_cosim: Dict[tuple, torch.Tensor] = {}

    def forward(self, x, masking, epoch, **kwargs):
        out = self.model(
            x,
            masking=masking,
            return_gene_embeddings=self.return_gene_embeddings,
            tokens_to_keep=self.tokens_to_keep,
            gp_of_interest=self.gp,
            return_attention=self.return_attention,
            epoch=epoch,
            return_mean_non_padding=self.return_mean_non_padding,
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
        output = self.forward(batch, masking=True, epoch=self.current_epoch)

        # Optionally calculate MLM for gene encoder
        if self.calc_gene_loss:
            gene_loss = self.compute_gene_loss(batch, output)
            self.log(
                'train/gene_masking_loss',
                gene_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            loss = gene_loss
        else:
            loss = torch.tensor(0.0).to(self.device)

        # GP masking loss
        if hasattr(self, 'warmup') and self.current_epoch >= self.warmup:
            if self.calc_gp_loss:
                loss_output = self.compute_gp_loss(batch, output)
                loss_per_gp = loss_output['loss_per_gp']
                self.log_gp_loss(loss_per_gp)
        else:
            loss_output = {'total_loss': torch.tensor(0.0).to(self.device)}

        loss += loss_output['total_loss']

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
        pass

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch, masking=True, epoch='val')

        if self.calc_gp_loss:
            loss_output = self.compute_gp_loss(batch, output)
        else:
            loss_output = {'total_loss': torch.tensor(0).to(self.device)}

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
            output = self.forward(
                batch, masking=False, epoch='test', masking_global=False
            )

            emb_dict = {}

            for i, gp in enumerate(self.model.gp_inputs):
                emb_dict[gp] = output['z'][:, i, :].detach().cpu()
                emb_dict[f'{gp}_num_genes'] = (
                    output['num_genes_per_cell_list'][i].cpu().numpy().T
                )

            if 'gene_encoder_cls' in output:
                emb_dict['gene_encoder_cls'] = output['gene_encoder_cls'].detach().cpu()

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
            output = self.forward(batch, masking=False, epoch=None)

            emb_dict = {}

            # Option 1: save the embeddings directly
            if self.return_gene_cosim is None:
                # Get embeddings of the relevant genes
                for i, gene in enumerate(self.tokens_to_keep):
                    gene_name = self.token_to_gene_to_keep_dict[gene]
                    emb_dict[gene_name] = output[gene].detach().cpu()
                    emb_dict[f'{gene_name}_rank'] = (
                        output[f'{gene}_rank'].detach().cpu()
                    )

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

            # Option 2: calculate (gene, GP) cosine similarity
            if self.return_gene_cosim == 'gene_to_gp':
                gp = output['z'][:, 0, :].squeeze().detach()
                for i, gene in enumerate(self.tokens_to_keep):
                    gene_name = self.token_to_gene_to_keep_dict[gene]
                    gene_emb = output[gene].detach()

                    cos_sim = F.cosine_similarity(gp, gene_emb)
                    cos_sim = cos_sim.cpu().numpy().squeeze()

                    # make sparse for memory efficiency
                    cos_sim = sparse.csr_matrix(cos_sim).T  # (1, batch) --> (batch, 1)

                    if gene_name not in self.gene_to_gp_cosim:
                        self.gene_to_gp_cosim[gene_name] = cos_sim
                    else:
                        # Use scipy.sparse.vstack to concatenate csr_matrices
                        self.gene_to_gp_cosim[gene_name] = sparse.vstack(
                            [self.gene_to_gp_cosim[gene_name], cos_sim]
                        )

                # Track metadata
                meta_dict = {}
                for k, v in batch.items():
                    if k != 'input_ids':
                        # if tensor, move to cpu
                        if isinstance(v, torch.Tensor):
                            v = v.cpu().numpy()
                        meta_dict[k] = v

                if self.gene_meta_dict is None:
                    self.gene_meta_dict = meta_dict
                else:
                    self.gene_meta_dict = {
                        k: np.concatenate([self.gene_meta_dict[k], v])
                        if isinstance(v, np.ndarray)
                        else self.gene_meta_dict[k] + v
                        for k, v in meta_dict.items()
                    }

            # Option 3: calculate (gene, gene) cosine similarity
            if self.return_gene_cosim == 'gene_to_gene':
                # Vectorized computation for gene-to-gene cosine similarity
                gene_embs = torch.stack(
                    [output[gene].detach() for gene in self.tokens_to_keep]
                )  # (num_genes, emb_dim)
                gene_names = [
                    self.token_to_gene_to_keep_dict[gene]
                    for gene in self.tokens_to_keep
                ]
                gene_embs_norm = F.normalize(gene_embs, p=2, dim=1)
                cos_sim_matrix = (
                    gene_embs_norm @ gene_embs_norm.T
                )  # (num_genes, num_genes)
                cos_sim_matrix_np = cos_sim_matrix.cpu().numpy()

                for i, gene1_name in enumerate(gene_names):
                    for j, gene2_name in enumerate(gene_names):
                        key = (gene1_name, gene2_name)
                        value = cos_sim_matrix_np[i, j]
                        if key not in self.gene_to_gene_cosim:
                            self.gene_to_gene_cosim[key] = np.array([value])
                        else:
                            self.gene_to_gene_cosim[key] = np.concatenate(
                                [self.gene_to_gene_cosim[key], np.array([value])]
                            )

            return None

        if self.return_attention:
            # returns a dictionary where each gene is a key
            if self.gp_for_downstream != 'cell_token':
                output = self.model.get_cls_attn(
                    batch, self.gp_for_downstream, masking=False
                )

                token_names = list(output.keys())

                gene_names = [
                    self.token_to_gene_to_keep_dict[t] if t != 'cls' else 'cls'
                    for t in token_names
                ]

            else:
                # for cell token (only implemented for global model)
                output = self.model.get_cell_token_attention(batch)
                gene_names = list(output.keys())

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
                var=pd.DataFrame(index=gene_names),
            )

            self.attn_adata_holder.append(adata)

            return None

    def on_test_epoch_end(self):
        if self.save_emb:
            if self.return_mean_non_padding:
                output_path = os.path.join(self.output_dir, 'mean_embeddings')
            else:
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

            if self.return_gene_cosim is None:
                self.gene_dataset.save_to_disk(output_name)
                self.gene_dataset = None

            if self.return_gene_cosim == 'gene_to_gp':
                # gene_to_gp_cosim is a dict where gene names are keys,
                # values are sparse matrices (n_cells, 1)
                # Stack all sparse matrices horizontally to get (n_cells, n_genes)
                gene_names = list(self.gene_to_gp_cosim.keys())
                cosim_matrix = sparse.hstack(
                    [self.gene_to_gp_cosim[gene] for gene in gene_names]
                )

                obs = pd.DataFrame(self.gene_meta_dict)

                # Pass sparse matrix directly to AnnData, set var_names
                output_adata = sc.AnnData(
                    X=cosim_matrix, obs=obs, var=pd.DataFrame(index=gene_names)
                )

                filename = (
                    f'gene_to_{self.gp}_cosine_similarity_{self.split_label}_set.h5ad'
                )
                filepath = os.path.join(output_path, filename)
                output_adata.write_h5ad(filepath)

                self.gene_to_gp_cosim = None
                self.gene_meta_dict = None

            if self.return_gene_cosim == 'gene_to_gene':
                # gene_to_gene_cosim is a dictionary where keys are gene tuples
                # values are cosine similarities
                # output a dataframe of shape (gene,gene)
                # with genes appropriately labelled
                # where x[i,j] is the mean cosine similarity
                gene_names = sorted(
                    set(
                        [k[0] for k in self.gene_to_gene_cosim.keys()]
                        + [k[1] for k in self.gene_to_gene_cosim.keys()]
                    )
                )
                cosim_matrix = pd.DataFrame(
                    index=gene_names, columns=gene_names, dtype=float
                )
                for (gene1, gene2), sims in self.gene_to_gene_cosim.items():
                    mean_sim = (
                        sims.mean().item() if hasattr(sims, 'mean') else np.mean(sims)
                    )
                    cosim_matrix.loc[gene1, gene2] = mean_sim
                output_path = os.path.join(self.output_dir, self.gene_dir_tag)
                os.makedirs(output_path, exist_ok=True)
                output_name = os.path.join(
                    output_path,
                    f'gene_to_gene_cosine_similarity_{self.split_label}_set.csv',
                )
                cosim_matrix.to_csv(output_name)
                self.gene_to_gene_cosim = {}
                return None

        if self.return_attention:
            output_path = os.path.join(self.output_dir, 'attention')
            os.makedirs(output_path, exist_ok=True)
            adata = ad.concat(self.attn_adata_holder)
            adata.write_h5ad(
                os.path.join(
                    output_path,
                    f'{self.gp_for_downstream}_attention_{self.split_label}_set.h5ad',
                )
            )

            print(
                'Saved attention adata to path:',
                os.path.join(
                    output_path,
                    f'{self.gp_for_downstream}_attention_{self.split_label}_set.h5ad',
                ),
            )

            return None

    def compute_gene_loss(self, batch, fw_pass_output):
        loss = F.cross_entropy(
            fw_pass_output['gene_mlm_logits'].reshape(
                -1, fw_pass_output['gene_mlm_logits'].shape[-1]
            ),
            fw_pass_output['gene_mlm_labels'].reshape(-1),
        )

        return loss

    def compute_gp_loss(self, batch, fw_pass_output):
        output = fw_pass_output

        # calculate MLM loss for each GP
        gp_loss_dict = {}
        loss = 0

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

                gp_loss_dict[self.model.gp_inputs[i]] = loss_i
                loss += loss_i

            else:
                gp_loss_dict[self.model.gp_inputs[i]] = torch.tensor(0).to(self.device)

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
            if isinstance(self.lr, float):
                return self.lr
            elif isinstance(self.lr, dict):
                # Use the learning rate from the dict if a key matches part of the name
                for key, lr in self.lr.items():
                    if key in name:
                        return lr
                # Default if no key matches
                if 'default' in self.lr:
                    return self.lr['default']
                else:
                    raise ValueError(
                        f'No matching lr for parameter {name},'
                        'and no default value provided'
                    )
            else:
                raise ValueError('lr must be either a float or a dict.')

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

        default_lr = self.lr if isinstance(self.lr, float) else self.lr['default']

        optimizer = self.optimizer_class(
            grouped_parameters, lr=default_lr, weight_decay=self.weight_decay
        )

        # Configure the learning rate scheduler.
        if self.lr_scheduler == 'ReduceLROnPlateau':
            print('Using ReduceLROnPlateau scheduler')
            LRscheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=2,  # default 10
                factor=0.1,  # default
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

            # Track cell-wise Pearson correlations
            self.train_pearson_corrs: List[torch.Tensor] = []
            self.val_pearson_corrs: List[torch.Tensor] = []

        # For learning global cell token
        for stage in ['train', 'val', 'test']:
            if self.global_loss == 'supervised':
                setattr(self, f'{stage}_clf_pred', {})
                setattr(self, f'{stage}_clf_true', {})

                for t in self.model.supervised_tasks:
                    setattr(self, f'{stage}_{t}_loss', [])
                    getattr(self, f'{stage}_clf_pred')[t] = []
                    getattr(self, f'{stage}_clf_true')[t] = []

    def forward(self, x, masking, masking_global=False, **kwargs):
        # masking_global is used to indicate
        # whether to apply masking in global transformer
        # set to False by default because we use the gpBase test_step()
        # for saving embeddings
        # epoch argument for compatibility with gpBase
        out = self.model(
            x,
            masking=masking,
            masking_global=masking_global,
            return_gene_embeddings=self.return_gene_embeddings,
            tokens_to_keep=self.tokens_to_keep,
            gp_of_interest=self.gp,
            return_attention=self.return_attention,
            epoch='Global',
            return_mean_non_padding=self.return_mean_non_padding,
        )

        return out

    def training_step(self, batch, batch_idx):
        output = self.forward(
            batch,
            masking=self.calc_gp_loss,
            masking_global=self.global_loss == 'masking',
        )

        if self.calc_gp_loss:
            loss_base = self.compute_gp_loss(batch, output)
            self.log_gp_loss(loss_base['loss_per_gp'])
        else:
            loss_base = {'total_loss': torch.tensor(0).to(self.device)}

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
            reconstruction_loss, dec_mean = self.compute_reconstruction_loss(
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

            # Compute cell-wise Pearson correlation (vectorized)
            true_counts = batch['counts']
            pred = dec_mean.detach().float()  # (batch_size, n_genes)
            target = true_counts.detach().float()  # (batch_size, n_genes)

            # Vectorized computation across all cells
            pred_mean = pred.mean(dim=1, keepdim=True)  # (batch_size, 1)
            target_mean = target.mean(dim=1, keepdim=True)  # (batch_size, 1)
            pred_centered = pred - pred_mean  # (batch_size, n_genes)
            target_centered = target - target_mean  # (batch_size, n_genes)

            numerator = (pred_centered * target_centered).sum(dim=1)  # (batch_size,)
            pred_std = torch.sqrt((pred_centered**2).sum(dim=1))  # (batch_size,)
            target_std = torch.sqrt((target_centered**2).sum(dim=1))  # (batch_size,)
            denominator = pred_std * target_std  # (batch_size,)

            # Handle zero denominator
            cell_corrs = numerator / denominator.clamp(min=1e-8)  # (batch_size,)
            cell_corrs = torch.where(
                denominator > 0, cell_corrs, torch.tensor(float('nan')).to(pred.device)
            )

            # Store correlations for this batch
            self.train_pearson_corrs.append(cell_corrs)

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
            # Compute mean of cell-wise Pearson correlations
            if len(self.train_pearson_corrs) > 0:
                all_corrs = torch.cat(self.train_pearson_corrs)

                if torch.isnan(all_corrs).any():
                    num_nan = torch.isnan(all_corrs).sum().item()
                    total = all_corrs.numel()
                    warnings.warn(
                        f'train pearson undefined for {num_nan}/{total} '
                        'cells due to zero variance in predictions or targets; '
                        'using nanmean.'
                    )
                mean_pearson_epoch = torch.nanmean(all_corrs.float())

                self.log(
                    'train/pearson',
                    mean_pearson_epoch,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

                # Reset for next epoch
                self.train_pearson_corrs = []

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
        output = self.forward(
            batch,
            masking=self.calc_gp_loss,
            masking_global=self.global_loss == 'masking',
        )

        # track_tissue = pd.Series(batch['tissue'])
        # print('Ground truth tissue', track_tissue.value_counts())

        if self.calc_gp_loss:
            loss_base = super().compute_gp_loss(batch, output)
        else:
            loss_base = {'total_loss': torch.tensor(0).to(self.device)}

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
            reconstruction_loss, dec_mean = self.compute_reconstruction_loss(
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

            # Compute cell-wise Pearson correlation (vectorized)
            true_counts = batch['counts']
            pred = dec_mean.detach().float()  # (batch_size, n_genes)
            target = true_counts.detach().float()  # (batch_size, n_genes)

            # Vectorized computation across all cells
            pred_mean = pred.mean(dim=1, keepdim=True)  # (batch_size, 1)
            target_mean = target.mean(dim=1, keepdim=True)  # (batch_size, 1)
            pred_centered = pred - pred_mean  # (batch_size, n_genes)
            target_centered = target - target_mean  # (batch_size, n_genes)

            numerator = (pred_centered * target_centered).sum(dim=1)  # (batch_size,)
            pred_std = torch.sqrt((pred_centered**2).sum(dim=1))  # (batch_size,)
            target_std = torch.sqrt((target_centered**2).sum(dim=1))  # (batch_size,)
            denominator = pred_std * target_std  # (batch_size,)

            # Handle zero denominator
            cell_corrs = numerator / denominator.clamp(min=1e-8)  # (batch_size,)
            cell_corrs = torch.where(
                denominator > 0, cell_corrs, torch.tensor(float('nan')).to(pred.device)
            )

            # Store correlations for this batch
            self.val_pearson_corrs.append(cell_corrs)

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
            # Compute mean of cell-wise Pearson correlations
            if len(self.val_pearson_corrs) > 0:
                all_corrs = torch.cat(self.val_pearson_corrs)

                if torch.isnan(all_corrs).any():
                    num_nan = torch.isnan(all_corrs).sum().item()
                    total = all_corrs.numel()
                    warnings.warn(
                        f'val pearson undefined for {num_nan}/{total} '
                        'cells due to zero variance in predictions or targets; '
                        'using nanmean.'
                    )
                mean_pearson = torch.nanmean(all_corrs.float())

                self.log(
                    'val/pearson',
                    mean_pearson,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

                # Reset for next epoch
                self.val_pearson_corrs = []

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

            output = self.forward(batch, masking=False, masking_global=False)

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
                    # flatten the batches and convert to numpy arrays
                    true_classes = torch.cat(self.cell_metadata[t]).cpu().numpy()

                    predicted_classes = np.array(meta_dict[f'{t}_pred_encoded'])
                    report = classification_report(
                        true_classes, predicted_classes, output_dict=True
                    )

                    output_df = wrangle_classification_report(report)

                    # convert encoded labels back to original labels
                    id_dict = {}
                    id_dict[t] = torch.cat(meta_dict[t]).cpu().numpy()
                    id_dict[t.replace('_id', '')] = [
                        item for l1 in meta_dict[t.replace('_id', '')] for item in l1
                    ]
                    meta_df = pd.DataFrame(id_dict)
                    meta_df = meta_df[[t, t.replace('_id', '')]].drop_duplicates()
                    label_mapping = {
                        str(k): v
                        for k, v in zip(
                            meta_df[t].values, meta_df[t.replace('_id', '')].values
                        )
                    }

                    output_df['label'] = (
                        output_df['output_class'].astype(str).map(label_mapping)
                    )

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

        if self.model.reconstruction_loss in ['nb', 'zinb']:
            # get re-scaled predictions
            pred_counts = output['count_output']['count_mean']
            batch_size_factor = torch.as_tensor(
                batch['size_factor'], device=pred_counts.device, dtype=pred_counts.dtype
            )

            dec_mean_gamma = output['count_output']['count_mean']
            size_factor_view = batch_size_factor.unsqueeze(1).expand(
                dec_mean_gamma.size(0), dec_mean_gamma.size(1)
            )
            dec_mean = dec_mean_gamma * size_factor_view

        return reconstruction_loss, dec_mean


class gpGlobalLoRA(gpGlobal):
    def __init__(self, lora_config_args, stage='train', **kwargs):
        # define global model
        super().__init__(**kwargs)
        self.modules_to_finetune = lora_config_args['modules_to_finetune']
        self.lora_config_dict = {
            k: v for k, v in lora_config_args.items() if k != 'modules_to_finetune'
        }

        self.setup_lora(stage=stage)

    def setup_lora(self, stage):
        gp_modules = []
        for n, p in self.model.named_parameters():
            if 'gf_wrapper' in self.modules_to_finetune:
                if 'attention.self.query' in n or 'attention.self.key' in n:
                    name = n.replace('.weight', '').replace('bias.', '')
                    gp_modules.append(name)

            if 'multi_gp_encoder' in self.modules_to_finetune:
                if 'attn.qkv' in n:
                    name = n.replace('.weight', '').replace('bias.', '')
                    gp_modules.append(name)

        lora_config = LoraConfig(
            r=self.lora_config_dict['r'],
            lora_alpha=self.lora_config_dict['lora_alpha'],
            target_modules=gp_modules,
            lora_dropout=self.lora_config_dict['lora_dropout'],
        )

        self.model = get_peft_model(self.model, lora_config)

        if stage == 'train':
            # Enable gradients global learner
            for name, param in self.model.named_parameters():
                if (
                    name == 'theta'
                    or 'cell_token_learner' in name
                    or 'clf_head' in name
                    or 'count_head' in name
                ):
                    param.requires_grad = True

            if 'multi_gp_encoder' not in self.modules_to_finetune:
                for n, p in self.model.named_parameters():
                    if 'multi_gp_encoder' in n:
                        p.requires_grad = False

            self.model.print_trainable_parameters()

    def state_dict(self):
        """Save model and LoRA adapter parameters."""
        base_state_dict = super().state_dict()
        # Save LoRA adapter parameters separately
        lora_state_dict = self.model.state_dict()  # Includes LoRA parameters
        base_state_dict.update(lora_state_dict)  # Merge dictionaries

        return base_state_dict

    def load_state_dict(self, state_dict, strict=False):
        """Load model and LoRA adapter parameters."""
        # # Extract LoRA configuration and modules to finetune
        # self.modules_to_finetune = state_dict['modules_to_finetune']
        # self.lora_config_args = state_dict['lora_config_args']

        # Separate LoRA adapter parameters
        lora_state_dict = {k: v for k, v in state_dict.items() if 'lora' in k}
        base_state_dict = {k: v for k, v in state_dict.items() if 'lora' not in k}

        # Load base model parameters
        super().load_state_dict(base_state_dict, strict=False)
        # Load LoRA adapter parameters
        self.model.load_state_dict(lora_state_dict, strict=False)


# ------------------------------------------------------
# Extra trainers
# ------------------------------------------------------


class gpAblation(gpGlobal):
    def __init__(self, compute_cosine=False, compute_delta_nb_loss=False, **kwargs):
        super().__init__(**kwargs)
        self.compute_cosine = compute_cosine
        self.compute_delta_nb_loss = compute_delta_nb_loss
        self.save_raw_embeddings = not (compute_cosine or compute_delta_nb_loss)
        self.cosine_adata = None
        self.delta_nb_loss_adata = None

    def build_perturbed_input_matrix(
        self, z, num_genes_per_cell_list, gp_pert_index=None
    ):
        z, gp_labels, attn_mask = self.model.cell_token_learner.build_input_matrix(
            z, num_genes_per_cell_list
        )

        # always mask the GP we are perturbing
        # set attn_mask to 0 when gp_label == gpert
        if gp_pert_index is not None:
            # handle cls separately
            attn_mask_cls = attn_mask[:, 0]
            attn_mask_gp = attn_mask[:, 1:]
            attn_mask_gp[gp_labels == gp_pert_index] = 0

            attn_mask = torch.cat([attn_mask_cls.unsqueeze(1), attn_mask_gp], dim=1)

            # and force the embedding to 0 just in case (?)
            z[gp_labels == gp_pert_index] = 0

        return z, gp_labels, attn_mask

    def compute_cell_level_reconstruction_loss(self, batch, output):
        """
        Compute reconstruction loss per cell (without taking mean across batch).
        This is needed for delta NB loss computation.
        """
        from ..Utils.losses import one_hot_encoder

        true_counts = batch['counts']
        batch_size_factor = torch.tensor(batch['size_factor']).to(true_counts.device)

        if self.model.reconstruction_loss == 'mse':
            loss_per_cell = F.mse_loss(
                output['count_output']['count_lognorm'], true_counts, reduction='none'
            ).sum(
                dim=-1
            )  # Sum over genes for each cell
            return loss_per_cell

        elif self.model.reconstruction_loss == 'zinb':
            dec_mean_gamma, dec_dropout = (
                output['count_output']['count_mean'],
                output['count_output']['count_dropout'],
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

            # Compute ZINB loss per cell (without mean)
            from ..Utils.losses import zinb

            loss_per_cell = -zinb(
                x=true_counts, mu=dec_mean, theta=dispersion, pi=dec_dropout
            ).sum(dim=-1)
            return loss_per_cell

        elif self.model.reconstruction_loss == 'nb':
            dec_mean_gamma = output['count_output']['count_mean']
            size_factor_view = batch_size_factor.unsqueeze(1).expand(
                dec_mean_gamma.size(0), dec_mean_gamma.size(1)
            )
            dec_mean = dec_mean_gamma * size_factor_view

            dispersion = F.linear(
                one_hot_encoder(batch['batch_key_id'], self.n_conditions_combined),
                self.theta,
            )
            dispersion = torch.exp(dispersion)

            # Compute NB loss per cell (without mean)
            from ..Utils.losses import nb

            loss_per_cell = -nb(x=true_counts, mu=dec_mean, theta=dispersion).sum(
                dim=-1
            )
            return loss_per_cell

        else:
            raise ValueError(
                f'Reconstruction loss {self.model.reconstruction_loss} not supported'
            )

    def test_step(self, batch, batch_idx):
        output = self.forward(batch, masking=False, masking_global=False)

        emb_dict = {}
        emb_dict['control'] = output['cell_token'].detach().cpu()

        if self.compute_cosine:
            cosine_dict = {}
            control_array = emb_dict['control'].numpy()

        if self.compute_delta_nb_loss:
            delta_nb_loss_dict = {}
            # Compute control reconstruction loss (cell-level)
            control_loss_per_cell = self.compute_cell_level_reconstruction_loss(
                batch, output
            )

        # Pertubations - zero out each GP
        for i, gp in enumerate(self.model.gp_inputs):
            z, gp_labels, attn_mask = self.build_perturbed_input_matrix(
                z=output['z'],
                num_genes_per_cell_list=output['num_genes_per_cell_list'],
                gp_pert_index=i,
            )

            encoder_output = self.model.cell_token_learner.encoder(
                z,
                gene_labels=gp_labels,
                attn_mask=attn_mask,
                masking=False,
                return_attention=False,
            )

            if self.compute_cosine:
                # get (1 - cosine similarity) to control embedding
                gp_array = encoder_output['cls'].detach().cpu().numpy()
                cos_sim = 1 - np.diag(cosine_similarity(control_array, gp_array))

                cosine_dict[gp] = cos_sim

            elif self.compute_delta_nb_loss:
                # Create perturbed output for loss computation
                perturbed_output = output.copy()
                perturbed_output['cell_token'] = encoder_output['cls']

                # CRITICAL: Re-compute count_output with the new cell token
                perturbed_output['count_output'] = self.model.count_head(
                    encoder_output['cls']
                )

                # Compute perturbed reconstruction loss (cell-level)
                perturbed_loss_per_cell = self.compute_cell_level_reconstruction_loss(
                    batch, perturbed_output
                )

                # Calculate delta loss (perturbed - control) per cell
                delta_loss_per_cell = perturbed_loss_per_cell - control_loss_per_cell
                delta_nb_loss_dict[gp] = delta_loss_per_cell.detach().cpu().numpy()

            else:
                emb_dict[f'{gp}_perturb'] = encoder_output['cls'].detach().cpu()

        # for outputting raw embeddings
        if self.save_raw_embeddings:
            # metadata
            for k, v in batch.items():
                if k != 'input_ids':
                    emb_dict[k] = v

            emb = Dataset.from_dict(emb_dict)

            if self.emb_dataset is None:
                self.emb_dataset = emb
            else:
                self.emb_dataset = concatenate_datasets([self.emb_dataset, emb])

        elif self.compute_cosine:
            meta_dict = {}

            for k, v in batch.items():
                if k != 'input_ids':
                    if isinstance(v, torch.Tensor):
                        meta_dict[k] = v.cpu().numpy()
                    else:
                        meta_dict[k] = v

            adata = sc.AnnData(
                X=pd.DataFrame(cosine_dict).values,
                obs=pd.DataFrame(meta_dict),
                var=pd.DataFrame(index=cosine_dict.keys()),
            )

            if self.cosine_adata is None:
                self.cosine_adata = adata
            else:
                self.cosine_adata = ad.concat([self.cosine_adata, adata])

        elif self.compute_delta_nb_loss:
            meta_dict = {}

            for k, v in batch.items():
                if (k != 'input_ids') and (k != 'counts'):
                    if isinstance(v, torch.Tensor):
                        meta_dict[k] = v.cpu().numpy()
                    else:
                        meta_dict[k] = v

            # Create DataFrame with cell-level delta losses
            delta_df = pd.DataFrame(
                delta_nb_loss_dict
            )  # Each column is a GP, each row is a cell

            adata = sc.AnnData(
                X=delta_df.values,
                obs=pd.DataFrame(meta_dict),
                var=pd.DataFrame(index=delta_nb_loss_dict.keys()),
            )

            if self.delta_nb_loss_adata is None:
                self.delta_nb_loss_adata = adata
            else:
                self.delta_nb_loss_adata = ad.concat([self.delta_nb_loss_adata, adata])

        return None

    def on_test_epoch_end(self):
        output_path = os.path.join(self.output_dir, 'with_gp_ablation')
        os.makedirs(output_path, exist_ok=True)
        output_name = os.path.join(output_path, f'{self.split_label}_set')

        if self.save_raw_embeddings:
            self.emb_dataset.save_to_disk(output_name)
            self.emb_dataset = None

        elif self.compute_cosine:
            self.cosine_adata.write_h5ad(output_name + '.h5ad')
            self.cosine_adata = None

        elif self.compute_delta_nb_loss:
            self.delta_nb_loss_adata.write_h5ad(output_name + '_delta_nb_loss.h5ad')
            self.delta_nb_loss_adata = None

        return None


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
            pearson = torch.corrcoef(pred.T, true.T)[0, 1]
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
