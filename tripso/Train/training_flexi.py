import os
import random
import warnings
from typing import (
    Dict,
    Literal,
    Optional,
)

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

# set up wandb
import wandb

# from deepspeed.ops.adam import DeepSpeedCPUAdam
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_only
from transformers import BertConfig

from ..Datamodules.datamodule import AnnDataset, txDataModule
from ..Models.baselines import gfGlobal
from ..Models.gp_model import gpTransformerBase, gpTransformerGlobal
from ..Trainers.trainer import gpBase, gpGlobal
from ..Utils.geneformer_utils import get_gf_repo
from ..Utils.utils import find_latest_file
from .training import (
    configure_callbacks,
    configure_logger,
    configure_save_id,
    configure_wandb,
)


def run_training_from_select_gps(
    dataset_path: str,
    gpdb_path: str,
    output_dir: str,
    gpdb_old: Optional[str] = None,
    batch_size: int = 32,
    mgm: float = 0.15,
    tissue: Optional[str] = None,
    n_heads: int = 8,
    n_blocks: int = 1,
    lr_scheduler: Literal[
        'CosineLRwithWarmUp', 'ReduceLROnPlateau'
    ] = 'ReduceLROnPlateau',
    n_epochs: int = 20,
    gene_format: Literal['symbol', 'ensembl'] = 'symbol',
    model_type: str = 'Base',
    model_type_old: str = 'Base',
    strategy: str = 'ddp_find_unused_parameters_true',
    attn_dropout: float = 0.0,
    lr: float = 1e-3,
    gp_inputs_old: Optional[list] = None,
    gp_inputs_new: Optional[list] = None,
    frac_for_training: Optional[float] = 1.0,
    global_loss: str = 'supervised',
    global_loss_old: str = 'reconstruction',
    classification_labels: Optional[list] = None,
    global_attn_heads: Optional[int] = 8,
    supervised_labels: Optional[dict] = None,
    supervised_labels_old: Optional[dict] = None,
    global_masking_rate: Optional[float] = 0.15,
    global_pos_emb: Optional[str] = 'sin_cos',
    global_attn_dropout: Optional[float] = 0.0,
    global_training: str = 'simultaneous',
    path_to_base_model: str = 'path/to/pretrained/model',
    global_n_blocks: int = 1,
    reconstruction_loss: Optional[str] = 'nb',
    adata_path: Optional[str] = None,
    use_flash: Optional[bool] = False,
    weight_decay: float = 0.0,
    sampler: Optional[str] = None,
    sample_by: Optional[str] = None,
    fm_encoder_pkg: str = 'geneformer',
    fm_encoder_name: str = 'gf-6L-30M-i2048',
    peft_config_path: Optional[str] = None,
    seed: Optional[int] = 0,
    data_seed: Optional[int] = None,
    set_gpfinder_weight_decay: Optional[float] = None,
    calc_gp_loss: bool = True,
    all_genes: Optional[list] = None,
    use_pos_emb: Optional[str] = 'sin_cos',
    vocab_gene_names: Optional[str] = None,
    num_nodes: int = 1,
    limit_train_batches: float = 1.0,
    limit_val_batches: float = 1.0,
    val_check_interval: float = 1.0,
    precision=32,  # 'bf16-mixed',
    bert_config: Dict = {},
    use_gene_embeddings: Optional[bool] = False,
    load_cell_token_learner: bool = False,
    gp_of_interest: Optional[str] = None,
    gp_for_downstream: Optional[str] = None,
    gp_latent_size: Optional[int] = None,
    accumulate_grad_batches: Optional[int] = 1,
    resume_training: bool = False,
):
    """
    Wrapper function for training Tripso model with flexible GP selection.
    This function allows training with separate old and new GP sets,
    this is useful for adding an individual GP or for GP discovery.

    Parameters
    ----------
    dataset_path : str
        Path to input tokenized dataset
    gpdb_path : str
        Path to gene program database (CSV with each column as a GP)
    output_dir : str
        Directory where experiment results will be saved
    gpdb_old : Optional[str], default=None
        Path to old/previous gene program database
        (GP which are already trained)
    batch_size : int, default=32
        Batch size for training
    mgm : float, default=0.15
        Masking ratio for masked gene modeling
    tissue : Optional[str], default=None
        Tissue name for logging experiment in wandb
    n_heads : int, default=8
        Number of attention heads in GP encoder
    n_blocks : int, default=1
        Number of transformer blocks in GP encoder
    lr_scheduler : Literal['CosineLRwithWarmUp', 'ReduceLROnPlateau'],
        default='ReduceLROnPlateau'
        Learning rate scheduler for optimizer
    n_epochs : int, default=20
        Number of training epochs
    gene_format : Literal['symbol', 'ensembl'], default='symbol'
        Format of gene names in GPDB
    model_type : str, default='Base'
        Model type for new GPs: 'Base', 'Global', 'Global_LoRA', or 'Mean'
    model_type_old : str, default='Base'
        Model type for old GPs: 'Base', 'Global', 'Global_LoRA', or 'Mean'
    strategy : str, default='ddp_find_unused_parameters_true'
        Multi-GPU strategy for PyTorch Lightning trainer
    attn_dropout : float, default=0.0
        Dropout rate for attention layers
    lr : float, default=1e-3
        Learning rate for optimizer
    gp_inputs_old : Optional[list], default=None
        List of GP names from old GPDB to include. If None, uses all
    gp_inputs_new : Optional[list], default=None
        List of GP names from new GPDB to include. If None, uses all
    frac_for_training : Optional[float], default=1.0
        Fraction of dataset to use for training (for development/testing)
    global_loss : str, default='supervised'
        Loss function for new global model: 'supervised', 'masking', or 'reconstruction'
    global_loss_old : str, default='reconstruction'
        Loss function for old global model: 'supervised', 'masking', or 'reconstruction'
    classification_labels : Optional[list], default=None
        List of labels for supervised classification (deprecated, use
        supervised_labels)
    global_attn_heads : Optional[int], default=8
        Number of attention heads for learning cell token in global
        model
    supervised_labels : Optional[dict], default=None
        Dict mapping label names to number of classes for new model
        supervised classification
    supervised_labels_old : Optional[dict], default=None
        Dict mapping label names to number of classes for old model
        supervised classification
    global_masking_rate : Optional[float], default=0.15
        Masking rate for global model when using masking loss
    global_pos_emb : Optional[str], default='sin_cos'
        Type of positional embedding for global model
    global_attn_dropout : Optional[float], default=0.0
        Dropout rate for attention layers in global model
    global_training : str, default='simultaneous'
        Training mode: 'simultaneous', 'sequential', 'finetune', or 'finetune_global'
        simultaneous : both old and new models are trained together
        sequential : old model is trained first, then new model (old model is frozen)
        finetune : previous GP blocks are finetuned along with new GP blocks
        finetune_global : only global model is finetuned after training old GPs
    path_to_base_model : str, default='path/to/pretrained/model'
        Path to pre-trained model checkpoint for sequential/finetuning training
    global_n_blocks : int, default=1
        Number of transformer blocks in global model
    reconstruction_loss : Optional[str], default='nb'
        Loss function for reconstruction: 'nb' (negative binomial),
        'zinb' (zero-inflated negative binomial), or 'mse'
        (mean squared error)
    adata_path : Optional[str], default=None
        Path to AnnData object, required for reconstruction loss
    use_flash : Optional[bool], default=False
        Whether to use flash attention in transformer blocks
    weight_decay : float, default=0.0
        Weight decay for optimizer
    sampler : Optional[str], default=None
        Sampling strategy for data loading
         Options: 'weighted' for WeightedRandomSampler,
         'length' for LengthGroupedSampler, or None.
    sample_by : Optional[str], default=None
        Column name in AnnData to sample by (used with sampler)
    fm_encoder_pkg : str, default='geneformer'
        Package for foundation model encoder: 'geneformer' or 'from_scratch'
    fm_encoder_name : str, default='gf-6L-30M-i2048'
        Name of foundation model encoder to use
    peft_config_path : Optional[str], default=None
        Path to PEFT (Parameter-Efficient Fine-Tuning) configuration file
    seed : Optional[int], default=0
        Random seed for reproducibility
    data_seed : Optional[int], default=None
        Random seed for data loading. If None, uses same as seed
    set_gpfinder_weight_decay : Optional[float], default=None
        Specific weight decay for GPFinder layers
    calc_gp_loss : bool, default=True
        Whether to calculate GP prediction loss
    all_genes : Optional[list], default=None
        List of all genes to consider. If provided, masks GP genes in gene encoder
    use_pos_emb : Optional[str], default='sin_cos'
        Type of positional embedding for gene encoder
    vocab_gene_names : Optional[str], default=None
        List of gene names in vocabulary for one-hot encoding
    num_nodes : int, default=1
        Number of nodes for distributed training
    limit_train_batches : float, default=1.0
        Fraction or number of training batches to use per epoch
    limit_val_batches : float, default=1.0
        Fraction or number of validation batches to use
    val_check_interval : float, default=1.0
        How often to check validation set. Float for fraction of epoch,
        int for number of batches
    precision : int or str, default=32
        Training precision: 32, 16, or 'bf16-mixed'
    bert_config : Dict, default={}
        Configuration dict for BERT model when training from scratch
    use_gene_embeddings : Optional[bool], default=False
        Model name (e.g., 'gf-12L-95M-i4096') or path to gene embeddings file,
        or False to initialize randomly
    load_cell_token_learner : bool, default=False
        Whether to load cell token learner from previous global training
    gp_of_interest : Optional[str], default=None
        Single GP to use exclusively in forward pass
        This is helpful if you only need to train/evaluate one GP
        rather than all of them.
    gp_for_downstream : Optional[str], default=None
        GP to focus on for downstream analysis
    gp_latent_size : Optional[int], default=None
        Size of GP latent representation. If None, uses default from model
    accumulate_grad_batches : Optional[int], default=1
        Number of batches to accumulate gradients over before updating weights
    resume_training : bool, default=False
        Set to True to resume training from checkpoint
    """
    ##########################################
    # Setup
    ##########################################

    # torch.set_float32_matmul_precision('medium')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if data_seed is None:
        data_seed = seed

    args = locals()

    # wandb.login()

    save_id = configure_save_id(args)

    early_stopping_callback, checkpoint_callback, lr_monitor = configure_callbacks(
        save_id, args
    )

    # initialize wandb logging
    configure_wandb(args, save_id)
    wandb_logger = configure_logger(args) or None

    ############################################################################
    # Dataset Preparation
    ############################################################################

    # Instantiate datamodule
    if fm_encoder_pkg == 'from_scratch':
        model_input_size = bert_config['tokenization_input_size']
    else:
        # Get Geneformer model config
        geneformer_repo_path = get_gf_repo()
        geneformer_model = os.path.join(
            geneformer_repo_path,
            fm_encoder_name,
        )

        gf_config = BertConfig.from_pretrained(geneformer_model)
        model_input_size = gf_config.max_position_embeddings

    txdata = txDataModule(
        folder=dataset_path,
        batch_size=batch_size,
        frac_for_training=frac_for_training,
        adata_path=adata_path,
        sampler=sampler,
        label_key=sample_by,
        seed=data_seed,
        model_input_size=model_input_size,
    )

    # Load gpdb
    gpdb = pd.read_csv(gpdb_path)
    args['gpdb_new'] = gpdb
    args['gpdb'] = gpdb

    gpdb_old = pd.read_csv(gpdb_old)
    args['gpdb_old'] = gpdb_old

    # --------------------------------------------------
    # Other arguments for set up
    # --------------------------------------------------

    if (model_type == 'Global') & (global_loss == 'reconstruction'):
        if adata_path is None:
            raise ValueError('Please provide path to anndata object')
        else:
            anndata_dataset = AnnDataset(adata_path)
            total_n_genes = anndata_dataset.get_n_genes()
            n_condition_combined = anndata_dataset.n_condition_combined

    else:
        total_n_genes = 0
        n_condition_combined = 1

    args['total_n_genes'] = total_n_genes
    args['n_condition_combined'] = n_condition_combined

    if (reconstruction_loss == 'mse') & (model_type == 'Global'):
        warnings.warn(
            'Using MSE loss for reconstruction'
            '\nMake sure you pass anndata object with log normalized counts'
        )

    ############################################################################
    # Train model
    ############################################################################

    model_v0 = configure_model_version(args, 'old')

    model_v1 = configure_model_version(args, 'new')

    gp_transformer_v0 = configure_lightning_module_version(model_v0, 'old', args)

    gp_transformer = configure_lightning_module_version(model_v1, 'new', args)

    # ----- Load pretrained model -------

    if resume_training:
        checkpoint_path = find_latest_file(output_dir, tissue, model_type)
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device('cpu'), weights_only=False
        )
        gp_transformer.load_state_dict(checkpoint['state_dict'])

    else:
        latest_ckpt = find_latest_file(path_to_base_model, tissue, model_type_old)
        checkpoint_path = os.path.join(path_to_base_model, latest_ckpt)
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device('cpu'), weights_only=False
        )
        state_dict = checkpoint['state_dict']

        model_state_dict = gp_transformer_v0.state_dict()
        if global_loss_old == 'reconstruction' and global_loss == 'supervised':
            irrelevant_params = [
                'theta',
                'model.count_head.softmax_output.0.weight',
                'model.count_head.softmax_output.0.bias',
            ]
            for param_name in state_dict:
                if param_name in irrelevant_params:
                    state_dict[param_name] = torch.zeros_like(
                        model_state_dict[param_name]
                    )

        # Remove params that are not in the new model
        state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

        gp_transformer_v0.load_state_dict(state_dict)

        # ----- Transfer weights of multi_gp_encoder -------
        for i, gp in enumerate(gp_transformer.model.gp_inputs):
            if gp in gp_transformer_v0.model.gp_inputs:
                # find index in original model
                idx = gp_transformer_v0.model.gp_inputs.index(gp)

                # transfer weights
                gp_transformer.model.multi_gp_encoder.encoder[
                    i
                ] = gp_transformer_v0.model.multi_gp_encoder.encoder[idx]

                # freeze weights for this block
                for name, param in gp_transformer.model.named_parameters():
                    if f'multi_gp_encoder.encoder.{i}' in name:
                        param.requires_grad = False
            else:
                continue

        # ----- Transfer weights of gf_wrapper -------
        gp_transformer.model.gf_wrapper = gp_transformer_v0.model.gf_wrapper

        # Freeze weights
        for name, param in gp_transformer.model.named_parameters():
            if 'gf_wrapper' in name:
                param.requires_grad = False

        # ----- Optionally transfer cell encoder -------
        if load_cell_token_learner:
            if (model_type == 'Global') & (model_type_old == 'Global'):
                gp_transformer.model.cell_token_learner = (
                    gp_transformer_v0.model.cell_token_learner
                )

                # # freeze cell encoder
                # for name, param in gp_transformer.model.named_parameters():
                #     if 'cell_encoder' in name:
                #         param.requires_grad = False

    # Lightning trainer
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            early_stopping_callback,
            checkpoint_callback,
            lr_monitor,
        ],
        logger=wandb_logger,
        devices=-1,
        accelerator='auto',
        precision=precision,
        # profiler='advanced',
        num_nodes=num_nodes,
        strategy=strategy,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
    )

    # Train the model
    trainer.fit(gp_transformer, txdata)

    # save logs to csv for custom plotting
    # Fetch logged data from wandb
    if rank_zero_only.rank == 0:
        api = wandb.Api()
        run = api.run(f'scGPL/{save_id}')

        # Get logged data as dataframe
        df = run.history()
        df.to_csv(f'{output_dir}/training_metrics.csv', index=False)

        wandb.finish()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------


def configure_model_version(args, tag):
    common_params = {
        'database': args[f'gpdb_{tag}'],
        'n_blocks': args['n_blocks'],
        'mgm_mask_ratio': args['mgm'],
        'num_heads': args['n_heads'],
        'attn_dropout': args['attn_dropout'],
        'gp_inputs': args[f'gp_inputs_{tag}'],
        'use_flash': args['use_flash'],
        'fm_encoder_pkg': args['fm_encoder_pkg'],
        'fm_encoder_name': args['fm_encoder_name'],
        'peft_config_path': args['peft_config_path'],
        'use_pos_emb': args['use_pos_emb'],
        'vocab_gene_names': args['vocab_gene_names'],
        'do_ensembl_conversion': args['gene_format'] == 'symbol',
        'bert_config': args['bert_config'],
        'all_genes': args['all_genes'],
        'gp_latent_size': args['gp_latent_size'],
        'use_gene_embeddings': args['use_gene_embeddings'],
    }

    global_params = {
        'global_attn_heads': args['global_attn_heads'],
        'global_masking_rate': args['global_masking_rate'],
        'global_n_blocks': args['global_n_blocks'],
        'reconstruction_loss': args['reconstruction_loss'],
        'total_n_genes': args['total_n_genes'],
    }

    if tag == 'old':
        global_params['supervised_labels'] = args['supervised_labels_old']
        global_params['global_loss'] = args[f'global_loss_{tag}']
        model_type = args['model_type_old']
    else:
        global_params['supervised_labels'] = args['supervised_labels']
        global_params['global_loss'] = args['global_loss']
        model_type = args['model_type']

    if model_type == 'Base':
        model = gpTransformerBase(**common_params)
        return model

    if model_type == 'Global':
        model = gpTransformerGlobal(**common_params, **global_params)
        return model

    if model_type == 'Mean':
        model = gfGlobal(**common_params)
        return model


def configure_lightning_module_version(model, tag, args):
    common_params = {
        'model': model,
        'lr': args['lr'],
        'total_epochs': args['n_epochs'],
        'lr_scheduler': args['lr_scheduler'],
        'output_dir': args['output_dir'],
        'weight_decay': args['weight_decay'],
        'set_gpfinder_weight_decay': args['set_gpfinder_weight_decay'],
        'optimizer': torch.optim.AdamW,
        # DeepSpeedCPUAdam
        # if args['strategy'].startswith('deepspeed')
        # else
        'gp': args['gp_of_interest'],
        'gp_for_downstream': args['gp_for_downstream'],
        'calc_gp_loss': args['calc_gp_loss'],
    }

    global_params = {
        'n_condition_combined': args['n_condition_combined'],
        'total_n_genes': args['total_n_genes'],
    }

    if tag == 'old':
        global_params['global_loss'] = args[f'global_loss_{tag}']
        model_type = args['model_type_old']
    else:
        global_params['global_loss'] = args['global_loss']
        model_type = args['model_type']

    if model_type == 'Base':
        pl_model = gpBase(**common_params)
        return pl_model

    if model_type == 'Global':
        pl_model = gpGlobal(**common_params, **global_params)
        return pl_model
