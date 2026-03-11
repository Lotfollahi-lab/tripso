import datetime
import os
import random
import uuid
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
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from transformers import BertConfig

from ..Datamodules.datamodule import AnnDataset, txDataModule
from ..Models.baselines import gfGlobal
from ..Models.gp_model import gpTransformerBase, gpTransformerGlobal
from ..Trainers.trainer import (
    gpBase,
    gpGlobal,
    gpGlobalLoRA,
)
from ..Utils.geneformer_utils import get_gf_repo
from ..Utils.utils import find_latest_file


def run_training(
    dataset_path: str,
    gpdb_path: str,
    output_dir: str,
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
    strategy: str = 'ddp_find_unused_parameters_true',
    attn_dropout: float = 0.0,
    lr: float = 1e-3,
    resume_training: Optional[bool] = False,
    gp_inputs: Optional[list] = None,
    frac_for_training: Optional[float] = 1.0,
    global_loss: str = 'supervised',
    classification_labels: Optional[list] = None,
    global_attn_heads: Optional[int] = 8,
    supervised_labels: Optional[dict] = None,
    global_masking_rate: Optional[float] = 0.15,
    global_attn_dropout: Optional[float] = 0.0,
    global_training: str = 'simultaneous',
    path_to_base_model: Optional[str] = None,  # 'path/to/pretrained/model',
    learn_new_gp: Optional[bool] = False,
    gp_to_learn: list = ['novel_gp'],
    global_n_blocks: int = 1,
    reconstruction_loss: Optional[str] = 'nb',
    adata_path: Optional[str] = None,
    use_flash: Optional[bool] = False,
    weight_decay: float = 0.0,
    sampler: Optional[str] = None,
    sample_by: Optional[str] = None,
    fm_encoder_name: str = 'gf-6L-30M-i2048',
    fm_encoder_pkg: str = 'geneformer',
    peft_config_path: Optional[str] = None,
    seed: Optional[int] = 0,
    data_seed: Optional[int] = None,
    supervised_rem_var: Optional[str] = None,
    num_nodes: int = 1,
    prbm_path: Optional[str] = None,
    use_l2_norm: Optional[bool] = False,
    gp_latent_size: Optional[int] = None,
    all_genes: Optional[list] = None,
    init_sparsity: Optional[float] = 0.0,
    # for large scale pretraining:
    limit_train_batches: Optional[float] = 1.0,
    limit_val_batches: Optional[float] = 1.0,
    val_check_interval: Optional[float] = 1.0,
    use_pos_emb: Optional[str] = 'sin_cos',
    global_pos_emb: Optional[str] = 'sin_cos',
    vocab_gene_names: Optional[list] = None,
    precision=32,  # 'bf16-mixed',
    bert_config: Dict = {},
    use_gene_embeddings: Optional[bool] = False,
    calc_gp_loss: Optional[bool] = True,
    calc_gene_loss: Optional[bool] = True,
    lora_config_args: Optional[dict] = None,
    warmup: Optional[int] = 0,
    accumulate_grad_batches: Optional[int] = 1,
):
    """
    Wrapper function for training Tripso model

    Parameters
    ----------
    dataset_path : str
        Path to input tokenized dataset
    gpdb_path : str
        Path to input gp database, a pandas csv where each column is a GP,
        with GP names as column names
    output_dir : str
        Directory where checkpoints and results will be saved
    batch_size : int, default=32
        Batch size for training
    mgm : float, default=0.15
        Masking ratio for masked gene modeling
        ie what proportion of genes to mask during training
    tissue : Optional[str], default=None
        Tissue name for logging experiment in wandb
    n_heads : int, default=8
        Number of heads for multi-head attention in GP encoder
    n_blocks : int, default=1
        Number of transformer blocks in GP encoder
    lr_scheduler : Literal['CosineLRwithWarmUp', 'ReduceLROnPlateau'],
        default='ReduceLROnPlateau'
        Learning rate scheduler for optimizer
    n_epochs : int, default=20
        Number of epochs to train for
    gene_format : Literal['symbol', 'ensembl'], default='symbol'
        Format in which gene names are stored in GPDB
    model_type : str, default='Base'
        One of 'Base', 'Global', 'Global_LoRA', or 'Mean'.
        'Base' trains only the GP encoder. 'Global' adds a cell-level
        transformer. 'Global_LoRA' uses LoRA for parameter-efficient training.
        'Mean' uses gene program mean embeddings.
    strategy : str, default='ddp_find_unused_parameters_true'
        Strategy for multi-GPU PyTorch Lightning trainer
    attn_dropout : float, default=0.0
        Dropout rate for attention layers
    lr : float, default=1e-3
        Learning rate for optimizer
    resume_training : Optional[bool], default=False
        Set to True to resume training from checkpoint
    gp_inputs : Optional[list], default=None
        List of GP names from GPDB to include in model. If None, uses all GP
    frac_for_training : Optional[float], default=1.0
        Fraction of the dataset to use for training (for development/testing)
    global_loss : str, default='supervised'
        Loss function for global model: 'supervised', 'masking', or 'reconstruction'
    classification_labels : Optional[list], default=None
        List of labels for supervised classification (deprecated, use supervised_labels)
    global_attn_heads : Optional[int], default=8
        Number of attention heads for learning cell token in global model
    supervised_labels : Optional[dict], default=None
        Dict mapping label names to number of classes for supervised classification
    global_masking_rate : Optional[float], default=0.15
        Masking rate for global model when using masking loss
    global_attn_dropout : Optional[float], default=0.0
        Dropout rate for attention layers in global model
    global_training : str, default='simultaneous'
        Training mode: 'simultaneous', 'sequential', 'finetune', 'finetune_global',
        or 'finetune_gene_encoder'. Controls how base and global models are trained
    path_to_base_model : Optional[str], default=None
        Path to pre-trained model checkpoint for sequential/finetuning training
    learn_new_gp : Optional[bool], default=False
        If True, load pretrained model, freeze most parameters, and learn new GP
    gp_to_learn : list, default=['novel_gp']
        List of GP names to learn when learn_new_gp is True
    global_n_blocks : int, default=1
        Number of transformer blocks in global model
    reconstruction_loss : Optional[str], default='nb'
        Loss function for reconstruction: 'nb' (negative binomial) or 'mse'
    adata_path : Optional[str], default=None
        Path to AnnData object with gene expression
        required for reconstruction loss
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
    fm_encoder_name : str, default='gf-6L-30M-i2048'
        Name of foundation model encoder to use
    fm_encoder_pkg : str, default='geneformer'
        Package for foundation model encoder: 'geneformer' or 'from_scratch'
    peft_config_path : Optional[str], default=None
        Path to PEFT (Parameter-Efficient Fine-Tuning) configuration file
    seed : Optional[int], default=0
        Random seed for reproducibility
    data_seed : Optional[int], default=None
        Random seed for data loading. If None, uses same as seed
    supervised_rem_var : Optional[str], default=None
        Variable to remove from supervised labels (currently unused)
    num_nodes : int, default=1
        Number of nodes for distributed training
    prbm_path : Optional[str], default=None
        Path to PRBM model (currently unused)
    use_l2_norm : Optional[bool], default=False
        Whether to use L2 normalization in model
    gp_latent_size : Optional[int], default=None
        Size of GP latent representation. If None, uses default from model
    all_genes : Optional[list], default=None
        List of all genes to consider. If provided, masks GP genes in gene encoder
    init_sparsity : Optional[float], default=0.0
        Initial sparsity level for sparse models
    limit_train_batches : Optional[float], default=1.0
        Fraction or number of training batches to use per epoch
    limit_val_batches : Optional[float], default=1.0
        Fraction or number of validation batches to use
    val_check_interval : Optional[float], default=1.0
        How often to check validation set. Float for fraction of epoch,
        int for number of batches
    use_pos_emb : Optional[str], default='sin_cos'
        Type of positional embedding for gene encoder
    global_pos_emb : Optional[str], default='sin_cos'
        Type of positional embedding for global model
    vocab_gene_names : Optional[list], default=None
        List of gene names in vocabulary for one-hot encoding
    precision : int or str, default=32
        Training precision: 32, 16, or 'bf16-mixed'
    bert_config : Dict, default={}
        Configuration dict for BERT model when training from scratch
    use_gene_embeddings : Optional[bool], default=False
        Model name (e.g., 'gf-12L-95M-i4096') or path to gene embeddings file,
        or False to initialize randomly
    calc_gp_loss : Optional[bool], default=True
        Whether to calculate GP prediction loss
    calc_gene_loss : Optional[bool], default=True
        Whether to calculate gene-level loss
    lora_config_args : Optional[dict], default=None
        Configuration arguments for LoRA when using Global_LoRA model
    warmup : Optional[int], default=0
        Number of warmup steps for learning rate scheduler
    accumulate_grad_batches : Optional[int], default=1
        Number of batches to accumulate gradients over before updating weights

    """

    ##########################################
    # Setup
    ##########################################

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

    torch.set_float32_matmul_precision('medium')

    args = locals()

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
        output_dir=output_dir,
    )

    # Load gpdb
    gpdb = pd.read_csv(gpdb_path)
    args['gpdb'] = gpdb

    # --------------------------------------------------
    # Other arguments for set up
    # --------------------------------------------------

    if ('Global' in model_type) & (global_loss == 'reconstruction'):
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

    if (reconstruction_loss == 'mse') & ('Global' in model_type):
        warnings.warn(
            'Using MSE loss for reconstruction'
            '\nMake sure you pass anndata object with log normalized counts'
        )

    ############################################################################
    # Train model
    ############################################################################

    model = configure_model(args)

    pl_model = configure_lightning_module(model, args)

    # Optionally load pretrained model
    if resume_training:
        pl_model = load_from_ckpt('resume_training', pl_model, args)
    elif global_training == 'sequential':
        pl_model = load_from_ckpt('sequential', pl_model, args)
    elif (global_training == 'finetune') | (global_training == 'finetune_global'):
        pl_model = load_from_ckpt('finetune', pl_model, args)
    elif global_training == 'finetune_gene_encoder':
        pl_model = load_from_ckpt('finetune_gene_encoder', pl_model, args)

    if learn_new_gp:
        pl_model = load_from_ckpt('learn_new_gp', pl_model, args)

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
        # ='advanced',
        num_nodes=num_nodes,
        strategy=strategy,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    # Train the model
    trainer.fit(pl_model, txdata)

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


def configure_save_id(args) -> str:
    """Define a unique ID to use in wandb logging and saving checkpoints.

    Parameters
    ----------
    args : dict
        Arguments dictionary with 'tissue' and 'model_type' keys.

    Returns
    -------
    str
        Unique save ID in format: YYYY-MM-DD_gp_transformer_<tissue>_<model_type>_<id>.
    """
    tissue = args['tissue']
    supervised_tag = args['model_type']

    wandb.login()

    # get date for today in YYYY-MM-DD format
    today = datetime.datetime.today().strftime('%Y-%m-%d')

    # create unique id for wandb run with 3 random characters
    unique_id = str(uuid.uuid4())[:3]

    save_id = f'{today}_gp_transformer_{tissue}_{supervised_tag}_{unique_id}'

    return save_id


@rank_zero_only
def configure_wandb(args, save_id) -> None:
    """Initialize Weights & Biases logging.

    Only runs on rank 0 in distributed training.

    Parameters
    ----------
    args : dict
        Arguments dictionary with 'output_dir' key.
    save_id : str
        Unique experiment identifier.
    """
    output_dir = args['output_dir']
    wandb_dir = os.path.join(output_dir, 'wandb_logs')

    # Check if directory exists
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    wandb.init(project='scGPL', id=save_id, dir=wandb_dir)


def configure_callbacks(save_id, args):
    """Configure PyTorch Lightning callbacks for training.

    Parameters
    ----------
    save_id : str
        Unique experiment identifier.
    args : dict
        Arguments dictionary with 'model_type', 'global_loss', 'output_dir',
        and other training configuration.

    Returns
    -------
    tuple
        (early_stopping_callback, checkpoint_callback, lr_monitor)
    """
    model_type = args['model_type']
    global_loss = args['global_loss']
    output_dir = args['output_dir']

    if (model_type == 'Global') | (model_type == 'Global_LoRA'):
        if global_loss == 'supervised':
            early_stopping_callback = EarlyStopping(
                monitor='val/accuracy',
                patience=3,
                mode='max',
            )
        elif global_loss == 'reconstruction':
            early_stopping_callback = EarlyStopping(
                monitor='val/pearson',
                patience=3,
                mode='max',
            )
        elif global_loss == 'masking':
            early_stopping_callback = EarlyStopping(
                monitor='val/loss',
                patience=3,
                mode='min',
            )
    elif model_type == 'Base':
        early_stopping_callback = EarlyStopping(
            monitor='train/loss_step',
            patience=50,
            mode='min',
        )

    # Define a directory for checkpoints within the output directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    # Make sure the directory exists, create it if not
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # monitor='val/loss',
        monitor='train/loss_step',
        dirpath=checkpoint_dir,
        filename=save_id,
        save_top_k=1,  # figure out how to get best checkpoint
        mode='min',
        save_last=True,
        # save every n steps --> issue if dataset has < n steps
        every_n_train_steps=500,  # 1_000,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    return early_stopping_callback, checkpoint_callback, lr_monitor


@rank_zero_only
def configure_logger(args):
    """Configure Weights & Biases logger for training.

    Only runs on rank 0 in distributed training.

    Parameters
    ----------
    args : dict
        Arguments dictionary with training configuration.

    Returns
    -------
    WandbLogger
        Configured Weights & Biases logger.
    """
    # create a logger to log training progress
    wandb_logger = WandbLogger(log_model=True)

    wandb_logger.experiment.config.update(
        {
            'dataset': args['dataset_path'].split('/')[-3],
            'supervise': args['model_type'],
            'architecture': 'gp_transformer',
            'epochs': args['n_epochs'],
            'mgm': args['mgm'],
            'n_heads': args['n_heads'],
            'n_blocks': args['n_blocks'],
            'lr_scheduler': args['lr_scheduler'],
            'batch_size': args['batch_size'],
            'strategy': args['strategy'],
            'attn_dropout': args['attn_dropout'],
            'transformer_block': 'preLN',
            'learning_rate': args['lr'],
            'frac_for_training': args['frac_for_training'],
            'use_flash': args['use_flash'],
            'weight_decay': args['weight_decay'],
            'use_pos_emb': args['use_pos_emb'],
            'precision': args['precision'],
            'fm_encoder_name': args['fm_encoder_name'],
            'fm_encoder_pkg': args['fm_encoder_pkg'],
            'bert_config': args['bert_config'],
            'use_gene_embeddings': args['use_gene_embeddings'],
            'gp_latent_size': args['gp_latent_size'],
            'mask_gp_genes_in_gene_encoder': isinstance(args['all_genes'], list),
            'sampling': 'random' if args['sampler'] is None else args['sampler'],
            'seed': args['seed'],
            'data_seed': args['data_seed'],
            'accumulate_grad_batches': args['accumulate_grad_batches'],
        }
    )

    if args['model_type'] == 'Global':
        wandb_logger.experiment.config.update(
            {
                'global_attn_heads': args['global_attn_heads'],
                'global_loss': args['global_loss'],
                'global_training': args['global_training'],
                'global_n_blocks': args['global_n_blocks'],
                'global_pos_emb': args['global_pos_emb'],
                'global_attn_dropout': args['global_attn_dropout'],
            }
        )

        if args['global_loss'] == 'supervised':
            wandb_logger.experiment.config.update(
                {
                    'classification_labels': args['classification_labels'],
                }
            )

        if args['global_loss'] == 'masking':
            wandb_logger.experiment.config.update(
                {
                    'global_masking_rate': args['global_masking_rate'],
                    'use_l2_norm_main': args['use_l2_norm'],
                    'warmup': args['warmup'],
                    'init_sparsity': args['init_sparsity'],
                }
            )

        if args['global_loss'] == 'reconstruction':
            wandb_logger.experiment.config.update(
                {
                    'reconstruction_loss': args['reconstruction_loss'],
                }
            )

    return wandb_logger


def configure_model(args):
    """Configure model based on arguments.

    Parameters
    ----------
    args : dict
        Arguments dictionary with model configuration including 'model_type',
        'gpdb', and other model parameters.

    Returns
    -------
    nn.Module
        Configured model (gpTransformerBase, gpTransformerGlobal, or gfGlobal).
    """
    common_params = {
        'database': args['gpdb'],
        'n_blocks': args['n_blocks'],
        'mgm_mask_ratio': args['mgm'],
        'num_heads': args['n_heads'],
        'attn_dropout': args['attn_dropout'],
        'gp_inputs': args['gp_inputs'],
        'use_flash': args['use_flash'],
        'learn_new_gp': args['learn_new_gp'],
        'peft_config_path': args['peft_config_path'],
        'use_pos_emb': args['use_pos_emb'],
        'vocab_gene_names': args['vocab_gene_names'],
        'do_ensembl_conversion': args['gene_format'] == 'symbol',
        'fm_encoder_name': args['fm_encoder_name'],
        'fm_encoder_pkg': args['fm_encoder_pkg'],
        'bert_config': args['bert_config'],
        'use_gene_embeddings': args['use_gene_embeddings'],
        'use_l2_norm': args['use_l2_norm'],
        'gp_latent_size': args['gp_latent_size'],
        'all_genes': args['all_genes'],
        'warmup': args['warmup'],
        'init_sparsity': args['init_sparsity'],
    }

    global_params = {
        'supervised_labels': args['supervised_labels'],
        'global_attn_heads': args['global_attn_heads'],
        'global_loss': args['global_loss'],
        'global_masking_rate': args['global_masking_rate'],
        'global_n_blocks': args['global_n_blocks'],
        'reconstruction_loss': args['reconstruction_loss'],
        'total_n_genes': args['total_n_genes'],
        'global_pos_emb': args['global_pos_emb'],
        'global_attn_dropout': args['global_attn_dropout'],
    }

    if args['model_type'] == 'Base':
        model = gpTransformerBase(**common_params)
        return model

    if (args['model_type'] == 'Global') | (args['model_type'] == 'Global_LoRA'):
        model = gpTransformerGlobal(**common_params, **global_params)
        return model

    if args['model_type'] == 'Mean':
        model = gfGlobal(**common_params)
        return model


def configure_lightning_module(model, args):
    """Configure PyTorch Lightning module.
    This is where architecture arguments are passed to the model.

    Parameters
    ----------
    model : nn.Module
        Base model to wrap.
    args : dict
        Arguments dictionary with training configuration.

    Returns
    -------
    pl.LightningModule
        Lightning module wrapper (gpBase or gpGlobal).
    """
    common_params = {
        'model': model,
        # 'model_type': args['model_type'],
        'lr': args['lr'],
        'total_epochs': args['n_epochs'],
        'lr_scheduler': args['lr_scheduler'],
        'output_dir': args['output_dir'],
        'weight_decay': args['weight_decay'],
        'optimizer': torch.optim.AdamW,  # DeepSpeedCPUAdam
        # if args['strategy'].startswith('deepspeed')
        # else
        'calc_gp_loss': args['calc_gp_loss'],
        'calc_gene_loss': args['calc_gene_loss'],
        'warmup': args['warmup'],
    }

    global_params = {
        'n_condition_combined': args['n_condition_combined'],
        'total_n_genes': args['total_n_genes'],
        'global_loss': args['global_loss'],
    }

    if args['model_type'] == 'Base':
        pl_model = gpBase(**common_params)
        return pl_model

    if args['model_type'] == 'Global':
        pl_model = gpGlobal(**common_params, **global_params)
        return pl_model

    if args['model_type'] == 'Global_LoRA':
        pl_model = gpGlobalLoRA(
            lora_config_args=args['lora_config_args'],
            **common_params,
            **global_params,
        )

        return pl_model


def load_from_ckpt(mode, pl_model, args):
    """Load model weights from checkpoint.

    Parameters
    ----------
    mode : {'resume_training', 'sequential', 'finetune'}
        Loading mode determining which weights to load and freeze.
        - 'resume_training': Resume training from last checkpoint
        - 'sequential': Load pretrained model for sequential training
            in this case, existing weights are loaded and base model
            is frozen
        - 'finetune': Load model for fine-tuning
    pl_model : pl.LightningModule
        Lightning module to load weights into.
    args : dict
        Arguments dictionary with checkpoint path information including
        'output_dir', 'tissue', 'model_type', and 'path_to_base_model'.

    Returns
    -------
    pl.LightningModule
        Model with loaded weights.
    """

    output_dir = args['output_dir']
    tissue = args['tissue']
    model_type = args['model_type']
    if args['path_to_base_model'] is None:
        path_to_base_model = output_dir
    else:
        path_to_base_model = args['path_to_base_model']

    if mode == 'resume_training':
        # latest_ckpt = find_latest_file(output_dir, tissue, model_type)
        latest_ckpt = os.path.join(path_to_base_model, 'checkpoints/last.ckpt')

        if model_type == 'Global':
            pl_model = gpGlobal.load_from_checkpoint(latest_ckpt, map_location='cpu')
        else:
            pl_model = gpBase.load_from_checkpoint(latest_ckpt, map_location='cpu')

        return pl_model

    elif mode == 'sequential':
        # latest_ckpt = find_latest_file(path_to_base_model, tissue, 'Base')
        # checkpoint_path = os.path.join(path_to_base_model, latest_ckpt)
        latest_ckpt = os.path.join(path_to_base_model, 'checkpoints/last.ckpt')
        checkpoint = torch.load(
            latest_ckpt, map_location=torch.device('cpu'), weights_only=False
        )
        pl_model.load_state_dict(checkpoint['state_dict'], strict=False)

        # freeze base model
        for name, param in pl_model.model.named_parameters():
            if (
                ('cell_token_learner' in name)
                | ('clf_head' in name)
                | ('count_head' in name)
                | ('theta' in name)
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

        return pl_model

    elif mode == 'finetune_gene_encoder':
        # latest_ckpt = find_latest_file(path_to_base_model, tissue, 'Base')
        # checkpoint_path = os.path.join(path_to_base_model, latest_ckpt)
        latest_ckpt = os.path.join(path_to_base_model, 'checkpoints/last.ckpt')

        checkpoint = torch.load(
            latest_ckpt, map_location=torch.device('cpu'), weights_only=False
        )
        pl_model.load_state_dict(checkpoint['state_dict'], strict=False)

        # freeze base model
        for name, param in pl_model.model.named_parameters():
            if (
                ('cell_token_learner' in name)
                | ('clf_head' in name)
                | ('count_head' in name)
                | ('gf_wrapper' in name)
                | ('theta' in name)
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

        return pl_model

    elif (mode == 'finetune') | (mode == 'finetune_global'):
        # try:
        #     latest_ckpt = find_latest_file(path_to_base_model, tissue, 'Global')
        # except FileNotFoundError:
        #     latest_ckpt = find_latest_file(path_to_base_model, tissue, 'Base')

        warnings.warn(
            'Finetuning global model from base model'
            'finetune_lr parameter is deprecated.'
            'Please specify learning rate for base model as'
            '{"gene_encoder": lr, "multi_gp_encoder": lr, "default": lr}'
        )

        latest_ckpt = os.path.join(path_to_base_model, 'checkpoints/last.ckpt')

        checkpoint = torch.load(
            latest_ckpt, map_location=torch.device('cpu'), weights_only=False
        )
        pl_model.load_state_dict(checkpoint['state_dict'], strict=False)

        return pl_model

    elif mode == 'learn_new_gp':
        checkpoint_path = find_latest_file(path_to_base_model, tissue, model_type)
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device('cpu'), weights_only=False
        )
        pl_model.load_state_dict(checkpoint['state_dict'], strict=False)
        pl_model.output_dir = output_dir

        # get indices of GP to learn
        gp_to_learn = args['gp_to_learn']

        if isinstance(gp_to_learn, str):
            gp_to_learn = [gp_to_learn]
        gp_idx = [
            pl_model.model.gp_inputs.index(gp)
            for gp in gp_to_learn
            if gp in pl_model.model.gp_inputs
        ]

        # freeze all GP
        for name, param in pl_model.model.named_parameters():
            if 'multi_gp_encoder' in name:
                param.requires_grad = False

        # unfreeze new GP
        for i in gp_idx:
            for name, param in pl_model.model.named_parameters():
                if f'multi_gp_encoder.encoder.{i}' in name:
                    param.requires_grad = True

        return pl_model
