import datetime
import os
import random
import uuid
import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

# set up wandb
import wandb
from deepspeed.ops.adam import DeepSpeedCPUAdam
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from ..Datamodules.datamodule import AnnDataset, txDataModule
from ..Models.gp_model import (
    GENEFORMER_MODEL_PATH,
    gfGlobal,
    gpTransformerBase,
    gpTransformerBaseWithPrompt,
    gpTransformerGlobal,
    gpTransformerGlobalWithPrompt,
    gpTransformerPrototypes,
)
from ..Trainers.trainer import (
    gpBase,
    gpGlobal,
    gpPrototypes,
)
from ..Utils.utils import find_latest_file


def run_training(
    dataset_path: str,
    gpdb_path: str,
    output_dir: str,
    gp_similarity_file: Optional[str] = None,
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
    gp_latent_size: int = 256,
    attn_dropout: float = 0.0,
    lr: float = 1e-3,
    finetune_lr: float = 1e-5,
    resume_training: Optional[bool] = False,
    gene_counts_df: Optional[str] = None,
    gp_inputs: Optional[list] = None,
    frac_for_training: Optional[float] = 1.0,
    lambda_gp_similarity: Optional[float] = 1e-2,
    global_loss: str = 'supervised',
    classification_labels: Optional[list] = None,
    global_attn_heads: Optional[int] = 8,
    supervised_labels: Optional[dict] = None,
    global_masking_rate: Optional[float] = 0.15,
    global_training: str = 'simultaneous',
    path_to_base_model: Optional[str] = None,  # 'path/to/pretrained/model',
    learn_new_gp: Optional[bool] = False,
    gp_to_learn: list = ['novel_gp'],
    global_n_blocks: int = 1,
    reconstruction_loss: Optional[str] = 'nb',
    adata_path: Optional[str] = None,
    use_flash: Optional[bool] = False,
    weight_decay: float = 0.0,
    use_weighted_sampler: Optional[bool] = False,
    sample_by: Optional[str] = None,
    geneformer_model_path: Optional[str] = GENEFORMER_MODEL_PATH,
    peft_config_path: Optional[str] = None,
    seed: Optional[int] = 0,
    supervised_rem_var: Optional[str] = None,
    set_gpfinder_weight_decay: Optional[float] = None,
    num_virtual_tokens: int = 0,
    virtual_tokens_label: Optional[str] = None,
    num_prompt_classes: Optional[int] = 0,
    num_nodes: int = 1,
    num_prototypes: int = 0,
    prototype_labels_key: Optional[str] = None,
    lambda_prototype_loss: float = 1e-2,
    prbm_path: Optional[str] = None,
    use_baseline_tk: Optional[bool] = False,
    tk_vocab_size: Optional[int] = 0,
    # for large scale pretraining:
    limit_val_batches: Optional[float] = 1.0,
    val_check_interval: Optional[float] = 1.0,
    mean_emb_dict: Optional[str] = None,
    gene2vec: Optional[str] = None,
    use_pos_emb: Optional[bool] = True,
    use_onehot_wrapper: Optional[bool] = False,
    vocab_gene_names: Optional[list] = None,
):
    """
    Wrapper function for training gpLearner model

    Parameters
    ----------
    dataset_path : str
        path to input tokenized dataset
    gpdb_path : str
        path to input gp database, a pandas csv where each column is a GP,
        with GP names as column names
    gp_similarity_file : str
        path to input gp similarity file, a numpy array
        where x[i,j] is the similarity between GP i and GP j
    output_dir : str
        directory where we will dump our experiment's results.
        If not given, then we will use the directory given as
        the 'results_dir' in the config file.
    batch_size : int
        batch size
    mgm : float
        masking ratio for masked gene modeling ablation experiments
    tissue : str
        tissue name for logging experiment in wandb equivalent to
        directory name in examples subfolder
    n_heads : int
        number of heads for multi-head attention
    n_blocks : int
        number of transformer blocks
    lr_scheduler : str
        learning rate scheduler for optimizer
        nb this is a string which will be converted to a class
    n_epochs : int
        number of epochs to train for
    gene_format : str
        format in which gene names are stored in GPDB
    model_type : str
        One of Base, Supervised or Unsupervised Where unsupervised has an
        extra self-attention head to learn a cell token based on GP tokens
    strategy : str
        strategy for multi-GPU lightning trainer
    gp_latent_size : int
        size of latent space for GP tokens if <256,
        will use MLP to reduce dimensions of Geneformer gene embeddings
        else take embeddings directly
    attn_dropout : float
        Dropout for attention layers
        NB only for final self attention block for now
    lr : float
        Model trainer learning rate
    resume_training : bool
        Set to True to resume training from checkpoint
    gene_counts_df : str
        Dataframe with the counts of each gene in the dataset
    gp_inputs : list
        Which GP from GPDB to include in model if None, defaults to all GP
    frac_for_training : float
        fraction of the dataset to use for training - default is 1.0
        (development only)
    n_blocks : int
        number of transformer blocks
    lambda_gp_similarity : float
        weight for gp similarity loss
    global_loss : str
        loss function for global model
    classification_labels : list
        list of labels for supervised classification
    supervised_labels : list
        Dict {label : num_classes} for supervised classification
        TO DO: provide either classification or supervised labels / check compatibility
    global_attn_heads : int
        number of heads for learning cell token
    global_training : str
        can be 'simultaneous' or 'sequential'
        if 'sequential' will train global model after training base model
        if 'simultaneous' will train global model at the same time as base model
    path_to_base_model : str
        path to pre-trained gpTransformer Base model for sequential training
    learn_new_gp : bool
        if True, load pretrained gpTransformer model, freeze,
        and learn new gpTransformer block
    gp_to_learn : list
        list of GP to learn if learn_new_gp is True
    global_n_blocks : int
        number of transformer blocks for final transformer block
    use_flash:
        whether to use flash attention in transformer block

    limit_val_batches : float
        (Union[int, float, None]) How often to check the validation set.
        Pass a float in the range [0.0, 1.0] to check after a fraction of
        the training epoch. Pass an int to check after a fixed number of
        training batches. An int value can only be higher than the number
        of training batches when check_val_every_n_epoch=None, which
        validates after every N training batches across epochs or during
        iteration-based training. Default: 1.0.
    val_check_interval : float
        (Optional[int]) Perform a validation loop every after every N
        training epochs. If None, validation will be done solely based
        on the number of training batches, requiring val_check_interval
        to be an integer value. Default: 1.
        from https://github.com/EveryVoiceTTS/EveryVoice/issues/204

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

    # torch.set_float32_matmul_precision('medium')

    args = locals()

    save_id = configure_wandb(args)

    early_stopping_callback, checkpoint_callback, lr_monitor = configure_callbacks(
        save_id, args
    )

    wandb_logger = configure_logger(args)

    ############################################################################
    # Dataset Preparation
    ############################################################################

    # Instantiate datamodule

    txdata = txDataModule(
        folder=dataset_path,
        batch_size=batch_size,
        frac_for_training=frac_for_training,
        adata_path=adata_path,
        use_weighted_sampler=use_weighted_sampler,
        label_key=sample_by,
        seed=seed,
        load_exp=use_onehot_wrapper is True,
    )

    # Load gpdb
    gpdb = pd.read_csv(gpdb_path)
    args['gpdb'] = gpdb

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

    # and similarity file
    if gp_similarity_file is not None:
        gp_similarity = np.load(gp_similarity_file, allow_pickle=True)
        gp_similarity = gp_similarity.astype('float32')

        # filter to match gp_inputs
        if gp_inputs is not None:
            # get indices for gp_inputs
            gp_idx = [gpdb.columns.get_loc(gp) for gp in gp_inputs]
            gp_similarity = gp_similarity[gp_idx, :][:, gp_idx]

    else:
        gp_similarity = None

    ############################################################################
    # Train model
    ############################################################################

    model = configure_model(args)

    pl_model = configure_lightning_module(model, gp_similarity, args)

    # Optionally load pretrained model
    if num_virtual_tokens > 0:
        pl_model = load_from_ckpt('virtual_tokens', pl_model, args)
    elif resume_training:
        pl_model = load_from_ckpt('resume_training', pl_model, args)
    elif global_training == 'sequential':
        pl_model = load_from_ckpt('sequential', pl_model, args)
    elif (global_training == 'finetune') | (global_training == 'finetune_global'):
        pl_model = load_from_ckpt('finetune', pl_model, args)

    if learn_new_gp:
        pl_model = load_from_ckpt('learn_new_gp', pl_model, args)

    # Optionally reset any trainer parameters
    pl_model.prototype_labels_key = prototype_labels_key
    pl_model.num_prototypes = num_prototypes
    pl_model.lambda_prototype_loss = lambda_prototype_loss

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
        precision='bf16-mixed' if strategy == 'ddp_find_unused_parameters_true' else 16,
        # profiler='advanced',
        num_nodes=num_nodes,
        strategy=strategy,
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
    )

    # Train the model
    trainer.fit(pl_model, txdata)

    # save logs to csv for custom plotting
    # Fetch logged data from wandb
    api = wandb.Api()

    if torch.cuda.device_count() > 1:
        run = api.run(f'scGPL/{save_id}_gpu_{str(rank_zero_only.rank)}')
    else:
        run = api.run(f'scGPL/{save_id}')

    # Get logged data as dataframe
    df = run.history()
    df.to_csv(f'{output_dir}/training_metrics.csv', index=False)

    wandb.finish()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------


def configure_wandb(args):
    # Get function specific arguments
    output_dir = args['output_dir']
    tissue = args['tissue']

    wandb.login()

    # get date for today in YYYY-MM-DD format
    today = datetime.datetime.today().strftime('%Y-%m-%d')

    supervised_tag = args['model_type']

    # create unique id for wandb run with 3 random characters
    unique_id = str(uuid.uuid4())[:3]

    save_id = f'{today}_gp_transformer_{tissue}_{supervised_tag}_{unique_id}'

    wandb_dir = os.path.join(output_dir, 'wandb_logs')

    # Check if directory exists
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    if torch.cuda.device_count() > 1:
        # multi gpu training with group logging
        wandb.init(
            project='scGPL',
            # group=f'{today}_gp_transformer_{tissue}_{supervised_tag}',
            # all runs are saved in one group for multi gpu training
            # =/ this doesnt work?
            id=save_id + f'_gpu_{str(rank_zero_only.rank)}',
            dir=wandb_dir,
        )
    else:
        wandb.init(project='scGPL', id=save_id, dir=wandb_dir)

    return save_id


def configure_callbacks(save_id, args):
    model_type = args['model_type']
    global_loss = args['global_loss']
    output_dir = args['output_dir']

    if model_type == 'Global':
        if global_loss == 'supervised':
            early_stopping_callback = EarlyStopping(
                monitor='val/accuracy',
                patience=3,
                mode='max',
            )
        else:
            early_stopping_callback = EarlyStopping(
                monitor='val/pearson',
                patience=3,
                mode='max',
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
        save_top_k=3,
        mode='min',
        save_last=True,
        # save every n steps --> issue if dataset has < n steps
        every_n_train_steps=10,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    return early_stopping_callback, checkpoint_callback, lr_monitor


def configure_logger(args):
    # create a logger to log training progress
    wandb_logger = WandbLogger(log_model=True)

    if rank_zero_only.rank == 0:
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
                'gp_latent_size': args['gp_latent_size'],
                'attn_dropout': args['attn_dropout'],
                'transformer_block': 'preLN',
                'learning_rate': args['lr'],
                'frac_for_training': args['frac_for_training'],
                'use_gp_similarity_loss': args['gp_similarity_file'] is not None,
                'lambda_gp_similarity': args['lambda_gp_similarity'],
                'use_flash': args['use_flash'],
                'weight_decay': args['weight_decay'],
                'num_virtual_tokens': args['num_virtual_tokens'],
                'condition_on_z_mean': args['mean_emb_dict'] is not None,
                'use_baseline_tk': args['use_baseline_tk'],
                'use_onehot_wrapper': args['use_onehot_wrapper'],
                'use_pos_emb': args['use_pos_emb'],
            }
        )

        if args['model_type'] == 'Global':
            wandb_logger.experiment.config.update(
                {
                    'global_attn_heads': args['global_attn_heads'],
                    'global_loss': args['global_loss'],
                    'global_training': args['global_training'],
                    'global_n_blocks': args['global_n_blocks'],
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
                    }
                )

            if args['global_loss'] == 'reconstruction':
                wandb_logger.experiment.config.update(
                    {
                        'reconstruction_loss': args['reconstruction_loss'],
                    }
                )

            if 'finetune' in args['global_training']:
                wandb_logger.experiment.config.update(
                    {
                        'finetune_lr': args['finetune_lr'],
                    }
                )

            if args['num_prototypes'] > 0:
                wandb_logger.experiment.config.update(
                    {
                        'num_prototypes': args['num_prototypes'],
                        'prototype_labels_key': args['prototype_labels_key'],
                        'lambda_prototype_loss': args['lambda_prototype_loss'],
                    }
                )

    return wandb_logger


def configure_model(args):
    common_params = {
        'gene_counts_df': args['gene_counts_df'],
        'database': args['gpdb'],
        'n_blocks': args['n_blocks'],
        'mgm_mask_ratio': args['mgm'],
        'num_heads': args['n_heads'],
        'gp_latent_size': args['gp_latent_size'],
        'attn_dropout': args['attn_dropout'],
        'gp_inputs': args['gp_inputs'],
        'use_flash': args['use_flash'],
        'learn_new_gp': args['learn_new_gp'],
        'geneformer_model': args['geneformer_model_path'],
        'peft_config_path': args['peft_config_path'],
        'use_baseline_tk': args['use_baseline_tk'],
        'tk_vocab_size': args['tk_vocab_size'],
        'gene2vec': args['gene2vec'],
        'use_pos_emb': args['use_pos_emb'],
        'use_onehot_wrapper': args['use_onehot_wrapper'],
        'vocab_gene_names': args['vocab_gene_names'],
        'do_ensembl_conversion': args['gene_format'] == 'symbol',
    }

    global_params = {
        'supervised_labels': args['supervised_labels'],
        'global_attn_heads': args['global_attn_heads'],
        'global_loss': args['global_loss'],
        'global_masking_rate': args['global_masking_rate'],
        'global_n_blocks': args['global_n_blocks'],
        'reconstruction_loss': args['reconstruction_loss'],
        'total_n_genes': args['total_n_genes'],
    }

    if args['num_virtual_tokens'] > 0:
        if args['model_type'] == 'Base':
            model = gpTransformerBaseWithPrompt(**common_params)

        elif args['model_type'] == 'Global':
            model = gpTransformerGlobalWithPrompt(**common_params, **global_params)

        return model

    if args['num_prototypes'] > 0:
        model = gpTransformerPrototypes(
            num_prototypes=args['num_prototypes'], **global_params
        )
        return model

    if args['model_type'] == 'Base':
        model = gpTransformerBase(**common_params)
        return model

    if args['model_type'] == 'Global':
        model = gpTransformerGlobal(**common_params, **global_params)
        return model

    if args['model_type'] == 'Mean':
        model = gfGlobal(**common_params)
        return model


def configure_lightning_module(model, gp_similarity, args):
    common_params = {
        'model': model,
        # 'model_type': args['model_type'],
        'lr': args['lr'],
        'finetune_lr': args['finetune_lr'],
        'use_finetune_lr': 'finetune' in args['global_training'],
        'total_epochs': args['n_epochs'],
        'lr_scheduler': args['lr_scheduler'],
        'use_gp_similarity_loss': gp_similarity is not None,
        'gp_similarity': gp_similarity,
        'output_dir': args['output_dir'],
        'lambda_gp_similarity': args['lambda_gp_similarity'],
        'weight_decay': args['weight_decay'],
        'set_gpfinder_weight_decay': args['set_gpfinder_weight_decay'],
        'optimizer': DeepSpeedCPUAdam
        if args['strategy'].startswith('deepspeed')
        else torch.optim.AdamW,
    }

    global_params = {
        'n_condition_combined': args['n_condition_combined'],
        'total_n_genes': args['total_n_genes'],
        'global_loss': args['global_loss'],
    }

    prototype_params = {
        'num_prototypes': args['num_prototypes'],
        'lambda_prototype_loss': args['lambda_prototype_loss'],
    }

    if args['num_prototypes'] > 0:
        pl_model = gpPrototypes(**common_params, **global_params, **prototype_params)

        return pl_model

    if args['model_type'] == 'Base':
        pl_model = gpBase(**common_params)
        return pl_model

    if args['model_type'] == 'Global':
        pl_model = gpGlobal(**common_params, **global_params)
        return pl_model


def load_from_ckpt(mode, pl_model, args):
    '''
    Load pretrained model

    Mode can be one of:
    - 'virtual_tokens'
    - 'resume_training'
    - 'sequential'
    - 'finetune'

    '''

    output_dir = args['output_dir']
    tissue = args['tissue']
    model_type = args['model_type']
    path_to_base_model = args['path_to_base_model']

    if mode == 'virtual_tokens':
        if args['path_to_base_model'] is not None:
            try:
                latest_ckpt = find_latest_file(path_to_base_model, tissue, model_type)
            except FileNotFoundError:
                latest_ckpt = find_latest_file(path_to_base_model, tissue, 'Base')
            checkpoint_path = os.path.join(path_to_base_model, latest_ckpt)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

            pl_model.load_state_dict(checkpoint['state_dict'], strict=False)

            # freeze everything but prompt tokens and global head
            for name, param in pl_model.model.named_parameters():
                if 'prompt' in name:
                    param.requires_grad = True
                elif (
                    ('cell_token_learner' in name)
                    | ('clf_head' in name)
                    | ('count_head' in name)
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        return pl_model

    elif mode == 'resume_training':
        latest_ckpt = find_latest_file(output_dir, tissue, model_type)
        pl_model = pl_model.load_from_checkpoint(latest_ckpt)

        return pl_model

    elif mode == 'sequential':
        latest_ckpt = find_latest_file(path_to_base_model, tissue, 'Base')
        checkpoint_path = os.path.join(path_to_base_model, latest_ckpt)
        checkpoint = torch.load(latest_ckpt, map_location=torch.device('cpu'))
        pl_model.load_state_dict(checkpoint['state_dict'], strict=False)

        # freeze base model
        for name, param in pl_model.model.named_parameters():
            if (
                ('cell_token_learner' in name)
                | ('clf_head' in name)
                | ('count_head' in name)
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

        return pl_model

    elif (mode == 'finetune') | (mode == 'finetune_global'):
        latest_ckpt = find_latest_file(path_to_base_model, tissue, 'Global')
        checkpoint_path = os.path.join(path_to_base_model, latest_ckpt)
        checkpoint = torch.load(latest_ckpt, map_location=torch.device('cpu'))
        pl_model.load_state_dict(checkpoint['state_dict'], strict=False)

        return pl_model

    elif mode == 'learn_new_gp':
        checkpoint_path = find_latest_file(path_to_base_model, tissue, model_type)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
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
