import datetime
import os
import random
import uuid
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch

# set up wandb
import wandb

# from deepspeed.ops.adam import DeepSpeedCPUAdam
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from ..Datamodules.datamodule import txDataModule
from ..Models.gp_model import gpTransformerBase, gpTransformerGlobal
from ..Trainers.trainer import scGPL
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
    add_remaining_var: Optional[bool] = False,
    frac_for_training: Optional[float] = 1.0,
    lambda_gp_similarity: Optional[float] = 1e-2,
    global_loss: str = 'supervised',
    classification_labels: Optional[list] = None,
    global_attn_heads: Optional[int] = 8,
    supervised_labels: Optional[dict] = None,
    global_masking_rate: Optional[float] = 0.15,
    global_training: str = 'simultaneous',
    path_to_base_model: str = 'path/to/pretrained/model',
    learn_new_gp: Optional[bool] = False,
    gp_to_learn: list = ['novel_gp'],
    global_n_blocks: int = 1,
    reconstruction_loss: Optional[str] = 'mse',
    adata_path: Optional[str] = None,
    use_flash: Optional[bool] = False,
    weight_decay: float = 0.0,
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
    add_remaining_var : bool
        whether to intialize a new transformer block covering non GP genes
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

    """
    ##########################################
    # Setup
    ##########################################

    torch.set_float32_matmul_precision('medium')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set seed for reproducibility
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)
    torch.manual_seed(seed)

    wandb.login()

    # get date for today in YYYY-MM-DD format
    today = datetime.datetime.today().strftime('%Y-%m-%d')

    supervised_tag = model_type

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

    early_stopping_callback = EarlyStopping(
        # monitor='val/loss',
        monitor='train/loss_epoch',
        patience=5,
        mode='min',
    )

    # Define a directory for checkpoints within the output directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    # Make sure the directory exists, create it if not
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # monitor='val/loss',
        monitor='train/loss_epoch',
        dirpath=checkpoint_dir,
        filename=save_id,
        save_top_k=1,
        mode='min',
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # create a logger to log training progress
    wandb_logger = WandbLogger(log_model=True)

    if rank_zero_only.rank == 0:
        wandb_logger.experiment.config.update(
            {
                'dataset': dataset_path.split('/')[-3],
                'supervise': model_type,
                'architecture': 'gp_transformer',
                'epochs': n_epochs,
                'mgm': mgm,
                'n_heads': n_heads,
                'n_blocks': n_blocks,
                'lr_scheduler': lr_scheduler,
                'batch_size': batch_size,
                'strategy': strategy,
                'gp_latent_size': gp_latent_size,
                'attn_dropout': attn_dropout,
                'transformer_block': 'preLN',
                'learning_rate': lr,
                'frac_for_training': frac_for_training,
                'use_gp_similarity_loss': gp_similarity_file is not None,
                'lambda_gp_similarity': lambda_gp_similarity,
                'use_flash': use_flash,
                'weight_decay': weight_decay,
            }
        )

        if model_type == 'Global':
            wandb_logger.experiment.config.update(
                {
                    'global_attn_heads': global_attn_heads,
                    'global_loss': global_loss,
                    'global_training': global_training,
                    'global_n_blocks': global_n_blocks,
                }
            )

            if global_loss == 'supervised':
                wandb_logger.experiment.config.update(
                    {
                        'classification_labels': classification_labels,
                    }
                )

            if global_loss == 'masking':
                wandb_logger.experiment.config.update(
                    {
                        'global_masking_rate': global_masking_rate,
                    }
                )

            if global_loss == 'reconstruction':
                wandb_logger.experiment.config.update(
                    {
                        'reconstruction_loss': reconstruction_loss,
                    }
                )

            if global_training == 'finetune':
                wandb_logger.experiment.config.update(
                    {
                        'finetune_lr': finetune_lr,
                    }
                )

    ############################################################################
    # Dataset Preparation
    ############################################################################

    # Optionally load anndata object
    if (model_type == 'Global') & (global_loss == 'reconstruction'):
        if adata_path is None:
            raise ValueError('Please provide path to anndata object')
        else:
            adata = sc.read_h5ad(adata_path)
            total_n_genes = adata.X.shape[1]
            if 'batch_key' in adata.obs.columns:
                n_condition_combined = adata.obs['batch_key'].nunique()
            else:
                if reconstruction_loss in ['zinb', 'nb']:
                    raise ValueError(
                        'No batch_key found'
                        'for ZINB or NB reconstruction loss'
                        'Please provide batch_key in adata.obs'
                        'by passing batch_keys argument to preprocess function'
                    )

    else:
        adata = None
        total_n_genes = 0
        n_condition_combined = 1

    # Instantiate dataset
    # (tokenized dataset should be created already)
    # txdata = DummyDataModule(folder = dataset_path, batch_size=batch_size)
    if reconstruction_loss == 'mse':
        transform_adata = True
    else:
        transform_adata = False

    txdata = txDataModule(
        folder=dataset_path,
        batch_size=batch_size,
        frac_for_training=frac_for_training,
        adata=adata,
        transform_adata=transform_adata,
    )

    # Load gpdb
    gpdb = pd.read_csv(gpdb_path)

    if gene_format == 'symbol':
        do_ensembl_conversion = True
    elif gene_format == 'ensembl':
        do_ensembl_conversion = False

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

    if gene_counts_df is not None:
        gene_counts_df = pd.read_csv(gene_counts_df)

    ############################################################################
    # Train model
    ############################################################################

    if model_type == 'Base':
        model = gpTransformerBase(
            gene_counts_df=gene_counts_df,
            database=gpdb,
            do_ensembl_conversion=do_ensembl_conversion,
            n_blocks=n_blocks,
            mgm_mask_ratio=mgm,
            num_heads=n_heads,
            gp_latent_size=gp_latent_size,
            attn_dropout=attn_dropout,
            gp_inputs=gp_inputs,
            add_remaining_var=add_remaining_var,
            use_flash=use_flash,
        )

    elif model_type == 'Global':
        # very slow --> provide dictionary as input
        # if global_loss == 'supervised':
        #     # set up dictionary with number of classes for supervised labels
        #     supervised_labels = txdata.count_unique_classes(classification_labels)

        model = gpTransformerGlobal(
            gene_counts_df=gene_counts_df,
            database=gpdb,
            do_ensembl_conversion=do_ensembl_conversion,
            n_blocks=n_blocks,
            mgm_mask_ratio=mgm,
            num_heads=n_heads,
            gp_latent_size=gp_latent_size,
            attn_dropout=attn_dropout,
            gp_inputs=gp_inputs,
            add_remaining_var=add_remaining_var,
            supervised_labels=supervised_labels,
            global_attn_heads=global_attn_heads,
            global_loss=global_loss,
            global_masking_rate=global_masking_rate,
            global_n_blocks=global_n_blocks,
            reconstruction_loss=reconstruction_loss,
            total_n_genes=total_n_genes,
            use_flash=use_flash,
        )

    else:
        raise ValueError('only model types Base or Global implemented for now')

    use_gp_similarity_loss = gp_similarity_file is not None

    # Set up gpTransformer main module
    if strategy.startswith('deepspeed'):
        # use deepspeed optimizer if using deepspeed strategy
        gp_transformer = scGPL(
            model,
            model_type,
            global_loss=global_loss,
            total_epochs=n_epochs,
            lr=lr,
            finetune_lr=finetune_lr,
            use_finetune_lr=global_training == 'finetune',
            lr_scheduler=lr_scheduler,
            # optimizer=DeepSpeedCPUAdam,
            use_gp_similarity_loss=use_gp_similarity_loss,
            gp_similarity=gp_similarity,
            output_dir=output_dir,
            lambda_gp_similarity=lambda_gp_similarity,
            n_condition_combined=n_condition_combined,
            total_n_genes=total_n_genes,
            weight_decay=weight_decay,
        )
    else:
        # otherwise defaults to pytorch AdamW
        gp_transformer = scGPL(
            model,
            model_type,
            global_loss=global_loss,
            lr=lr,
            finetune_lr=finetune_lr,
            use_finetune_lr=global_training == 'finetune',
            total_epochs=n_epochs,
            lr_scheduler=lr_scheduler,
            use_gp_similarity_loss=use_gp_similarity_loss,
            gp_similarity=gp_similarity,
            output_dir=output_dir,
            lambda_gp_similarity=lambda_gp_similarity,
            n_condition_combined=n_condition_combined,
            total_n_genes=total_n_genes,
            weight_decay=weight_decay,
        )

    # For continuing training from checkpoint
    if resume_training:
        latest_ckpt = find_latest_file(output_dir, tissue, model_type)
        checkpoint_path = os.path.join(output_dir, latest_ckpt)
        checkpoint = torch.load(checkpoint_path)
        gp_transformer.load_state_dict(checkpoint['state_dict'])
        n_epochs = checkpoint['epoch'] + n_epochs

    # For training global model after base model
    if global_training == 'sequential':
        if path_to_base_model is None:
            raise ValueError(
                'Please provide path to pre-trained'
                'gpTransformer Base model for sequential training'
            )
        # look for Base model to load
        # if not found, this will raise an error
        latest_ckpt = find_latest_file(path_to_base_model, tissue, 'Base')
        checkpoint_path = os.path.join(path_to_base_model, latest_ckpt)
        print('Loading from checkpoint', checkpoint_path)
        checkpoint = torch.load(latest_ckpt)
        gp_transformer.load_state_dict(checkpoint['state_dict'], strict=False)
        # n_epochs = checkpoint['epoch'] + n_epochs  # TO DO : do we need this line?

        # reset output directory
        gp_transformer.output_dir = output_dir

        # freeze base model
        for name, param in gp_transformer.model.named_parameters():
            if (
                ('cell_token_learner' in name)
                | ('clf_head' in name)
                | ('count_head' in name)
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # For training global model after base model
    # but finetuning original GP blocks
    if global_training == 'finetune':
        if path_to_base_model is None:
            raise ValueError(
                'Please provide path to pre-trained'
                'gpTransformer Base model for finetuning'
            )
        # look for Base model to load
        # if not found, this will raise an error
        latest_ckpt = find_latest_file(path_to_base_model, tissue, 'Base')
        checkpoint_path = os.path.join(path_to_base_model, latest_ckpt)
        print('Loading from checkpoint', checkpoint_path)
        checkpoint = torch.load(latest_ckpt)
        gp_transformer.load_state_dict(checkpoint['state_dict'], strict=False)
        # n_epochs = checkpoint['epoch'] + n_epochs  # TO DO : do we need this line?

        # reset output directory
        gp_transformer.output_dir = output_dir

    # Learning new GP
    if learn_new_gp:
        # load pretrained model
        latest_ckpt = find_latest_file(path_to_base_model, tissue, model_type)
        checkpoint_path = os.path.join(path_to_base_model, latest_ckpt)
        checkpoint = torch.load(checkpoint_path)
        gp_transformer.load_state_dict(checkpoint['state_dict'], strict=False)
        n_epochs = checkpoint['epoch'] + n_epochs
        gp_transformer.output_dir = output_dir

        # get indices of GP to learn
        if isinstance(gp_to_learn, str):
            gp_to_learn = [gp_to_learn]
        gp_idx = [
            gp_transformer.model.gp_inputs.index(gp)
            for gp in gp_to_learn
            if gp in gp_transformer.model.gp_inputs
        ]

        # freeze all GP
        for name, param in gp_transformer.model.named_parameters():
            if 'multi_gp_encoder' in name:
                param.requires_grad = False

        # unfreeze new GP
        for i in gp_idx:
            for name, param in gp_transformer.model.named_parameters():
                if f'multi_gp_encoder.encoder.{i}' in name:
                    param.requires_grad = True

    # check number of available GPUs
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
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
            accelerator='auto',  # uses ddp per default for multi-gpu training
            strategy=strategy,
            precision='bf16-mixed',
            # profiler='simple',
        )
    else:
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
            precision='bf16-mixed',
            # profiler='simple',
            strategy=strategy,
        )

    # Ready to train with new learning rate
    trainer.fit(gp_transformer, txdata)

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
