import argparse
import datetime
import os
import random
import sys
import uuid

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

from ..Datamodules.datamodule import txDataModule
from ..Models.gp_model import gpTransformerBase
from ..Trainers.trainer import scGPL
from ..Utils.utils import bool_flag, find_latest_file

############################################
# Arg Parser Function
############################################


def build_parser():
    """
    Helper function to build our program's argument parser.

    :returns ArgumentParser: The parser for our program's configuration.
    """
    parser = argparse.ArgumentParser(
        description=('Runs experiments for our tabular architecture.'),
    )
    parser.add_argument(
        '--dataset_path', '-p', default=None, help=('path to input tokenized dataset')
    )

    parser.add_argument(
        '--gpdb_path',
        '-g',
        default=None,
        help=(
            'path to input gp database, a pandas csv where '
            ' each column is a GP, with GP names as column names'
            '(eventually will be output of some processing pipeline)'
        ),
    )

    parser.add_argument(
        '--gp_similarity_file',
        default=None,
        help=(
            'path to input gp similarity file, a numpy array'
            'where x[i,j] is the similarity between GP i and GP j'
        ),
    )

    parser.add_argument(
        '--output_dir',
        '-o',
        default=None,
        help=(
            "directory where we will dump our experiment's results. If not "
            "given, then we will use the directory given as the 'results_dir' in "
            'the config file.'
        ),
        metavar='path',
    )

    parser.add_argument(
        '--mgm',
        '-m',
        default=0.5,
        type=float,
        help=('masking ratio for masked gene modeling ablation experiments'),
    )

    parser.add_argument(
        '--n_epochs',
        '-n',
        default=3,
        type=int,
        help=('number of epochs to train for'),
    )

    parser.add_argument(
        '--gene_format',
        default='symbol',
        help=('format in which gene names are stored in GPDB'),
    )

    parser.add_argument(
        '--batch_size',
        '-b',
        default=32,
        type=int,
        help=('batch size'),
    )

    parser.add_argument(
        '--model_type',
        '-u',
        default='Base',
        help=(
            'One of Base, Supervised or Unsupervised'
            'Where unsupervised has an extra self-attention head'
            'to learn a cell token based on GP tokens'
        ),
    )

    parser.add_argument(
        '--tissue',
        '-t',
        default=None,
        type=str,
        help=(
            'tissue name for logging experiment in wandb'
            'equivalent to directory name in examples subfolder'
        ),
    )

    parser.add_argument(
        '--n_heads',
        default=1,
        type=int,
        help=('number of heads for multi-head attention'),
    )

    parser.add_argument(
        '--n_blocks',
        default=1,
        type=int,
        help=('number of transformer blocks'),
    )

    parser.add_argument(
        '--lr_scheduler',
        default='ReduceLROnPlateau',
        help=(
            'learning rate scheduler for optimizer'
            'nb this is a string which will be converted to a class'
        ),
    )

    parser.add_argument(
        '--strategy',
        default='ddp_find_unused_parameters_true',
        help=('strategy for multi-GPU lightning trainer'),
    )

    parser.add_argument(
        '--gp_latent_size',
        default=64,
        type=int,
        help=(
            'size of latent space for GP tokens'
            'if <256, will use MLP to reduce dimensions of Geneformer gene embeddings'
            'else take embeddings directly'
        ),
    )

    parser.add_argument(
        '--attn_dropout',
        default=0,
        type=float,
        help=(
            'Dropout for attention layers'
            'NB only for final self attention block for now'
        ),
    )

    parser.add_argument(
        '--transformer_block',
        default='PreLN',
        help=(
            'transformer encoder architecture'
            'default is PreLN as in DiNO'
            'option for ResiDual, sigma_reparam'
        ),
    )

    parser.add_argument(
        '--lr',
        default=1e-3,
        type=float,
        help=('Model trainer learning rate'),
    )

    parser.add_argument(
        '--resume_training',
        default=False,
        type=bool_flag,
        help=("Set to 'from_checkpoint' to resume training from checkpoint"),
    )

    parser.add_argument(
        '--gene_counts_df',
        default=None,
        help=('Dataframe with the counts of each gene in the dataset'),
    )

    parser.add_argument(
        '--gp_inputs',
        default=None,
        nargs='+',
        help=('Which GP from GPDB to include in model' 'if None, defaults to all GP'),
    )

    return parser


############################################
# Main Function
############################################


def run_training(
    dataset_path,
    gpdb_path,
    gp_similarity_file,
    output_dir,
    batch_size,
    mgm,
    tissue,
    n_heads,
    n_blocks,
    lr_scheduler,
    n_epochs,
    gene_format,
    model_type,
    strategy,
    gp_latent_size,
    attn_dropout,
    transformer_block,
    lr,
    resume_training,
    gene_counts_df,
    gp_inputs,
):
    ##########################################
    # Setup
    ##########################################

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
                'transformer_block': transformer_block,
                'learning_rate': lr,
            }
        )

    ############################################################################
    # Dataset Preparation
    ############################################################################

    # Instantiate dataset
    # (tokenized dataset should be created already)
    # txdata = DummyDataModule(folder = dataset_path, batch_size=batch_size)
    txdata = txDataModule(folder=dataset_path, batch_size=batch_size)

    # dataset for getting number of classes
    # full_dataset = txDataset(folder = dataset_path)
    # print("Full dataset:", len(full_dataset), "cells")

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
        )

    else:
        raise ValueError('only Base implemented for now')

    use_gp_similarity_loss = gp_similarity_file is not None

    # Set up gpTransformer main module
    if strategy.startswith('deepspeed'):
        # use deepspeed optimizer if using deepspeed strategy
        gp_transformer = scGPL(
            model,
            model_type,
            total_epochs=n_epochs,
            lr=lr,
            lr_scheduler=lr_scheduler,
            optimizer=DeepSpeedCPUAdam,
            use_gp_similarity_loss=use_gp_similarity_loss,
            output_dir=output_dir,
        )
    else:
        # otherwise defaults to pytorch AdamW
        gp_transformer = scGPL(
            model,
            model_type,
            lr=lr,
            total_epochs=n_epochs,
            lr_scheduler=lr_scheduler,
            use_gp_similarity_loss=use_gp_similarity_loss,
            output_dir=output_dir,
        )

    # For continuing training from checkpoint
    if resume_training == 'from_checkpoint':
        latest_ckpt = find_latest_file(output_dir, tissue, model_type)
        checkpoint_path = os.path.join(output_dir, latest_ckpt)
        checkpoint = torch.load(checkpoint_path)
        gp_transformer.load_state_dict(checkpoint['state_dict'])
        n_epochs = checkpoint['epoch'] + n_epochs

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
            precision=16,
            profiler='simple',
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
            precision=16,
            profiler='advanced',
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

    ############################################################################
    print(' ')
    print('***')
    print('DONE')
    print('***')
    print(' ')

    return 0


################################################################################
# ENTRY POINT
################################################################################

if __name__ == '__main__':
    # First generate our argument parser
    parser = build_parser()
    args = parser.parse_args()
    args_dict = vars(args)

    # Then run our main function with those arguments
    sys.exit(run_training(**args_dict))
