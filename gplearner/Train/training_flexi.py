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
from ..Models.gp_model import (
    gpTransformerBase,
    gpTransformerBaseWithPrompt,
    gpTransformerGlobal,
    gpTransformerGlobalLinear,
    gpTransformerGlobalWithPrompt,
)
from ..Trainers.trainer import (
    gpBase,
    gpGlobal,
)
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
    model_type_old: str = 'Base',
    strategy: str = 'ddp_find_unused_parameters_true',
    attn_dropout: float = 0.0,
    lr: float = 1e-3,
    gp_inputs_old: Optional[list] = None,
    gp_inputs_new: Optional[list] = None,
    frac_for_training: Optional[float] = 1.0,
    lambda_gp_similarity: Optional[float] = 1e-2,
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
    learn_new_gp: Optional[bool] = False,
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
    use_go_similarity_loss: bool = False,
    lambda_go_similarity: float = 1e-2,
    go_similarity_path: Optional[str] = None,
    go_similarity_gp: Optional[str] = 'hvg',
    use_gp_similarity_loss: bool = False,
    num_virtual_tokens: int = 0,
    all_genes: Optional[list] = None,
    use_pos_emb: Optional[str] = 'sin_cos',
    use_onehot_wrapper: bool = False,
    vocab_gene_names: Optional[str] = None,
    num_nodes: int = 1,
    limit_train_batches: float = 1.0,
    limit_val_batches: float = 1.0,
    val_check_interval: float = 1.0,
    precision=32,  # 'bf16-mixed',
    bert_config: Dict = {},
    use_gf_embeddings: Optional[bool] = False,
    load_cell_token_learner: bool = False,
    gp_of_interest: Optional[str] = None,
    gp_for_downstream: Optional[str] = None,
    gp_latent_size: Optional[int] = None,
    accumulate_grad_batches: Optional[int] = 1,
    resume_training: bool = False,
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
    attn_dropout : float
        Dropout for attention layers
        NB only for final self attention block for now
    lr : float
        Model trainer learning rate
    resume_training : bool
        Set to True to resume training from checkpoint
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
    load_cell_token_learner:
        whether to load the cell token learner from previous global training
    gp_of_interest
        Sole GP to use in forward pass
    gp_for_downstream
        GP for downstream calculation of attention etc.
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
        load_exp=use_onehot_wrapper is True,
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

    # and similarity file
    if gp_similarity_file is not None:
        gp_similarity = np.load(gp_similarity_file, allow_pickle=True)
        gp_similarity = gp_similarity.astype('float32')

        # filter to match gp_inputs
        if gp_inputs_new is not None:
            # get indices for gp_inputs
            gp_idx = [gpdb.columns.get_loc(gp) for gp in gp_inputs_new]
            gp_similarity = gp_similarity[gp_idx, :][:, gp_idx]

    else:
        gp_similarity = None

    ############################################################################
    # Train model
    ############################################################################

    model_v0 = configure_model_version(args, 'old')

    model_v1 = configure_model_version(args, 'new')

    gp_transformer_v0 = configure_lightning_module_version(
        model_v0, 'old', gp_similarity, args
    )

    gp_transformer = configure_lightning_module_version(
        model_v1, 'new', gp_similarity, args
    )

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
        'learn_new_gp': args['learn_new_gp'],
        'fm_encoder_pkg': args['fm_encoder_pkg'],
        'fm_encoder_name': args['fm_encoder_name'],
        'peft_config_path': args['peft_config_path'],
        'use_pos_emb': args['use_pos_emb'],
        'use_onehot_wrapper': args['use_onehot_wrapper'],
        'vocab_gene_names': args['vocab_gene_names'],
        'do_ensembl_conversion': args['gene_format'] == 'symbol',
        'bert_config': args['bert_config'],
        'all_genes': args['all_genes'],
        'gp_latent_size': args['gp_latent_size'],
        'use_gf_embeddings': args['use_gf_embeddings'],
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

    if args['num_virtual_tokens'] > 0:
        if args[f'model_type_{tag}'] == 'Base':
            model = gpTransformerBaseWithPrompt(**common_params)

        elif args[f'model_type_{tag}'] == 'Global':
            model = gpTransformerGlobalWithPrompt(**common_params, **global_params)

        return model

    if model_type == 'Base':
        model = gpTransformerBase(**common_params)
        return model

    if model_type == 'Global':
        if args['gp_of_interest'] is not None:
            model = gpTransformerGlobalLinear(**common_params, **global_params)
        else:
            model = gpTransformerGlobal(**common_params, **global_params)
        return model

    if model_type == 'Mean':
        model = gfGlobal(**common_params)
        return model


def configure_lightning_module_version(model, tag, gp_similarity, args):
    common_params = {
        'model': model,
        'lr': args['lr'],
        'total_epochs': args['n_epochs'],
        'lr_scheduler': args['lr_scheduler'],
        'use_gp_similarity_loss': gp_similarity is not None,
        'gp_similarity': gp_similarity,
        'output_dir': args['output_dir'],
        'lambda_gp_similarity': args['lambda_gp_similarity'],
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
