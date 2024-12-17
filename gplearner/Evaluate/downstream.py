import os
import random
import warnings
from typing import (
    Any,
    Dict,
    Optional,
)

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import pytorch_lightning as pl
import scanpy as sc
import scprep
import seaborn as sns
import torch
from captum.attr import GuidedGradCam
from datasets import load_from_disk
from geneformer import ENSEMBL_DICTIONARY_FILE, TOKEN_DICTIONARY_FILE
from pytorch_lightning.loggers import CSVLogger
from scipy.spatial.distance import cosine
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from ..Datamodules.datamodule import (
    EmbDataModule,
    iEmbDataModule,
    iTxDataModule,
    txDataModule,
)
from ..Metrics.metrics import evaluate_emd_ref_vs_query
from ..Models.baselines import gfGlobal
from ..Models.interpretability import iGlobalWrapper, iGpWrapper
from ..Trainers.trainer import (
    EmbEvaluator,
    gpBase,
    gpGlobal,
    gpPrototypes,
)
from ..Utils.geneformer_utils import get_gf_repo
from ..Utils.utils import (
    MidpointNormalize,
    align_indices,
    build_token_to_gene_name_dict,
    find_latest_file,
    remove_single_data_points,
    summarize_attributions,
    wrangle_classification_report,
)

# for exporting pdfs
matplotlib.rcParams['pdf.fonttype'] = 42
# torch.set_float32_matmul_precision('medium')

############################################
# Main class
############################################


class gpEval:
    """
    Main class for running downstream evaluation tasks on trained models
    Parameters
    ----------
    dataset_path : str
        Path to folder containing tokenized dataset
    gpdb_path : str
        Path to gene program database
    gp_similarity_file : str
        Path to gene program similarity matrix
    output_dir : str
        Path to directory where we will save outputs
    batch_size : int
        Batch size
    n_blocks : int
        Number of transformer blocks
    gene_format : str
        Format in which gene names are stored in GPDB
    tissue : str
        Tissue name for logging experiment in wandb
        Equivalent to directory name in examples subfolder
    model_type : str
        One of Base, Supervised or Unsupervised
        Where unsupervised has an extra self-attention head
        to learn a cell token based on GP tokens
    n_heads : int
        Number of heads for multi-head attention
    gp_latent_size : int
        Size of latent space for GP tokens
        If <256, will use MLP to reduce dimensions of Geneformer gene embeddings
        Else take embeddings directly
    gp_inputs : list
        Which GP from GPDB to include in model
        if None, defaults to all GP
    gene_counts_df : str
        Dataframe with the counts of each gene in the dataset
    supervised_labels : list
        Dict {label : num_classes} for supervised classification
    global_attn_heads : int
        number of heads for learning cell token in global attention model
    global_loss :
        loss used to train global attention model
        (for compatibility with gpGlobal init)

    Returns
    -------

    """

    def __init__(
        self,
        gpdb_path: Optional[str] = None,
        output_dir: str = '/path/to/output/',
        dataset_path: Optional[str] = None,
        tissue: Optional[str] = 'test',
        model_type: Optional[str] = 'Base',
        model_type_in_checkpoint: Optional[str] = None,
        batch_size: Optional[int] = 128,
        path_to_trained_model: Optional[str] = None,
        seed: Optional[int] = 0,
        hparam_save: Optional[str] = 'all',
        num_virtual_tokens: Optional[int] = 0,
        cond_to_shift: Optional[Dict] = None,
        return_classification_report: Optional[bool] = False,
        # for gpmean only
        # otherwise loaded from checkpoint
        gene_format: Optional[str] = 'symbol',
        gp_inputs: Optional[list] = None,
        gpmean_fm_encoder_pkg: Optional[str] = 'geneformer',
        gpmean_fm_encoder_name: Optional[str] = 'gf-6L-30M-i2048',
    ):
        # set seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.seed = seed

        # Search for .ckpt files in the directory
        if model_type != 'Mean':
            tag = (
                model_type_in_checkpoint
                if model_type_in_checkpoint is not None
                else model_type
            )

            if path_to_trained_model is None:
                model_path = output_dir
            else:
                model_path = path_to_trained_model

            latest_ckpt = find_latest_file(model_path, tissue, tag)
            print('Latest .ckpt file:', latest_ckpt)
            self.checkpoint_path = os.path.join(model_path, latest_ckpt)

        gpdb = pd.read_csv(gpdb_path)

        # change directory for saving outputs
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir
        self.tissue = tissue
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.gpdb = gpdb
        self.gpmean_fm_encoder_pkg = gpmean_fm_encoder_pkg
        self.gpmean_fm_encoder_name = gpmean_fm_encoder_name

        # to avoid error when using geneformer finetuned model
        self.hparam_save = hparam_save

        # Set up gpTransformer lightning module
        self.model_type = model_type
        self.num_virtual_tokens = num_virtual_tokens
        self.cond_to_shift = cond_to_shift

        # save hparam for gpmean
        if model_type == 'Mean':
            self.gpdb = gpdb
            self.do_ensembl_conversion = gene_format != 'ensembl'

            if gp_inputs is None:
                self.gp_inputs = list(gpdb.columns)
            elif isinstance(gp_inputs, str):
                self.gp_inputs = [gp_inputs]
            else:
                self.gp_inputs = gp_inputs

        self.gp_transformer = self._init_trainer(
            return_classification_report=return_classification_report,
            hparam_save=self.hparam_save,
            num_virtual_tokens=num_virtual_tokens,
        )

    def _init_trainer(
        self,
        return_gene_embeddings=False,
        tokens_to_keep=None,
        genes_to_keep=None,
        gene_dir_tag=None,
        return_attention=False,
        gp=None,
        return_classification_report=False,
        test_random_baseline=False,
        save_emb=False,
        split_label=None,
        hparam_save='ignore_model',  # fine for test time?
        num_virtual_tokens=0,
        return_virtual_tokens=False,
        token_to_gene_to_keep_dict=None,
    ):
        if self.model_type == 'Base':
            gp_transformer = gpBase.load_from_checkpoint(
                self.checkpoint_path, hparam_save=hparam_save, map_location='cpu'
            )

        elif self.model_type == 'Global':
            gp_transformer = gpGlobal.load_from_checkpoint(
                self.checkpoint_path, hparam_save=hparam_save, map_location='cpu'
            )

        elif self.model_type == 'Prototypes':
            gp_transformer = gpPrototypes.load_from_checkpoint(
                self.checkpoint_path, hparam_save=hparam_save, map_location='cpu'
            )

        elif self.model_type == 'Mean':
            # only needs gp mean model set up in init
            # to do: option for passing custom token dictonary file?
            # (those are the only args)
            model = gfGlobal(
                database=self.gpdb,
                do_ensembl_conversion=self.do_ensembl_conversion,
                fm_encoder_pkg=self.gpmean_fm_encoder_pkg,
                fm_encoder_name=self.gpmean_fm_encoder_name,
            )

            gp_transformer = gpGlobal(model=model, global_loss='mean')

        # reset attributes overwritten by loading from checkpoint
        gp_transformer.return_gene_embeddings = return_gene_embeddings
        gp_transformer.tokens_to_keep = tokens_to_keep
        gp_transformer.genes_to_keep = genes_to_keep
        gp_transformer.token_to_gene_to_keep_dict = token_to_gene_to_keep_dict
        gp_transformer.gene_dir_tag = gene_dir_tag
        gp_transformer.return_attention = return_attention
        gp_transformer.gp = gp
        gp_transformer.return_classification_report = return_classification_report
        gp_transformer.output_dir = self.output_dir
        gp_transformer.test_random_baseline = test_random_baseline
        gp_transformer.save_emb = save_emb
        gp_transformer.split_label = split_label
        gp_transformer.return_virtual_tokens = return_virtual_tokens

        gp_transformer.model.multi_gp_encoder.num_virtual_tokens = num_virtual_tokens
        gp_transformer.model.cond_to_shift = self.cond_to_shift

        if hasattr(gp_transformer.model, 'cell_token_learner'):
            gp_transformer.model.cell_token_learner.num_virtual_tokens = (
                num_virtual_tokens
            )

        # Extract model
        self.model = gp_transformer.model
        self.gp_inputs = gp_transformer.model.gp_inputs

        # Extract pretrained encoder config
        self.fm_encoder_pkg = gp_transformer.model.fm_encoder_pkg

        if self.fm_encoder_pkg == 'geneformer':
            self.fm_encoder_name = gp_transformer.model.fm_encoder_name
            self.max_len = (
                gp_transformer.model.gf_wrapper.gf.config.max_position_embeddings
            )
        elif self.fm_encoder_pkg == 'from_scratch':
            self.fm_encoder_name = gp_transformer.model.fm_encoder_pkg
            self.max_len = self.model.gf_wrapper.model.config.max_position_embeddings

        # Disable flash for attention matrix generation
        if return_attention:
            for i, gp in enumerate(self.gp_inputs):
                for j in range(self.model.multi_gp_encoder.n_blocks):
                    self.model.multi_gp_encoder.encoder[i].blocks[
                        j
                    ].attn.use_flash = False

            if hasattr(self.model, 'cell_token_learner'):
                for j in range(self.model.cell_token_learner.n_blocks):
                    self.model.cell_token_learner.encoder.blocks[
                        j
                    ].attn.use_flash = False

        return gp_transformer

    def generate_embeddings(self, split='train', precision=32):
        '''
        Save embeddings as Dataset
        '''

        gp_transformer = self._init_trainer(
            save_emb=True,
            split_label=split,
            hparam_save=self.hparam_save,
            num_virtual_tokens=self.num_virtual_tokens,
        )

        txdata = txDataModule(
            folder=self.dataset_path,
            batch_size=self.batch_size,
            data_split_to_pass_to_test_step=split,
            seed=self.seed,
            fm_encoder_name=self.fm_encoder_name,
            model_input_size=self.max_len,
        )

        trainer = pl.Trainer(
            max_epochs=1, devices=1, accelerator='auto', precision=precision
        )

        trainer.test(gp_transformer, txdata)

    @staticmethod
    def evaluate_embeddings(
        y_label,
        folder_path,
        output_dir,
        emb_label,
        task='classification',
        emb_dim=256,
        lr=1e-3,
        batch_size=128,
        num_workers=1,
        meta_labels=None,
        data_type='dataset',
        n_epochs=3,
        continuous_cov=[],
        use_weighted_sampler=False,
        sample_by=None,
        filter_key=None,
        filter_value=None,
        encode_covariate=False,
        filter_tag=None,
        condition_variable=None,
        # development
        frac_for_training=1,
        mode=None,
    ):
        '''
        Train nn.Linear layer based on embeddings
        '''

        os.makedirs(os.path.join(output_dir, 'cell_metrics'), exist_ok=True)
        ckpt_dir = os.path.join(output_dir, 'evaluation_model_checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)

        # set seed for reproducibility
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f'Evaluating {emb_label} embeddings')
        if filter_tag is None:
            filter_tag = (
                f'_{filter_key}_{filter_value}' if filter_value is not None else ''
            )
        else:
            filter_tag = f'_{filter_tag}'
        if task == 'classification':
            clf_label = y_label
        else:
            clf_label = None

        if meta_labels is None:
            meta_labels = [y_label]

        emb_dm = EmbDataModule(
            folder_path,
            batch_size=batch_size,
            num_workers=num_workers,
            emb_label=emb_label,
            meta_labels=meta_labels,
            data_type=data_type,
            continuous_cov=continuous_cov,
            use_weighted_sampler=use_weighted_sampler,
            label_key=sample_by,
            filter_key=filter_key,
            filter_value=filter_value,
            clf_label=clf_label,
            encode_covariate=encode_covariate,
            frac_for_training=frac_for_training,
            mode=mode,
            condition_variable=condition_variable,
        )

        emb_dm.setup()

        emb_evaluator = EmbEvaluator(
            n_classes=emb_dm.num_classes,
            emb_dim=emb_dim,
            task=task,
            lr=lr,
            emb_label=emb_label,
            y_label=y_label,
            output_dir=output_dir,
            filter_tag=filter_tag,
            num_condition_cat=emb_dm.num_condition_classes,
        )

        logger = CSVLogger(
            os.path.join(output_dir, 'evaluation_logs'),
            name=f"{emb_label}_{y_label.replace('_id', '')}{filter_tag}",
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=ckpt_dir,
            filename=f"{y_label.replace('_id', '')}_{emb_label}_{task}{filter_tag}",
            save_top_k=1,
            mode='min',
        )

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            callbacks=[checkpoint_callback],
            devices=1,
            accelerator='auto',
            logger=logger,
            # precision=16,
        )

        trainer.fit(emb_evaluator, emb_dm)
        trainer.test(emb_evaluator, emb_dm)

    def visualize(
        self,
        label_to_plot,
        data_to_plot='test',
        gp_to_plot=None,
        subsample=None,
    ):
        """
        UMAP of GP embeddings
        """
        os.chdir(self.output_dir)

        if isinstance(label_to_plot, str):
            label_to_plot = [label_to_plot]

        if gp_to_plot is None:
            gp_to_plot = list(self.gp_inputs)

        if isinstance(gp_to_plot, str):
            gp_to_plot = [gp_to_plot]

        emb = load_from_disk(os.path.join('embeddings', f'{data_to_plot}_set'))

        if subsample is not None:
            emb = emb.shuffle(seed=0).select(range(subsample))

        for gp in gp_to_plot:
            x = np.array(emb[gp])
            y = pd.DataFrame(
                {k: emb[k] for k in emb.column_names if k in label_to_plot}
            )
            adata = sc.AnnData(X=x, obs=y)

            for c in label_to_plot:
                adata = remove_single_data_points(adata, c)

            sc.pp.neighbors(adata, use_rep='X', n_neighbors=15)
            sc.tl.umap(adata)

            for c in label_to_plot:
                sc.pl.umap(
                    adata,
                    color=c,
                    save=f'_{self.tissue}_{gp}_by_{c}.pdf',
                    frameon=False,
                )

    @staticmethod
    def _load_and_save_latent(self, adata, new, model_name):
        if new.shape[0] != adata.shape[0]:
            idx_union = set(new.obs['idx']).union(set(adata.obs['idx']))
            new = new[new.obs['idx'].isin(idx_union)]
            adata = adata[adata.obs['idx'].isin(idx_union)]

        adata.obsm[model_name] = new.X

        return adata

    def generate_gene_embeddings(
        self,
        pathway,
        split='train',
        obs_key=None,
        obs_value=None,
        data_frac=1,
        genes_to_keep=None,
        output_tag=None,
        do_ensembl_conversion=True,
        precision=32,
    ):
        """
        Save gene embeddings as Dataset

        Parameters
        ----------
        split : str
            Data split to use for generating embeddings
        obs_key : str
            Key in adata.obs to filter on
        obs_value : str
            Value in adata.obs to filter on
        data_frac : float
            Fraction of data to use for generating embeddings
        pathway : str
            Pathway to use for generating embeddings
        genes_to_keep : list
            Genes to generate embeddings for
            if None --> all genes


        Use find_genes_in_multiple_gp or get_genes_in_single_gp from Utils.utils
        for GP selection

        """
        os.chdir(self.output_dir)

        gene_dir_tag = f'{pathway}_gene_embeddings'

        if obs_value is not None:
            if isinstance(obs_value, str):
                gene_dir_tag += f'_from_{obs_value}'
            else:
                unpacked_label = '_'.join(map(str, obs_value))
                gene_dir_tag += f'_{unpacked_label}'

        if output_tag is not None:
            gene_dir_tag += f'_{output_tag}'

        tokens_to_keep, token_to_gene_to_keep_dict = build_token_to_gene_name_dict(
            self.gp_transformer.model.gene_name_path,
            self.gp_transformer.model.gene_token_path,
            genes_to_keep,
            do_ensembl_conversion,
        )

        gp_transformer = self._init_trainer(
            return_gene_embeddings=True,
            gene_dir_tag=gene_dir_tag,
            tokens_to_keep=tokens_to_keep,
            genes_to_keep=tokens_to_keep,
            gp=pathway,
            split_label=split,
            num_virtual_tokens=self.num_virtual_tokens,
            token_to_gene_to_keep_dict=token_to_gene_to_keep_dict,
        )

        txdata = txDataModule(
            folder=self.dataset_path,
            batch_size=self.batch_size,
            data_split_to_pass_to_test_step=split,
            filter_key=obs_key,
            filter_value=obs_value,
            frac_for_generation=data_frac,
            seed=self.seed,
            fm_encoder_name=self.fm_encoder_name,
            model_input_size=self.max_len,
        )

        trainer = pl.Trainer(
            max_epochs=1, devices=1, accelerator='auto', precision=precision
        )
        trainer.test(gp_transformer, txdata)

    def visualize_gene_embeddings(
        self,
        cell_label_to_plot,
        genes_to_plot,
        gene_label_to_plot,
        gene_label_df,
        gene_embedding_dir,
        output_dir,
        pathway=None,
        frac=1,
        gene_col_name='gene',
    ):
        if isinstance(cell_label_to_plot, str):
            cell_label_to_plot = [cell_label_to_plot]

        # Load gene embeddings
        emb = load_from_disk(gene_embedding_dir)
        emb = emb.shuffle(seed=0).select(range(int(frac * len(emb))))

        # Wrangle into anndata
        holder = []

        for g in genes_to_plot:
            if g not in emb.column_names:
                warnings.warn(f'{g} not in embeddings. Skipping {g}')
                continue

            x = np.array(emb[g])
            y = pd.DataFrame(
                {k: emb[k] for k in emb.column_names if k in cell_label_to_plot}
            )
            y['gene'] = g
            y['geneformer_rank'] = np.array(emb[f'{g}_rank'])
            gdata = sc.AnnData(X=x, obs=y)
            # remove missing genes
            gdata = gdata[gdata.obs['geneformer_rank'] != -1]
            holder.append(gdata)

        adata = ad.concat(holder)

        # add gene metadata
        if gene_label_df is not None:
            adata.obs = adata.obs.join(
                gene_label_df.set_index(gene_col_name), on='gene'
            )

        if adata.shape[0] == 0:
            raise ValueError('No genes remaining after removing missing genes')

        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.umap(adata)

        # change directory for saving figures
        os.chdir(output_dir)

        for c in cell_label_to_plot:
            sc.pl.umap(
                adata,
                color=c,
                save=f'_{pathway}_genes_by_{c}.pdf',
                frameon=False,
            )

        for c in gene_label_to_plot:
            sc.pl.umap(
                adata,
                color=c,
                save=f'_{pathway}_genes_by_{c}.pdf',
                frameon=False,
            )

    def generate_attention_matrix(
        self,
        gp,
        genes_to_keep=None,
        do_ensembl_conversion=True,
        split='test',
        precision=32,
    ):
        """
        Get attention weights from gpTransformer
        """
        os.chdir(self.output_dir)

        if (gp != 'cell_token') and (gp not in self.gp_inputs):
            raise ValueError(f'{gp} must be one of "cell_token" or {self.gp_inputs}')

        if gp != 'cell_token':
            _, token_to_gene_to_keep_dict = build_token_to_gene_name_dict(
                self.gp_transformer.model.gene_name_path,
                self.gp_transformer.model.gene_token_path,
                genes_to_keep,
                do_ensembl_conversion,
            )
        else:
            token_to_gene_to_keep_dict = None

        # Initialize trainer
        txdata = txDataModule(
            folder=self.dataset_path,
            batch_size=self.batch_size,
            data_split_to_pass_to_test_step=split,
            fm_encoder_name=self.fm_encoder_name,
            model_input_size=self.max_len,
        )

        gp_transformer = self._init_trainer(
            return_attention=True,
            gp=gp,
            num_virtual_tokens=self.num_virtual_tokens,
            split_label=split,
            token_to_gene_to_keep_dict=token_to_gene_to_keep_dict,
        )

        trainer = pl.Trainer(
            max_epochs=1, devices=1, accelerator='auto', precision=precision
        )

        trainer.test(gp_transformer, txdata)

    def test_random_baseline(self, adata_path):
        '''
        Compare Pearson and MSE of count reconstruction for true cells vs random cells
        '''

        # to do check reconstruction loss
        if (self.model_type != 'Global') & (self.model.global_loss != 'reconstruction'):
            raise ValueError(
                'Random baseline only implemented for reconstruction '
                'loss from global cell token,'
                f'not {self.model_type}, {self.model.global_loss}'
            )

        # Initialize trainer
        os.chdir(self.output_dir)

        print('Dataset path', self.dataset_path)

        txdata = txDataModule(
            folder=self.dataset_path,
            batch_size=self.batch_size,
            adata_path=adata_path,
            seed=self.seed,
            fm_encoder_name=self.fm_encoder_name,
            model_input_size=self.max_len,
        )

        gp_transformer = self._init_trainer(
            test_random_baseline=True, num_virtual_tokens=self.num_virtual_tokens
        )
        trainer = pl.Trainer(max_epochs=1, devices=1, accelerator='auto', precision=32)
        trainer.test(gp_transformer, txdata)

    def evaluate_supervised_model(self):
        txdata = txDataModule(
            folder=self.dataset_path,
            batch_size=self.batch_size,
            seed=self.seed,
            fm_encoder_name=self.fm_encoder_name,
            model_input_size=self.max_len,
        )

        trainer = pl.Trainer(max_epochs=1, devices=1, accelerator='auto', precision=32)
        trainer.test(self.gp_transformer, txdata)

    def generate_virtual_tokens(self, split='test'):
        '''
        Extract virtual tokens
        '''
        gp_transformer = self._init_trainer(
            split_label=split,
            hparam_save=self.hparam_save,
            num_virtual_tokens=self.num_virtual_tokens,
            return_virtual_tokens=True,
        )

        txdata = txDataModule(
            folder=self.dataset_path,
            batch_size=self.batch_size,
            data_split_to_pass_to_test_step=split,
            seed=self.seed,
            fm_encoder_name=self.fm_encoder_name,
            model_input_size=self.max_len,
        )

        trainer = pl.Trainer(max_epochs=1, devices=1, accelerator='auto', precision=32)

        trainer.validate(gp_transformer, txdata)


################################
# Reference/query distance
################################


def calculate_gp_emd(
    data,
    source_key,
    ref_label,
    query_label,
    condition_key,
    output_dir,
    filename=None,
    gp=None,
    gpdb=None,
    filtering_dict=None,
):
    '''
    Calculate EMD between reference and query distributions

    '''

    if isinstance(data, str):
        if data.endswith('.h5ad'):
            data = sc.read(data)
        else:
            data = load_from_disk(data)

    if (gpdb is None) and (gp is None):
        raise ValueError('Please provide either gp or gpdb')

    if gp is None:
        gpdb = pd.read_csv(gpdb)
        gpx = list(gpdb.columns)
    elif isinstance(gp, str):
        gpx = [gp]
    else:
        gpx = gp

    holder = []

    for gp in gpx:
        if filtering_dict is None:
            adata = sc.AnnData(
                X=np.array(data[gp]),
                obs=data.select_columns(
                    list(set([source_key, condition_key]))
                ).to_pandas(),
            )

        else:
            adata = sc.AnnData(
                X=np.array(data[gp]),
                obs=data.select_columns(
                    list(set([source_key, condition_key])) + list(filtering_dict.keys())
                ).to_pandas(),
            )

            for k, v in filtering_dict.items():
                if k not in adata.obs.columns:
                    print(f'Key {k} not found in adata.obs columns. Skipping.')  # noqa
                    continue
                if isinstance(v, list):
                    adata = adata[adata.obs[k].isin(v)]
                else:
                    adata = adata[adata.obs[k] == v]

        # get reference distribution
        ref = adata[adata.obs[source_key] == ref_label]
        query = adata[adata.obs[source_key] == query_label]

        # calculate EMD
        emd_df = evaluate_emd_ref_vs_query(ref, query, condition_key, condition_key)
        emd_df['embedding'] = gp
        holder.append(emd_df)
        print('embdf', emd_df.head())

    output_df = pd.concat(holder)

    if filename is None:
        filename = f'{ref_label}_vs_{query_label}_emd.csv'

    output_df.to_csv(
        os.path.join(output_dir, filename),
        index=False,
    )

    return None


################################
# Grad-CAM
################################


def calculate_gp_attribution_scores(
    gpdb_path,
    dataset_path,
    data_split,
    gp_latent_size,
    model_checkpoint,
    obs_key,
    obs_value,
    output_dir,
    gp,
    block_n=-1,
    total_n_cells=None,
    task='classification',
    gpdb_ref_path=None,
    gene_format='symbol',
    emb_dataset_path=None,
    gene_counts_df=None,
    gp_inputs=None,
    model_type='Base',
    peft_config_path=None,
    fm_encoder_pkg='geneformer',
    fm_encoder_name='gf-6L-30M-i2048',
    output_file_name=None,
    model_input_size: int = 2048,
):
    '''
    Calculate attribution scores for each gene program
    '''

    # --------------------------
    # Set seed
    # --------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seed
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --------------------------
    # Set up Geneformer
    # --------------------------

    if fm_encoder_pkg == 'geneformer':
        gf_repo_path = get_gf_repo()
        geneformer_model = os.path.join(gf_repo_path, fm_encoder_name)

        if '4096' in fm_encoder_name:
            gene_token_path = TOKEN_DICTIONARY_FILE
            gene_name_path = ENSEMBL_DICTIONARY_FILE

        else:
            gene_token_path = os.path.join(
                gf_repo_path,
                'geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl',
            )

            gene_name_path = os.path.join(
                gf_repo_path,
                'geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl',
            )

    elif fm_encoder_pkg == 'from_scratch':
        gf_repo_path = get_gf_repo()
        if model_type == 'Base':
            gpformer = gpBase.load_from_checkpoint(
                model_checkpoint,
                strict=False,
                map_location='cpu',
            )
        elif model_type == 'Global':
            gpformer = gpGlobal.load_from_checkpoint(
                model_checkpoint, strict=False, map_location='cpu'
            )

        geneformer_model = gpformer.model.gf_wrapper

        if geneformer_model.model.config.max_position_embeddings == 4096:
            gene_token_path = TOKEN_DICTIONARY_FILE
            gene_name_path = ENSEMBL_DICTIONARY_FILE
        else:
            gene_token_path = os.path.join(
                gf_repo_path,
                'geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl',
            )

            gene_name_path = os.path.join(
                gf_repo_path,
                'geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl',
            )

    else:
        raise ValueError(
            'Please provide valid fm_encoder_pkg (geneformer or from_scratch)'
        )

    # --------------------------
    # Set up dataloader
    # --------------------------

    gpdb = pd.read_csv(gpdb_path)

    if gp_inputs is None:
        gp_inputs = list(gpdb.columns)
    elif isinstance(gp_inputs, str):
        gp_inputs = [gp]

    if gene_counts_df is not None:
        gene_counts_df = pd.read_csv(gene_counts_df)

    txdata = iTxDataModule(
        folder=dataset_path,
        batch_size=1,
        return_tuple=True,
        gp=gp,
        gpdb=gpdb,
        do_ensembl_conversion=(gene_format != 'ensembl'),
        filter_key=obs_key,
        filter_value=obs_value,
        geneformer_model=geneformer_model,
        peft_config_path=peft_config_path,
        gene_name_path=gene_name_path,
        gene_token_path=gene_token_path,
        model_input_size=model_input_size,
        fm_encoder_pkg=fm_encoder_pkg,
    )

    txdata.setup()
    dataloader = getattr(txdata, data_split + '_dataloader')()
    print('Number of cells', len(dataloader))

    if total_n_cells is None:
        total_n_cells = len(dataloader)

    y_label = obs_key

    # --------------------------
    # Set up model
    # --------------------------

    if model_type == 'Base':
        gp_transformer = gpBase.load_from_checkpoint(
            model_checkpoint,
            strict=False,
            map_location='cpu',
        )
    elif model_type == 'Global':
        gp_transformer = gpGlobal.load_from_checkpoint(
            model_checkpoint, strict=False, map_location='cpu'
        )

    elif model_type == 'Prototypes':
        gp_transformer = gpPrototypes.load_from_checkpoint(
            model_checkpoint, strict=False, map_location='cpu'
        )

    # Load classification layer
    # or train if not available
    ckpt_dir = os.path.join(output_dir, 'evaluation_model_checkpoints')
    clf_ckpt = f"{y_label.replace('_id', '')}_{gp}_{task}"

    if os.path.exists(os.path.join(ckpt_dir, f'{clf_ckpt}.ckpt')):
        clf_layer = EmbEvaluator.load_from_checkpoint(
            os.path.join(ckpt_dir, f'{clf_ckpt}.ckpt')
        )

    else:
        if emb_dataset_path is None:
            raise ValueError(
                'Please provided path to embeddings for training linear layer'
            )
        gpEval.evaluate_embeddings(
            y_label=y_label,
            encode_covariate=True,
            folder_path=emb_dataset_path,
            output_dir=output_dir,
            emb_label=gp,
            task=task,
            emb_dim=gp_latent_size,
            lr=1e-3,
            batch_size=128,
            num_workers=1,
            data_type='dataset',
            n_epochs=3,
            continuous_cov=[],
        )

        clf_layer = EmbEvaluator.load_from_checkpoint(
            os.path.join(ckpt_dir, f'{clf_ckpt}.ckpt')
        )

    imodel = iGpWrapper(
        gp_transformer,
        clf_layer,
        gp_of_interest=gp,
    )

    imodel = imodel.to(device)

    # --------------------------
    # Run attribution
    # --------------------------

    # set up attribution
    gc = GuidedGradCam(imodel, imodel.gp_block.blocks[block_n].mlp)

    attribution_scores: Dict[Any, Any] = {}
    all_tokens = set()
    counter = 0

    for b in tqdm(dataloader):
        if counter < total_n_cells:
            counter += 1
            emb = b[0].to(device)

            edict = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in b[1].items()
            }

            input_ids = (emb, edict)
            token_labels = edict['token_labels'].squeeze().cpu().numpy().tolist()
            for labels in token_labels:
                all_tokens.add(labels)

            attributions = gc.attribute(
                input_ids[0],
                target=edict[f'{obs_key}_id'],
                additional_forward_args=input_ids[1],
            )

            attr_norm = summarize_attributions(attributions).detach().cpu().numpy()

            for i, t in enumerate(token_labels):
                if t in attribution_scores.keys():
                    attribution_scores[t] += [attr_norm[i]]
                    attribution_scores[f'{t}_abs'] += [np.abs(attr_norm[i])]
                    attribution_scores[f'{t}_rank'] += [i]
                else:
                    attribution_scores[t] = [attr_norm[i]]
                    attribution_scores[f'{t}_abs'] = [np.abs(attr_norm[i])]
                    attribution_scores[f'{t}_rank'] = [i]

        else:
            break

    for t in all_tokens:
        attribution_scores[t] = np.nanmean(attribution_scores[t])
        attribution_scores[f'{t}_abs'] = np.nanmean(attribution_scores[f'{t}_abs'])
        attribution_scores[f'{t}_std'] = np.nanstd(attribution_scores[t])
        attribution_scores[f'{t}_abs_std'] = np.nanstd(attribution_scores[f'{t}_abs'])
        attribution_scores[f'{t}_rank'] = np.nanmean(attribution_scores[f'{t}_rank'])

    rows = []

    for t in all_tokens:
        row = {
            'token': t,
            'attribution_score': attribution_scores[t],
            'abs_attribution_score': attribution_scores[f'{t}_abs'],
            'std_attribution_score': attribution_scores[f'{t}_std'],
            'abs_std_attribution_score': attribution_scores[f'{t}_abs_std'],
            'rank': attribution_scores[f'{t}_rank'],
        }

        rows.append(row)

    attribution_df = pd.DataFrame(rows)

    # Add gene conversion
    gene_df = pd.DataFrame(imodel.gene_conversion)
    gene_df = gene_df.join(attribution_df.set_index('token'), on='token')

    # add GP labels
    if gpdb_ref_path is None:
        gpdb_ref_path = gpdb_path

    # add GP labels
    if gpdb_ref_path is None:
        gpdb_ref_path = gpdb_path

    gpdb_og = pd.read_csv(gpdb_ref_path)

    for ogp in gpdb_og.columns:
        if gene_format == 'symbol':
            gene_df[ogp] = np.where(gene_df['symbol'].isin(gpdb_og[ogp]), 1, 0)
        else:
            gene_df[ogp] = np.where(gene_df['ensembl'].isin(gpdb_og[ogp]), 1, 0)

    if output_file_name is None:
        output_file_name = f'{gp}_attribution_scores_{obs_value}_block_{block_n}.csv'

    gene_df.to_csv(
        os.path.join(output_dir, output_file_name),
        index=False,
    )


def calculate_cell_token_attribution_scores(
    gpdb_path,
    dataset_path,
    emb_dataset_path,
    data_split,
    gp_latent_size,
    model_checkpoint,
    obs_key,
    obs_value,
    output_dir,
    # for EmbEvaluator
    emb_label,
    task,
    encode_covariate=True,
    save_plot=False,
    gp_inputs=None,
    use_embedding=False,
    pretrained_emb=None,
    supervised_labels=None,
    block_n=-1,
):
    # --------------------------
    # Set seed
    # --------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seed
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --------------------------
    # Set up dataloader
    # --------------------------

    gpdb = pd.read_csv(gpdb_path)

    if gp_inputs is None:
        gp_inputs = list(gpdb.columns)

    emb_dm = iEmbDataModule(
        folder_path=emb_dataset_path,
        batch_size=1,
        gp_inputs=gp_inputs,
        meta_labels=obs_key,
        clf_label=obs_key,
        encode_covariate=encode_covariate,
    )

    emb_dm.setup()

    dataloader = getattr(emb_dm, data_split + '_dataloader')()

    # --------------------------
    # Build dictionary for class : id conversion
    # --------------------------

    y_label = obs_key + '_id'

    if encode_covariate:
        datax = load_from_disk(dataset_path)
        labels = datax.unique(obs_key)
        conversion_dict = {k: i for i, k in enumerate(labels)}

    else:
        datax = load_from_disk(dataset_path)
        datax = datax.select_columns([y_label, obs_key])
        conversion = datax.to_pandas().drop_duplicates()
        conversion_dict = {
            k: v for k, v in zip(conversion[obs_key], conversion[y_label])
        }

    # --------------------------
    # Set up model
    # --------------------------

    gp_transformer = gpGlobal.load_from_checkpoint(
        model_checkpoint,
        strict=False,
        map_location='cpu',
    )

    # Load classification layer
    # or train if not available
    if gp_transformer.model.global_loss != 'supervised':
        ckpt_dir = os.path.join(output_dir, 'evaluation_model_checkpoints')
        clf_ckpt = f"{y_label.replace('_id', '')}_{emb_label}_{task}"
        if os.path.exists(os.path.join(ckpt_dir, f'{clf_ckpt}.ckpt')):
            clf_layer = EmbEvaluator.load_from_checkpoint(
                os.path.join(ckpt_dir, f'{clf_ckpt}.ckpt')
            )

        else:
            gpEval.evaluate_embeddings(
                y_label=obs_key,
                folder_path=emb_dataset_path,
                output_dir=output_dir,
                emb_label=emb_label,
                task=task,
                emb_dim=gp_latent_size,
                lr=1e-3,
                batch_size=128,
                num_workers=1,
                data_type='dataset',
                n_epochs=3,
                continuous_cov=[],
                encode_covariate=encode_covariate,
            )

            clf_layer = EmbEvaluator.load_from_checkpoint(
                os.path.join(ckpt_dir, f'{clf_ckpt}.ckpt')
            )

        task_index = None
    else:
        clf_layer = None
        tasks = list(supervised_labels.keys())
        if not obs_key.endswith('_id'):
            task_tag = obs_key + '_id'
        else:
            task_tag = obs_key
        task_index = tasks.index(task_tag)

    imodel = iGlobalWrapper(
        gp_transformer,
        clf_layer,
        global_loss=gp_transformer.model.global_loss,
        use_embedding=use_embedding,
        pretrained_emb=pretrained_emb,
        vocab_size=len(gpdb.columns),
        embedding_dim=gp_latent_size,
        task_index=task_index,
    )
    imodel = imodel.to(device)

    # --------------------------
    # Run attribution
    # --------------------------

    # set up attribution
    gc = GuidedGradCam(imodel, imodel.global_block.encoder.blocks[block_n].mlp)

    attribution_scores = {}

    for g in gp_inputs:
        attribution_scores[g] = []
        attribution_scores[f'{g}_abs'] = []

    for b in tqdm(dataloader):
        if b[1][obs_key] == [obs_value]:
            emb = b[0].to(device)

            edict = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in b[1].items()
            }

            input_ids = (emb, edict)

            attributions = gc.attribute(
                input_ids[0],
                target=conversion_dict[obs_value],
                additional_forward_args=input_ids[1],
            )

            attr_norm = summarize_attributions(attributions).detach().cpu().numpy()

            for i, g in enumerate(gp_inputs):
                attribution_scores[g] += [attr_norm[i]]
                attribution_scores[f'{g}_abs'] += [abs(attr_norm[i])]

    for g in gp_inputs:
        attribution_scores[g] = np.nanmean(np.array(attribution_scores[g]))
        attribution_scores[f'{g}_std'] = np.nanstd(np.array(attribution_scores[g]))

        # Absolute values
        attribution_scores[f'{g}_abs'] = np.nanmean(
            np.array(attribution_scores[f'{g}_abs'])
        )
        attribution_scores[f'{g}_abs_std'] = np.nanstd(
            np.array(attribution_scores[f'{g}_abs'])
        )

    rows = []
    for g in gp_inputs:
        row = {
            'GP': g,
            'scores': attribution_scores[g],
            'scores_abs': attribution_scores[f'{g}_abs'],
            # 'score_std': attribution_scores[f'{g}_std'],
            # 'score_abs_std': attribution_scores[f'{g}_abs_std'],
        }
        rows.append(row)

    # Convert the list of dictionaries to a DataFrame
    attribution_df = pd.DataFrame(rows)

    attribution_df.to_csv(
        os.path.join(
            output_dir, f'cell_token_attribution_scores_{obs_value}_block_{block_n}.csv'
        ),
        index=False,
    )

    if save_plot:
        # Sorting the DataFrame by 'scores_abs' in descending order
        attribution_df_sorted_abs = attribution_df.sort_values(
            by='scores_abs', ascending=False
        )

        plt.figure()
        ax = sns.barplot(attribution_df_sorted_abs, x='GP', y='scores_abs')

        ax.set_title(obs_value.capitalize().replace('_', ' '))
        ax.set_ylabel('Absolute attribution score')
        ax.set_xlabel('Gene Program')

        # Rotating x-tick labels 90 degrees
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir, f'cell_token_abs_attribution_scores_{obs_value}.pdf'
            )
        )
        plt.close()

        # Sorting the DataFrame by 'scores' in descending order
        attribution_df_sorted = attribution_df.sort_values(by='scores', ascending=False)

        plt.figure()
        ax = sns.barplot(attribution_df_sorted, x='GP', y='scores')

        ax.set_title(obs_value.capitalize().replace('_', ' '))
        ax.set_ylabel('Attribution score')
        ax.set_xlabel('Gene Program')

        # Rotating x-tick labels 90 degrees
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        plt.tight_layout()

        plt.savefig(
            os.path.join(output_dir, f'cell_token_attribution_scores_{obs_value}.pdf')
        )
        plt.close()


def visualize_with_gene_exp(
    output_dir,
    adata_path,
    gene_name,
    gp_to_plot,
    data_to_plot='test',
    label_to_plot=None,
    subsample=None,
    obs_key=None,
    obs_value=None,
    obs_key2=None,
    obs_value2=None,
    return_adata=False,
    scale=False,
):
    """
    UMAP of GP embeddings
    """
    os.chdir(output_dir)

    if label_to_plot is not None and isinstance(label_to_plot, str):
        label_to_plot = [label_to_plot]

    if isinstance(gp_to_plot, str):
        gp_to_plot = [gp_to_plot]

    if obs_value is not None and not isinstance(obs_value, list):
        obs_value = [obs_value]

    if obs_value2 is not None and not isinstance(obs_value2, list):
        obs_value2 = [obs_value2]

    emb = load_from_disk(os.path.join(output_dir, f'embeddings/{data_to_plot}_set'))

    if subsample is not None:
        emb = emb.shuffle(seed=0).select(range(subsample))

    gene_exp = sc.read_h5ad(adata_path)
    gene_exp = gene_exp[:, gene_exp.var.index == gene_name]

    for gp in gp_to_plot:
        x = np.array(emb[gp])

        var_to_keep = ['idx']
        if label_to_plot is not None:
            var_to_keep += label_to_plot
        if obs_key is not None:
            var_to_keep.append(obs_key)
        if obs_key2 is not None:
            var_to_keep.append(obs_key2)

        y = pd.DataFrame({k: emb[k] for k in emb.column_names if k in var_to_keep})

        adata = sc.AnnData(X=x, obs=y)

        # set obs name
        adata.obs = adata.obs.set_index('idx')

        if obs_key is not None:
            adata = adata[adata.obs[obs_key].isin(obs_value)]

        if obs_key2 is not None:
            adata = adata[adata.obs[obs_key2].isin(obs_value2)]

        gx = gene_exp[adata.obs.index, :]

        # Optionally scale expr to 0-1
        if scale:
            gx.X = MinMaxScaler().fit_transform(gx.X.toarray())

        adata.obs[f'{gene_name}_exp'] = gx.X.toarray().flatten()

        if label_to_plot is not None:
            for c in label_to_plot:
                adata = remove_single_data_points(adata, c)

        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.umap(adata)

        plot_tag = '_'.join(obs_value) if obs_value is not None else ''
        plot_tag += '_'.join(obs_value2) if obs_value2 is not None else ''

        # Set up color palette as in
        # https://scanpy-tutorials.readthedocs.io/en/latest/plotting/advanced.html#colors
        vmin = adata.obs[f'{gene_name}_exp'].min()
        vmax = adata.obs[f'{gene_name}_exp'].max()
        vpadding = (vmax - vmin) * 0.1
        norm = MidpointNormalize(vmin=vmin - vpadding, vmax=vmax + vpadding, midpoint=0)

        # Plot umap
        fig = sc.pl.umap(
            adata,
            color=f'{gene_name}_exp',
            cmap='coolwarm',
            # s=20,
            norm=norm,
            return_fig=True,
            show=False,
            frameon=False,
        )

        cmap_yticklabels = np.array([t._y for t in fig.axes[1].get_yticklabels()])
        fig.axes[1].set_ylim(
            0,  # for normalized gene expression
            min(cmap_yticklabels[cmap_yticklabels > vmax]),
        )

        # Save the figure as a PDF
        fig.savefig(
            os.path.join(
                output_dir, f'figures/umap_{gene_name}_exp_in_{gp}{plot_tag}.pdf'
            ),
            format='pdf',
        )

        if label_to_plot is not None:
            for c in label_to_plot:
                sc.pl.umap(
                    adata,
                    color=c,
                    save=f'_{gp}_by_{c}{plot_tag}.pdf',
                    frameon=False,
                )

    if return_adata:
        return adata


def calc_eval_metrics(
    train_set,
    test_set,
    gp,
    label,
    output_dir,
    k=20,
    data_type='dataset',
    task='classification',
    normalize=False,
    model_type='knn',
):
    np.random.seed(0)
    random.seed(0)

    # Extract features and labels based on the data type
    if data_type == 'dataset':
        X_train = train_set[gp]
        X_test = test_set[gp]
        y_train = train_set[label]
        y_test = test_set[label]
    elif data_type == 'h5ad':
        if gp != 'cell_token':
            X_train = train_set[:, train_set.var.index.str.contains(gp, case=False)].X
            X_test = test_set[:, test_set.var.index.str.contains(gp, case=False)].X
            if normalize:
                X_train = X_train / X_train.sum(axis=1)[:, None]
                X_test = X_test / X_test.sum(axis=1)[:, None]
        else:
            X_train = train_set.X
            X_test = test_set.X
            if normalize:
                X_train = X_train / X_train.sum(axis=1)[:, None]
                X_test = X_test / X_test.sum(axis=1)[:, None]
        y_train = train_set.obs[label]
        y_test = test_set.obs[label]

    # Initialize the model based on task and model type
    if model_type == 'knn':
        if task == 'classification':
            model = KNeighborsClassifier(n_neighbors=k)
        elif task == 'regression':
            model = KNeighborsRegressor(n_neighbors=k)
    elif model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'logistic':
        if task != 'classification':
            raise ValueError(
                'Logistic regression can only be used for classification tasks.'
            )
        model = LogisticRegression(class_weight='balanced')

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate based on the task
    if task == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        report = classification_report(y_test, y_pred, output_dict=True)
        output_df = wrangle_classification_report(report)
        output_df = output_df[
            ~output_df['output_class'].isin(['macro avg', 'weighted avg'])
        ]

        # Wrangle labels
        if data_type == 'dataset':
            label_conversion = (
                test_set.select_columns([label, label.replace('_id', '')])
                .to_pandas()
                .drop_duplicates()
            )
        elif data_type == 'h5ad':
            label_conversion = test_set.obs[
                [label, label.replace('_id', '')]
            ].drop_duplicates()

        conversion_dict = {
            str(k): v
            for k, v in zip(
                label_conversion[label], label_conversion[label.replace('_id', '')]
            )
        }

        output_df['output_class'] = output_df['output_class'].astype(str)
        output_df['output_class'] = output_df['output_class'].map(conversion_dict)

        # Save the classification output
        os.makedirs(output_dir, exist_ok=True)
        print('Saving to', os.path.join(output_dir, f'{label}_from_{gp}.csv'))
        output_df.to_csv(
            os.path.join(output_dir, f'{label}_from_{gp}.csv'), index=False
        )

    elif task == 'regression':
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'Mean Squared Error (MSE): {mse:.2f}')
        print(f'Mean Absolute Error (MAE): {mae:.2f}')
        print(f'R-squared (R2): {r2:.2f}')

        metrics = {
            'Mean Squared Error': mse,
            'Mean Absolute Error': mae,
            'R-squared': r2,
        }
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(
            output_dir, f'{label}_regression_metrics_from_{gp}.csv'
        )
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}, {value}\n')

    return None


# --------------------------
# Gene embeddings analysis
# --------------------------


def calculate_gene_to_gp_cosine_similarity(
    gene_data,
    cell_data,
    gp,
    genes,
):
    # 1. Extract cell embeddings
    print('Exctracting cell data')
    cell_idx = cell_data['idx']
    cell = np.asarray(cell_data[gp])

    # 2. Extract gene embeddings
    print(
        'Extracting gene data'
        ' (depending on the size of your data, this can take a few minutes)'
    )
    gene_idx = gene_data['idx']
    gene_arrays = {
        gene: np.asarray(gene_data[gene])
        for gene in genes
        if gene in gene_data.column_names
    }

    # 3. Align indices
    print('Aligning indices')
    shared_indices, cell, gene_arrays = align_indices(
        cell_idx, gene_idx, cell, gene_arrays
    )

    # 4. Calculate cosine similarity
    print('Calculating cosine similarity')
    cosim_df = pd.DataFrame(index=shared_indices, columns=gene_arrays.keys())

    for gene, gene_arr in gene_arrays.items():
        cosim_df[gene] = [1 - cosine(gp, g) for gp, g in zip(cell, gene_arr)]

    return cosim_df


def pivot_cosine_similarity_data_longer(
    cosim_df,
    cell_data,
    meta_cols,
    missing_gene_threshold=None,
    fillna=False,
):
    # Remove genes with too many missing values
    if missing_gene_threshold:
        col_to_drop = []

        for g in cosim_df.columns:
            if (
                len(cosim_df[cosim_df[g].isna()])
                > missing_gene_threshold * cosim_df.shape[0]
            ):
                col_to_drop.append(g)

        print('Number of columns to drop', len(col_to_drop))

        cosim_df = cosim_df.drop(columns=col_to_drop)

    if 'idx' not in meta_cols:
        meta_cols = ['idx'] + meta_cols

    obs_df = cell_data.select_columns(meta_cols).to_pandas()

    merged_df = cosim_df.join(obs_df.set_index('idx'))
    merged_df['idx'] = merged_df.index.values

    # Wrangling dataframe into long format
    df_long = merged_df.melt(
        id_vars=meta_cols, var_name='gene', value_name='cosine_sim'
    )

    return df_long


def calculate_gene_significance(ref_data, query_data):
    """Calculate p-values and significance levels for all genes."""
    all_genes = ref_data['gene'].unique()
    results = []
    for gene in all_genes:
        ref_samples = ref_data[ref_data['gene'] == gene]['cosine_sim']
        query_samples = query_data[query_data['gene'] == gene]['cosine_sim']
        if len(ref_samples) > 1 and len(query_samples) > 1:
            t_stat, p_value = ttest_ind(ref_samples, query_samples, equal_var=True)
            mean_ref = ref_samples.mean()
            mean_query = query_samples.mean()
            effect_size = mean_ref - mean_query
            results.append(
                {
                    'gene': gene,
                    'p_value': p_value,
                    'mean_ref': mean_ref,
                    'mean_query': mean_query,
                    'effect_size': effect_size,
                }
            )

            # Create DataFrame for all genes
    results_df = pd.DataFrame(results)

    # Apply Benjamini-Hochberg correction
    results_df['p_adjusted'] = multipletests(
        results_df['p_value'], alpha=0.05, method='fdr_bh'
    )[1]

    # Add significance stars
    results_df['significance'] = ''
    for i, p_value in enumerate(results_df['p_adjusted']):
        if not np.isnan(p_value):
            if p_value < 0.001:
                results_df.loc[i, 'significance'] = '***'
            elif p_value < 0.01:
                results_df.loc[i, 'significance'] = '**'
            elif p_value < 0.05:
                results_df.loc[i, 'significance'] = '*'

    return results_df


def assign_bar_colors(genes, gp_to_color, gpdb):
    colors = []  # List to store colors for each gene
    for gene in genes:
        color_assigned = 'gray'  # Default color if gene is not found
        for gp, color in gp_to_color.items():
            if gene in gpdb[gp].values:
                color_assigned = color
                break
        colors.append(color_assigned)
    return colors


def visualize_cosine_similarity(
    df_long,
    obs_col,
    obs_value_1,
    obs_value_2,
    fillna=False,
    topn=10,
    save_to=None,
    gp_to_color=None,  # {GP : color}
    gpdb=None,  # for gene membership
):
    # Optionally fill in missing values
    if fillna:
        df_long = df_long.fillna(0)

    # Filter data for ref and query lineages
    ref = df_long[df_long[obs_col] == obs_value_1]
    query = df_long[df_long[obs_col] == obs_value_2]

    # Calculate significance for all genes
    all_gene_stats = calculate_gene_significance(ref, query)

    # Select top 10 genes with highest average cosine similarity in ref and query
    ref_top10 = all_gene_stats.nlargest(topn, 'mean_ref')
    query_top10 = all_gene_stats.nlargest(topn, 'mean_query')

    # Select top 10 genes with strongest effect size differences
    top10_diff_ref = all_gene_stats.nlargest(topn, 'effect_size')  # ref > query
    top10_diff_query = all_gene_stats.nsmallest(topn, 'effect_size')  # query > ref

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    fig.set_facecolor('white')

    # 1. Top 10 genes with highest average cosine similarity in ref
    if gp_to_color:
        palette = dict(
            zip(
                ref_top10['gene'],
                assign_bar_colors(ref_top10['gene'], gp_to_color, gpdb),
            )
        )
        sns.barplot(
            data=ref_top10,
            x='mean_ref',
            y='gene',
            ax=axs[0, 0],
            errorbar=None,
            palette=palette,
        )
    else:
        sns.barplot(data=ref_top10, x='mean_ref', y='gene', ax=axs[0, 0], errorbar=None)

    axs[0, 0].set_title(
        f'Top 10 Genes with Highest Average Cosine Similarity ({obs_value_1})'
    )
    axs[0, 0].set_xlabel('Cosine Similarity')
    axs[0, 0].set_ylabel('Gene')

    # 2. Top 10 genes with highest average cosine similarity in query
    if gp_to_color:
        palette = dict(
            zip(
                query_top10['gene'],
                assign_bar_colors(query_top10['gene'], gp_to_color, gpdb),
            )
        )
        sns.barplot(
            data=query_top10,
            x='mean_query',
            y='gene',
            ax=axs[0, 1],
            errorbar=None,
            palette=palette,
        )
    else:
        sns.barplot(
            data=query_top10, x='mean_query', y='gene', ax=axs[0, 1], errorbar=None
        )

    axs[0, 1].set_title(
        f'Top 10 Genes with Highest Average Cosine Similarity ({obs_value_2})'
    )
    axs[0, 1].set_xlabel('Cosine Similarity')
    axs[0, 1].set_ylabel('Gene')

    # 3. Top 10 genes with strongest difference (ref > query)
    if gp_to_color:
        palette = dict(
            zip(
                top10_diff_ref['gene'],
                assign_bar_colors(top10_diff_ref['gene'], gp_to_color, gpdb),
            )
        )
        sns.barplot(
            data=top10_diff_ref,
            x='effect_size',
            y='gene',
            ax=axs[1, 0],
            errorbar=None,
            palette=palette,
        )
    else:
        sns.barplot(
            data=top10_diff_ref, x='effect_size', y='gene', ax=axs[1, 0], color='salmon'
        )

    for i, (effect, gene, significance) in enumerate(
        zip(
            top10_diff_ref['effect_size'],
            top10_diff_ref['gene'],
            top10_diff_ref['significance'],
        )
    ):
        if significance:
            axs[1, 0].text(
                effect + 0.0002,
                i,
                significance,
                ha='left',
                va='center',
                fontsize=12,
                color='darkred',
            )
    axs[1, 0].set_title(
        f'Top 10 Genes with Strongest Difference ({obs_value_1} > {obs_value_2})'
    )
    axs[1, 0].set_xlabel('Difference in Cosine Similarity')
    axs[1, 0].set_ylabel('Gene')

    # 4. Top 10 genes with strongest difference (query > ref)
    if gp_to_color:
        palette = dict(
            zip(
                top10_diff_query['gene'],
                assign_bar_colors(top10_diff_query['gene'], gp_to_color, gpdb),
            )
        )
        sns.barplot(
            data=top10_diff_query,
            x='effect_size',
            y='gene',
            ax=axs[1, 1],
            errorbar=None,
            palette=palette,
        )
    else:
        sns.barplot(
            data=top10_diff_query,
            x='effect_size',
            y='gene',
            ax=axs[1, 1],
            color='skyblue',
        )

    for i, (effect, gene, significance) in enumerate(
        zip(
            top10_diff_query['effect_size'],
            top10_diff_query['gene'],
            top10_diff_query['significance'],
        )
    ):
        if significance:
            axs[1, 1].text(
                effect - 0.0002,
                i,
                significance,
                ha='right',
                va='center',
                fontsize=12,
                color='navy',
            )
    axs[1, 1].set_title(
        f'Top 10 Genes with Strongest Difference ({obs_value_2} > {obs_value_1})'
    )
    axs[1, 1].set_xlabel('Difference in Cosine Similarity')
    axs[1, 1].set_ylabel('Gene')

    # Adjust layout to avoid squishing
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    if save_to:
        plt.savefig(save_to)

    plt.show()


# -------------------------------------------
# Phate
# -------------------------------------------


def compute_phate(adata, n_components=2):
    phate_operator = phate.PHATE(n_components=n_components)
    data_phate = phate_operator.fit_transform(adata.X)
    return data_phate


def plot_phate(data_phate, adata, label, label_order, output_file, color_map='Paired'):
    # Desired fixed order of categories
    fixed_order = [c for c in label_order if c in adata.obs[label].unique()]

    # Ensure 'ct_broad' is a simple Series (not MultiIndex)
    cell_label = adata.obs[label].values  # Extract as a 1D array

    # Convert 'ct_broad' to a pandas Categorical with the fixed order
    cell_label = pd.Categorical(cell_label, categories=fixed_order, ordered=True)

    # Get unique categories based on the fixed order
    unique_categories = fixed_order

    # Generate a colormap and normalize it based on the number of unique categories
    if color_map is None:
        cmap = plt.get_cmap()  # Use default colormap
    else:
        cmap = plt.get_cmap(color_map)
    norm = plt.Normalize(vmin=0, vmax=len(unique_categories) - 1)

    # Map each category to a color
    colors = [cmap(norm(i)) for i in range(len(unique_categories))]

    # Create a dictionary mapping categories to colors
    category_color_map = dict(zip(unique_categories, colors))

    # Apply the colors in the scatter plot
    # Convert 'ct_broad' to color labels based on the category_color_map
    color_labels = np.array([category_color_map[category] for category in cell_label])

    scprep.plot.scatter2d(
        data_phate,
        c=color_labels,  # Use the mapped colors directly
        ticks=False,
        legend=False,
    )  # Disable automatic legend

    # Manually create the legend with the correct colors and fixed order
    for category, color in category_color_map.items():
        plt.scatter([], [], label=category, color=color)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Save the plot as a PDF
    plt.savefig(output_file, format='pdf', bbox_inches='tight')  # Save plot to PDF

    # Show plot
    plt.show()
