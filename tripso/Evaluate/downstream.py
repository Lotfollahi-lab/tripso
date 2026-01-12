import os
import random
import warnings
from typing import (
    Dict,
    Literal,
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
import scipy.sparse as sp
import scprep
import seaborn as sns
import torch
from datasets import load_from_disk
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    to_rgba,
)
from scipy.sparse import issparse
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

try:
    # Check if running in a Jupyter notebook
    from IPython import get_ipython

    if 'IPKernelApp' in get_ipython().config:
        from tqdm.notebook import tqdm  # noqa: F401
    else:
        from tqdm import tqdm  # noqa: F401
except (ImportError, AttributeError):
    # Default to regular tqdm in case of any issues
    pass

from ..Datamodules.datamodule import txDataModule
from ..Metrics.metrics import evaluate_emd_ref_vs_query
from ..Models.baselines import gfGlobal
from ..Trainers.trainer import (
    gpAblation,
    gpBase,
    gpGlobal,
    gpGlobalLoRA,
)
from ..Utils.utils import (
    MidpointNormalize,
    _bh_with_nans,
    _resolve_sig_colors,
    _stars,
    _t_equal_var_sparse,
    assign_bar_colors,
    build_token_to_gene_name_dict,
    remove_single_data_points,
    wrangle_classification_report,
)

# for exporting pdfs
matplotlib.rcdefaults()
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
    output_dir : str
        Path to directory where we will save outputs
        and where model checkoints are stored
    batch_size : int
        Batch size for evaluation step
    n_blocks : int
        Number of transformer blocks
    gene_format : str
        Format in which gene names are stored in GPDB
        One of 'symbol' or 'ensembl'
    tissue : str
        Tissue name for logging experiment in wandb
        This is also present in model checkpoint name
    model_type : str
        Base (GP blocks only) or Global (with cell token)
    n_heads : int
        Number of heads for multi-head attention
    gp_latent_size : int
        Size of latent space for GP tokens
    gp_inputs : list
        Which GP from GPDB to include in model
        if None, defaults to all GP
    supervised_labels : list
        Dict {label : num_classes} for supervised classification
    global_attn_heads : int
        number of heads for learning cell token in global attention model
    global_loss :
        loss used to train global attention model
        (for compatibility with gpGlobal init)

    """

    def __init__(
        self,
        gpdb_path: Optional[str] = None,
        output_dir: str = '/path/to/output/',
        dataset_path: Optional[str] = None,
        tissue: Optional[str] = 'test',
        model_type: Optional[str] = 'Base',
        batch_size: Optional[int] = 128,
        path_to_trained_model: Optional[str] = None,
        seed: Optional[int] = 0,
        hparam_save: Optional[str] = 'all',
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
            if path_to_trained_model is None:
                model_path = output_dir
            else:
                model_path = path_to_trained_model

            # latest_ckpt = find_latest_file(model_path, tissue, tag)
            latest_ckpt = os.path.join(model_path, 'checkpoints/last.ckpt')

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
        )

    def _init_trainer(
        self,
        return_gene_embeddings=False,
        return_gene_cosim=None,
        tokens_to_keep=None,
        genes_to_keep=None,
        gene_dir_tag=None,
        return_attention=False,
        gp=None,
        gp_for_downstream=None,
        return_classification_report=False,
        test_random_baseline=False,
        save_emb=False,
        split_label=None,
        hparam_save='ignore_model',  # fine for test time?
        return_virtual_tokens=False,
        token_to_gene_to_keep_dict=None,
        return_mean_non_padding=False,
    ):
        if self.model_type == 'Base':
            gp_transformer = gpBase.load_from_checkpoint(
                self.checkpoint_path, hparam_save=hparam_save, map_location='cpu'
            )

        elif self.model_type == 'Global':
            gp_transformer = gpGlobal.load_from_checkpoint(
                self.checkpoint_path, hparam_save=hparam_save, map_location='cpu'
            )

        elif self.model_type == 'Global_LoRA':
            gp_transformer = gpGlobalLoRA.load_from_checkpoint(
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
        gp_transformer.return_gene_cosim = return_gene_cosim
        gp_transformer.tokens_to_keep = tokens_to_keep
        gp_transformer.genes_to_keep = genes_to_keep
        gp_transformer.token_to_gene_to_keep_dict = token_to_gene_to_keep_dict
        gp_transformer.gene_dir_tag = gene_dir_tag
        gp_transformer.return_attention = return_attention
        gp_transformer.gp = gp
        gp_transformer.gp_for_downstream = gp_for_downstream
        gp_transformer.return_classification_report = return_classification_report
        gp_transformer.output_dir = self.output_dir
        gp_transformer.test_random_baseline = test_random_baseline
        gp_transformer.save_emb = save_emb
        gp_transformer.split_label = split_label
        gp_transformer.return_virtual_tokens = return_virtual_tokens
        gp_transformer.return_mean_non_padding = return_mean_non_padding

        gp_transformer.model.cond_to_shift = self.cond_to_shift

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
            # TO DO --> flexibly account for different model sizes
            self.max_len = 4096

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

        # Freeze all parameters for LoRA model
        if self.model_type == 'Global_LoRA':
            print('\nFreezing parameters for LoRA model\n')
            for param in self.model.parameters():
                param.requires_grad = False

        return gp_transformer

    def generate_embeddings(
        self, split='train', precision=32, return_mean_non_padding=False
    ):
        '''
        Save embeddings as Dataset
        '''

        gp_transformer = self._init_trainer(
            save_emb=True,
            split_label=split,
            hparam_save=self.hparam_save,
            return_mean_non_padding=return_mean_non_padding,
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

    def visualize(
        self,
        label_to_plot,
        data_to_plot='test',
        gp_to_plot=None,
        subsample=None,
        method: Literal['umap', 'pca'] = 'umap',
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

            if method == 'umap':
                sc.pp.neighbors(adata, use_rep='X', n_neighbors=15)
                sc.tl.umap(adata)

                for c in label_to_plot:
                    sc.pl.umap(
                        adata,
                        color=c,
                        save=f'_{self.tissue}_{gp}_by_{c}.pdf',
                        frameon=False,
                    )

            elif method == 'pca':
                sc.pp.pca(adata)

                for c in label_to_plot:
                    sc.pl.pca(
                        adata,
                        color=c,
                        save=f'_{self.tissue}_{gp}_by_{c}.pdf',
                        frameon=False,
                    )

            else:
                raise ValueError('method must be one of ["umap", "pca"].')

    def generate_gene_embeddings(
        self,
        gp_for_forward: Optional[str],
        gp_for_downstream: str,
        split='train',
        obs_key=None,
        obs_value=None,
        data_frac=1,
        genes_to_keep=None,
        output_tag=None,
        do_ensembl_conversion=True,
        precision=32,
        return_gene_cosim=None,
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
            value of obs_key to keep
        data_frac : float
            Fraction of data to use for generating embeddings
        gp_for_forward: str or None
            Pathway to use for model forward pass
            This is helpful if you only need to run forward pass
            on one GP rather than all of them.
        gp_for_downstream : str
            Pathway to use for focus of downstream analysis
        genes_to_keep : list
            Genes to generate embeddings for
            if None --> all genes
        return_gene_cosim:
            if None, get gene embeddings
            if gene_to_gp: anndata with (gene, GP) cosine similarity
            if gene_to_gene: matrix of mean (gene, gene) cosine similarity

        Use find_genes_in_multiple_gp or get_genes_in_single_gp from Utils.utils
        for GP selection
        """
        os.chdir(self.output_dir)

        gene_dir_tag = f'{gp_for_downstream}_gene_embeddings'

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
            return_gene_cosim=return_gene_cosim,
            gene_dir_tag=gene_dir_tag,
            tokens_to_keep=tokens_to_keep,
            genes_to_keep=tokens_to_keep,
            gp=gp_for_forward,
            gp_for_downstream=gp_for_downstream,
            split_label=split,
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
        gp_for_forward,
        gp_for_downstream,
        genes_to_keep=None,
        do_ensembl_conversion=True,
        split='test',
        precision=32,
    ):
        """
        Get attention weights from gpTransformer
        """
        os.chdir(self.output_dir)

        if (gp_for_downstream != 'cell_token') and (
            gp_for_downstream not in self.gp_inputs
        ):
            raise ValueError(
                f'{gp_for_downstream} must be one of "cell_token" or {self.gp_inputs}'
            )

        if gp_for_downstream != 'cell_token':
            _, token_to_gene_to_keep_dict = build_token_to_gene_name_dict(
                self.gp_transformer.model.gene_name_path,
                self.gp_transformer.model.gene_token_path,
                genes_to_keep,
                do_ensembl_conversion,
            )
        else:
            token_to_gene_to_keep_dict = None

        # Initialize trainer
        gp_transformer = self._init_trainer(
            return_attention=True,
            gp=gp_for_forward,
            gp_for_downstream=gp_for_downstream,
            split_label=split,
            token_to_gene_to_keep_dict=token_to_gene_to_keep_dict,
        )

        txdata = txDataModule(
            folder=self.dataset_path,
            batch_size=self.batch_size,
            data_split_to_pass_to_test_step=split,
            fm_encoder_name=self.fm_encoder_name,
            model_input_size=self.max_len,
        )

        trainer = pl.Trainer(
            max_epochs=1, devices=1, accelerator='auto', precision=precision
        )

        trainer.test(gp_transformer, txdata)

    def evaluate_supervised_model(self, precision=32):
        txdata = txDataModule(
            folder=self.dataset_path,
            batch_size=self.batch_size,
            seed=self.seed,
            fm_encoder_name=self.fm_encoder_name,
            model_input_size=self.max_len,
        )

        trainer = pl.Trainer(
            max_epochs=1, devices=1, accelerator='auto', precision=precision
        )
        trainer.test(self.gp_transformer, txdata)


class gpAblationEval(gpEval):
    """
    Evaluation class for gene program ablation studies.

    Extends gpEval to support ablation analysis by generating embeddings with
    individual gene programs removed. Can compute cosine similarity between
    ablated and full embeddings, or calculate changes in reconstruction loss.

    Parameters
    ----------
    main_ckpt_dir : str
        Path to directory containing the main model checkpoint (last.ckpt).
    compute_cosine : bool, default=False
        Whether to compute cosine similarity between ablated and full embeddings.
    compute_delta_nb_loss : bool, default=False
        Whether to compute change in negative binomial loss after ablation.
    adata_path : str, optional
        Path to .h5ad file for count reconstruction analysis. Required if
        compute_delta_nb_loss=True.
    *args
        Additional positional arguments passed to parent gpEval class.
    **kwargs
        Additional keyword arguments passed to parent gpEval class.

    Attributes
    ----------
    main_ckpt_dir : str
        Full path to the checkpoint file.
    compute_cosine : bool
        Flag for cosine similarity computation.
    compute_delta_nb_loss : bool
        Flag for NB loss computation.
    adata_path : str or None
        Path to count data for reconstruction.

    Notes
    -----
    - Only supports 'Global' model_type (raises error for 'Base')
    - When compute_cosine or compute_delta_nb_loss is True, raw embeddings
      are not saved to conserve memory
    - Ablation is performed by systematically removing each gene program and
      evaluating the impact on embeddings or reconstruction

    Examples
    --------
    >>> evaluator = gpAblationEval(
    ...     main_ckpt_dir='path/to/checkpoint',
    ...     compute_cosine=True,
    ...     gpdb_path='path/to/gpdb.csv',
    ...     output_dir='path/to/output',
    ...     dataset_path='path/to/data'
    ... )
    >>> evaluator.generate_embeddings(split='test')
    """

    def __init__(
        self,
        main_ckpt_dir,
        compute_cosine=False,
        compute_delta_nb_loss=False,
        adata_path=None,
        *args,
        **kwargs,
    ):
        self.main_ckpt_dir = os.path.join(main_ckpt_dir, 'checkpoints/last.ckpt')
        self.compute_cosine = compute_cosine
        self.compute_delta_nb_loss = compute_delta_nb_loss
        self.adata_path = adata_path
        super().__init__(*args, **kwargs)

    def _init_trainer(self, split_label=None, **kwargs):
        """
        Initialize the ablation trainer module.

        Loads the gpAblation model from checkpoint and configures it for
        ablation analysis. Sets flags for cosine similarity or NB loss
        computation based on initialization parameters.

        Parameters
        ----------
        split_label : str, optional
            Dataset split to use ('train', 'test', or 'val').
        **kwargs
            Additional keyword arguments (currently unused but maintained
            for compatibility with parent class).

        Returns
        -------
        gp_transformer : gpAblation
            Configured ablation trainer ready for evaluation.

        Raises
        ------
        ValueError
            If model_type is 'Base' (ablation only supports 'Global' models).

        Notes
        -----
        - Automatically sets save_raw_embeddings=False when computing
          cosine or NB loss metrics to save memory
        - Extracts model configuration including encoder package and max length
        """
        if self.model_type == 'Base':
            raise ValueError('Ablation not implemented for Base model')

        elif self.model_type == 'Global':
            gp_transformer = gpAblation.load_from_checkpoint(
                self.main_ckpt_dir, hparam_save='ignore_model', map_location='cpu'
            )

        # reset attributes overwritten by loading from checkpoint
        gp_transformer.save_emb = True
        gp_transformer.split_label = split_label
        gp_transformer.output_dir = self.output_dir
        gp_transformer.compute_cosine = self.compute_cosine
        gp_transformer.compute_delta_nb_loss = self.compute_delta_nb_loss
        gp_transformer.save_raw_embeddings = not (
            self.compute_cosine or self.compute_delta_nb_loss
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
            # TO DO --> flexibly account for different model sizes
            self.max_len = 4096

        return gp_transformer

    def generate_embeddings(
        self, split='train', precision=32, return_mean_non_padding=False
    ):
        """
        Generate embeddings for ablation analysis.

        Performs systematic ablation by removing each gene program and generating
        embeddings.
        For larger datasets, it is more efficient to compute cosine similarity or
        changes in reconstruction loss rather than saving raw embeddings.

        Parameters
        ----------
        split : str, default='train'
            Dataset split to use for generating embeddings ('train', 'test', or 'val').
        precision : int, default=32
            Numerical precision for PyTorch Lightning trainer (16, or 32).
        return_mean_non_padding : bool, default=False
            Whether to return mean embeddings excluding padding tokens.
            (for compatibility with parent class)

        Returns
        -------
        None
            Results are saved to output_dir based on the configured analysis type:
            - If compute_cosine=True: saves cosine similarities between ablated
              and full embeddings
            - If compute_delta_nb_loss=True: saves changes in reconstruction loss
            - Otherwise: saves raw ablated embeddings as HuggingFace Dataset

        Notes
        -----
        - Requires adata_path to be set during initialization if computing NB loss
        - Each gene program is ablated iteratively under the hood
        - Output files are named according to the split and ablation configuration

        """
        gp_transformer = self._init_trainer(
            save_emb=True,
            split_label=split,
            hparam_save=self.hparam_save,
            return_mean_non_padding=return_mean_non_padding,
        )

        txdata = txDataModule(
            folder=self.dataset_path,
            batch_size=self.batch_size,
            data_split_to_pass_to_test_step=split,
            seed=self.seed,
            fm_encoder_name=self.fm_encoder_name,
            model_input_size=self.max_len,
            adata_path=self.adata_path,  # for reconstruction loss calculation
        )

        trainer = pl.Trainer(
            max_epochs=1, devices=1, accelerator='auto', precision=precision
        )

        trainer.test(gp_transformer, txdata)


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
    """
    Calculate Earth Mover's Distance (EMD) between reference and query distributions.

    Computes the EMD metric to quantify the distributional difference between
    reference and query gene program embeddings across different conditions.

    Parameters
    ----------
    data : str or datasets.Dataset or anndata.AnnData
        Input data containing embeddings. Can be:
        - Path to .h5ad file
        - Path to HuggingFace dataset
        - Loaded dataset or AnnData object
    source_key : str
        Column name in data.obs identifying reference vs query samples.
    ref_label : str
        Value in source_key column identifying reference samples.
    query_label : str
        Value in source_key column identifying query samples.
    condition_key : str
        Column name for grouping samples within reference/query sets.
    output_dir : str
        Directory path where results CSV will be saved.
    filename : str, optional
        Custom filename for output CSV. If None, uses format:
        '{ref_label}_vs_{query_label}_emd.csv'
    gp : str or list of str, optional
        Specific gene program(s) to analyze. If None, uses all from gpdb.
    gpdb : str, optional
        Path to gene program database CSV. Required if gp is None.
    filtering_dict : dict, optional
        Dictionary of {column: value(s)} for filtering data before analysis.
        Values can be single items or lists.

    Returns
    -------
    None
        Results are saved to a CSV file in output_dir.

    Notes
    -----
    Either gp or gpdb must be provided to specify gene programs to analyze.
    """

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
# Visualization
################################


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
    Visualize gene program embeddings with gene expression overlay on UMAP.

    Creates UMAP plots of gene program embeddings colored by gene expression
    levels. Optionally filters data and displays additional metadata labels.

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
    """
    Calculate evaluation metrics for gene program embeddings using supervised learning.

    Trains a classifier or regressor on gene program embeddings and evaluates
    performance on a test set. Supports multiple model types and tasks.

    Parameters
    ----------
    train_set : datasets.Dataset or anndata.AnnData
        Training data containing embeddings and labels.
    test_set : datasets.Dataset or anndata.AnnData
        Test data containing embeddings and labels.
    gp : str
        Gene program name to use for embeddings. For AnnData, use 'cell_token'
        to use full .X matrix.
    label : str
        Column name containing target labels for classification/regression.
    output_dir : str
        Directory where results CSV will be saved.
    k : int, default=20
        Number of neighbors for KNN models.
    data_type : {'dataset', 'h5ad'}, default='dataset'
        Format of input data.
    task : {'classification', 'regression'}, default='classification'
        Type of supervised learning task.
    normalize : bool, default=False
        Whether to normalize embeddings by row sum. Only for AnnData input.
    model_type : {'knn', 'linear', 'logistic'}, default='knn'
        Type of model to use:
        - 'knn': K-nearest neighbors
        - 'linear': Linear regression (regression only)
        - 'logistic': Logistic regression (classification only)

    Returns
    -------
    None
        Results are saved to CSV file in output_dir.

    Notes
    -----
    For classification tasks, saves a detailed classification report with
    precision, recall, and F1 scores per class. For regression tasks, saves
    MSE, MAE, and R² metrics.
    """
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
        print(f'Accuracy: {accuracy: .2f}')

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

        print(f'Mean Squared Error (MSE): {mse: .2f}')
        print(f'Mean Absolute Error (MAE): {mae: .2f}')
        print(f'R-squared (R2): {r2: .2f}')

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


# ===================================
# Gene, GP cosine similarity
# ===================================


def calculate_gene_significance(
    input_data,
    obs_col=None,
    obs_value_ref=None,
    obs_value_query=None,
    adata_gene_threshold=0.9,
    fillna=False,
):
    """
    Calculate statistical significance of gene differences between two groups.

    Performs gene-wise t-tests comparing reference and query groups, with
    Benjamini-Hochberg multiple testing correction. Supports both AnnData
    (with sparse matrices) and long-format DataFrame inputs.

    Parameters
    ----------
    input_data : anndata.AnnData or pandas.DataFrame
        Input data containing gene information. For AnnData, uses .X matrix
        (sparse supported) with genes in .var_names. For DataFrame, must
        contain columns: ['gene', 'cosine_sim', obs_col].
    obs_col : str, optional
        Column name containing group labels. Required if obs_value_ref and
        obs_value_query are provided.
    obs_value_ref : str, optional
        Value in obs_col identifying the reference group.
    obs_value_query : str, optional
        Value in obs_col identifying the query group.
    adata_gene_threshold : float, default=0.9
        For AnnData input only. Removes genes that are zero in more than this
        fraction of cells (0 to 1).
    fillna : bool, default=False
        For DataFrame input only. If True, fills NaN values in 'cosine_sim'
        with 0.

    Returns
    -------
    results : pandas.DataFrame
        DataFrame with columns:
        - 'gene': gene name
        - 'p_value': raw p-value from t-test
        - 'mean_ref': mean value in reference group
        - 'mean_query': mean value in query group
        - 'effect_size': difference (mean_query - mean_ref)
        - 'p_adjusted': Benjamini-Hochberg adjusted p-value
        - 'significance': star notation ('***', '**', '*', or '')

    Raises
    ------
    ValueError
        If required parameters are missing or no cells found for specified groups.
    KeyError
        If required columns are missing from DataFrame input.

    Notes
    -----
    - For AnnData, efficiently handles sparse matrices without densification
    - Uses equal-variance t-test
    - Significance stars: *** p<0.001, ** p<0.01, * p<0.05
    - Genes with insufficient data (n<=1) for either group get NaN p-values
    """
    # ---- DataFrame path (backwards compatible) ----
    if isinstance(input_data, pd.DataFrame):
        if obs_col is None or obs_value_ref is None or obs_value_query is None:
            raise ValueError(
                'For DataFrame input, provide obs_col, obs_value_ref, obs_value_query.'
            )
        df = input_data.copy()
        for col in ['gene', 'cosine_sim', obs_col]:
            if col not in df.columns:
                raise KeyError(f'DataFrame missing column {col!r}')
        if fillna:
            df['cosine_sim'] = df['cosine_sim'].fillna(0.0)

        ref = df[df[obs_col] == obs_value_ref]
        query = df[df[obs_col] == obs_value_query]
        genes = pd.Index(sorted(set(ref['gene']).union(set(query['gene']))))

        out = []

        for g in genes:
            r = ref.loc[ref['gene'] == g, 'cosine_sim'].to_numpy()
            q = query.loc[query['gene'] == g, 'cosine_sim'].to_numpy()
            mR = np.nanmean(r) if r.size else np.nan
            mQ = np.nanmean(q) if q.size else np.nan
            if r.size > 1 and q.size > 1:
                _, p = ttest_ind(r, q, equal_var=True, nan_policy='omit')
            else:
                p = np.nan
            out.append(
                dict(gene=g, p_value=p, mean_ref=mR, mean_query=mQ, effect_size=mQ - mR)
            )

        res = pd.DataFrame(out)
        res['p_adjusted'] = _bh_with_nans(res['p_value'])
        res['significance'] = [_stars(p) for p in res['p_adjusted']]
        return res

    # ---- AnnData path (sparse native) ----
    if (ad is not None) and isinstance(input_data, ad.AnnData):
        adata = input_data
        if obs_col is None or obs_value_ref is None or obs_value_query is None:
            raise ValueError(
                'For AnnData input, provide obs_col, obs_value_ref, obs_value_query.'
            )
        if obs_col not in adata.obs.columns:
            raise KeyError(f'{obs_col!r} not found in adata.obs')

        X = adata.X
        n_cells = X.shape[0]

        # filter genes by nonzero fraction (works for sparse/dense)
        if sp.issparse(X):
            Xc = X if sp.isspmatrix_csc(X) else X.tocsc(copy=False)
            nnz_per_gene = np.asarray(Xc.getnnz(axis=0)).ravel()
        else:
            nnz_per_gene = np.count_nonzero(X, axis=0)
        zero_frac = 1.0 - (nnz_per_gene / max(n_cells, 1))
        keep = zero_frac <= adata_gene_threshold
        if not np.any(keep):
            raise ValueError("All genes filtered out by 'adata_gene_threshold'.")

        genes_kept = adata.var_names[keep].to_numpy()
        Xg = X[:, keep]

        labels = adata.obs[obs_col].to_numpy()
        mask_ref = labels == obs_value_ref
        mask_query = labels == obs_value_query
        if not mask_ref.any():
            raise ValueError(f'No cells found for obs_value_ref={obs_value_ref!r}')
        if not mask_query.any():
            raise ValueError(f'No cells found for obs_value_query={obs_value_query!r}')

        stats = _t_equal_var_sparse(Xg[mask_ref, :], Xg[mask_query, :])

        res = pd.DataFrame(
            {
                'gene': genes_kept,
                'p_value': stats['p_value'],
                'mean_ref': stats['mean_ref'],
                'mean_query': stats['mean_query'],
                'effect_size': stats['effect'],
            }
        )
        res['p_adjusted'] = _bh_with_nans(res['p_value'])
        res['significance'] = [_stars(p) for p in res['p_adjusted']]
        return res

    raise ValueError('input_data must be AnnData or a long-format DataFrame.')


def plot_top_genes(
    stats_df,
    obs_value_ref=None,
    obs_value_query=None,
    gp_to_color=None,
    gpdb=None,
    topn=10,
    figsize=(18, 14),
    show_significance=True,
    color_scheme='default',  # 'default' or 'significance'
    significance_palette='Blues',  # cmap name OR [ref_color, query_color]
    palette_as_gradient=False,  # if True, build gradient(s) from provided colors
    hspace=0.5,
    wspace=0.5,
    save_to=None,
):
    """
    Create a 2x2 grid of bar plots showing top genes by various metrics.

    Generates four subplots:
    1. Top genes by mean cosine similarity in reference group
    2. Top genes by mean cosine similarity in query group
    3. Top genes with strongest negative effect (ref > query)
    4. Top genes with strongest positive effect (query > ref)

    Parameters
    ----------
    stats_df : pandas.DataFrame
        Output from calculate_gene_significance with columns:
        ['gene', 'mean_ref', 'mean_query', 'effect_size', 'p_adjusted', 'significance'].
    obs_value_ref : str, optional
        Label for reference group (used in titles).
    obs_value_query : str, optional
        Label for query group (used in titles).
    gp_to_color : str, optional
        Gene program name for custom coloring. Requires gpdb.
    gpdb : pandas.DataFrame, optional
        Gene program database for custom coloring.
    topn : int, default=10
        Number of top genes to display in each subplot.
    figsize : tuple, default=(18, 14)
        Figure size in inches (width, height).
    show_significance : bool, default=True
        Whether to show significance stars on effect size plots.
    color_scheme : {'default', 'significance'}, default='default'
        Coloring scheme:
        - 'default': uses gp_to_color if provided, else default seaborn colors
        - 'significance': colors bars by -log10(p_adjusted) with colorbar
    significance_palette : str or list, default='Blues'
        For color_scheme='significance':
        - str: matplotlib colormap name
        - list: [ref_color, query_color] for custom colors
    palette_as_gradient : bool, default=False
        If True with list significance_palette, creates white-to-color gradients.
    hspace : float, default=0.5
        Vertical spacing between subplots.
    wspace : float, default=0.5
        Horizontal spacing between subplots.
    save_to : str, optional
        Path to save figure. If None, figure is only displayed.

    Returns
    -------
    None
        Displays and optionally saves the plot.

    Notes
    -----
    - Significance stars: *** p<0.001, ** p<0.01, * p<0.05
    - Non-significant genes (p≥0.05) shown in gray when color_scheme='significance'
    - Effect size is calculated as mean_query - mean_ref
    """
    # defensive copy & ordering
    df = stats_df.copy()

    # Top-N by means
    ref_top = df.nlargest(topn, 'mean_ref')
    query_top = df.nlargest(topn, 'mean_query')

    # Top-N by effect size (signed)
    sig_diff_ref = df[df['effect_size'] < 0]
    sig_diff_query = df[df['effect_size'] > 0]
    top_diff_ref = sig_diff_ref.nsmallest(topn, 'effect_size')
    top_diff_query = sig_diff_query.nlargest(topn, 'effect_size')

    # Resolve significance colors/palette
    is_listlike, ref_col, qry_col, cmap_name = _resolve_sig_colors(significance_palette)

    # ---------------- Plotting (your layout) ----------------
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.set_facecolor('white')

    # 1) Top mean (ref)
    if gp_to_color:
        palette = dict(
            zip(ref_top['gene'], assign_bar_colors(ref_top['gene'], gp_to_color, gpdb))
        )
        sns.barplot(
            data=ref_top,
            x='mean_ref',
            y='gene',
            ax=axs[0, 0],
            errorbar=None,
            palette=palette,
        )
    else:
        sns.barplot(data=ref_top, x='mean_ref', y='gene', ax=axs[0, 0], errorbar=None)
    title_ref = f'({obs_value_ref})' if obs_value_ref is not None else '(ref)'
    axs[0, 0].set_title(
        f'Top {topn} Genes: Highest Mean Cosine Similarity\n{title_ref}'
    )
    axs[0, 0].set_xlabel('Cosine Similarity')
    axs[0, 0].set_ylabel('Gene')

    # 2) Top mean (query)
    if gp_to_color:
        palette = dict(
            zip(
                query_top['gene'],
                assign_bar_colors(query_top['gene'], gp_to_color, gpdb),
            )
        )
        sns.barplot(
            data=query_top,
            x='mean_query',
            y='gene',
            ax=axs[0, 1],
            errorbar=None,
            palette=palette,
        )
    else:
        sns.barplot(
            data=query_top, x='mean_query', y='gene', ax=axs[0, 1], errorbar=None
        )
    title_qry = f'({obs_value_query})' if obs_value_query is not None else '(query)'
    axs[0, 1].set_title(
        f'Top {topn} Genes: Highest Mean Cosine Similarity\n{title_qry}'
    )
    axs[0, 1].set_xlabel('Cosine Similarity')
    axs[0, 1].set_ylabel('Gene')

    # 3) Strongest difference (ref > query)   -> negative effect_size
    if color_scheme == 'significance':
        vals = top_diff_ref['effect_size']
        pvals_raw = top_diff_ref['p_adjusted']
        min_nonzero = pvals_raw[pvals_raw > 0].min() if (pvals_raw > 0).any() else 1e-20
        pvals = pvals_raw.replace(0, min_nonzero)
        neglogp = -np.log10(pvals)

        if palette_as_gradient:
            base_color = to_rgba(ref_col if is_listlike else 'tab:red')
            cmap = LinearSegmentedColormap.from_list(
                'ref_cmap', [(1, 1, 1, 1), base_color]
            )
        else:
            cmap = plt.get_cmap(cmap_name)

        norm = Normalize(
            vmin=neglogp.min() if len(neglogp) else 0,
            vmax=neglogp.max() if len(neglogp) else 1,
        )
        colors = cmap(norm(neglogp)) if len(neglogp) else None
        axs[1, 0].barh(
            y=top_diff_ref['gene'], width=vals, color=colors, edgecolor='black'
        )
        axs[1, 0].invert_yaxis()
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs[1, 0])
        cbar.set_label('-log10(Adjusted p-value)', rotation=270, labelpad=15)

        # gray-out non-significant labels
        yticklabels = axs[1, 0].get_yticklabels()
        idx = top_diff_ref.set_index('gene')['p_adjusted']
        for lab in yticklabels:
            g = lab.get_text()
            if pd.notna(idx.get(g, np.nan)) and idx[g] >= 0.05:
                lab.set_color('gray')
        axs[1, 0].set_yticklabels(yticklabels)
    else:
        if gp_to_color:
            palette = dict(
                zip(
                    top_diff_ref['gene'],
                    assign_bar_colors(top_diff_ref['gene'], gp_to_color, gpdb),
                )
            )
            sns.barplot(
                data=top_diff_ref,
                x='effect_size',
                y='gene',
                ax=axs[1, 0],
                errorbar=None,
                palette=palette,
            )
        else:
            sns.barplot(
                data=top_diff_ref,
                x='effect_size',
                y='gene',
                ax=axs[1, 0],
                color='salmon',
            )

    if show_significance:
        for i, (effect, star) in enumerate(
            zip(top_diff_ref['effect_size'], top_diff_ref['significance'])
        ):
            if star:
                axs[1, 0].text(
                    effect - 1e-4,
                    i,
                    star,
                    ha='left',
                    va='center',
                    fontsize=12,
                    color='darkred',
                )

    axs[1, 0].set_title(
        f'Top {topn} Genes: Strongest Difference\n({obs_value_ref or "ref"}'
        f'> {obs_value_query or "query"})'
    )
    axs[1, 0].set_xlabel('Difference in Cosine Similarity')
    axs[1, 0].set_ylabel('Gene')

    # 4) Strongest difference (query > ref)   -> positive effect_size
    if color_scheme == 'significance':
        vals = top_diff_query['effect_size']
        pvals_raw = top_diff_query['p_adjusted']
        min_nonzero = pvals_raw[pvals_raw > 0].min() if (pvals_raw > 0).any() else 1e-20
        pvals = pvals_raw.replace(0, min_nonzero)
        neglogp = -np.log10(pvals)

        if palette_as_gradient:
            base_color = to_rgba(qry_col if is_listlike else 'tab:blue')
            cmap = LinearSegmentedColormap.from_list(
                'query_cmap', [(1, 1, 1, 1), base_color]
            )
        else:
            cmap = plt.get_cmap(cmap_name)

        norm = Normalize(
            vmin=neglogp.min() if len(neglogp) else 0,
            vmax=neglogp.max() if len(neglogp) else 1,
        )
        colors = cmap(norm(neglogp)) if len(neglogp) else None
        axs[1, 1].barh(
            y=top_diff_query['gene'], width=vals, color=colors, edgecolor='black'
        )
        axs[1, 1].invert_yaxis()
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs[1, 1])
        cbar.set_label('-log10(Adjusted p-value)', rotation=270, labelpad=15)

        # gray-out non-significant labels
        yticklabels = axs[1, 1].get_yticklabels()
        idx = top_diff_query.set_index('gene')['p_adjusted']
        for lab in yticklabels:
            g = lab.get_text()
            if pd.notna(idx.get(g, np.nan)) and idx[g] >= 0.05:
                lab.set_color('gray')
        axs[1, 1].set_yticklabels(yticklabels)
    else:
        if gp_to_color:
            palette = dict(
                zip(
                    top_diff_query['gene'],
                    assign_bar_colors(top_diff_query['gene'], gp_to_color, gpdb),
                )
            )
            sns.barplot(
                data=top_diff_query,
                x='effect_size',
                y='gene',
                ax=axs[1, 1],
                errorbar=None,
                palette=palette,
            )
        else:
            sns.barplot(
                data=top_diff_query,
                x='effect_size',
                y='gene',
                ax=axs[1, 1],
                color='skyblue',
            )

    if show_significance:
        for i, (effect, star) in enumerate(
            zip(top_diff_query['effect_size'], top_diff_query['significance'])
        ):
            if star:
                axs[1, 1].text(
                    effect + 1e-4,
                    i,
                    star,
                    ha='right',
                    va='center',
                    fontsize=12,
                    color='navy',
                )

    axs[1, 1].set_title(
        f'Top {topn} Genes: Strongest Difference\n({obs_value_query or "query"}'
        f'> {obs_value_ref or "ref"})'
    )
    axs[1, 1].set_xlabel('Difference in Cosine Similarity')
    axs[1, 1].set_ylabel('Gene')

    # layout & save
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    if save_to:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()


# -------------------------------------------
# Phate
# -------------------------------------------


def compute_phate(adata, n_components=2):
    """
    Compute PHATE dimensionality reduction on AnnData object.

    PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding)
    is a dimensionality reduction method that preserves both local and global
    structure in high-dimensional data.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing data in .X matrix.
    n_components : int, default=2
        Number of dimensions in the PHATE embedding.

    Returns
    -------
    data_phate : numpy.ndarray
        PHATE embedding with shape (n_cells, n_components).
    """
    phate_operator = phate.PHATE(n_components=n_components)
    data_phate = phate_operator.fit_transform(adata.X)
    return data_phate


def plot_phate(
    data_phate, adata, label, label_order, output_file, color_map='Spectral'
):
    """
    Create and save a PHATE scatter plot colored by categorical labels.

    Generates a 2D scatter plot of PHATE embeddings with points colored
    according to a categorical label, using a fixed category order.

    Parameters
    ----------
    data_phate : numpy.ndarray
        PHATE embedding coordinates with shape (n_cells, n_components).
    adata : anndata.AnnData
        AnnData object containing metadata in .obs.
    label : str
        Column name in adata.obs to use for coloring points.
    label_order : list of str
        Desired order of categories for color assignment and legend.
    output_file : str
        Path where the plot will be saved as PDF.
    color_map : str, default='Spectral'
        Name of matplotlib colormap to use for category colors.

    Returns
    -------
    None
        Saves plot to output_file and displays it.

    Notes
    -----
    - Categories not present in data are automatically filtered from label_order
    - Legend is placed outside the plot area (upper left, bbox_to_anchor=(1, 1))
    - Uses scprep.plot.scatter2d for plotting
    """
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

    plt.rcdefaults()

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


# -------------------------------------------
# GP importance score visualization
# -------------------------------------------


def plot_gp_score_fold_change(df, title_ct, top_n=10, color_map='Blues', save_to=None):
    """
    Create horizontal bar plot of top gene programs by log fold change.

    Visualizes the top gene programs ranked by log fold change, with bars
    colored by statistical significance (-log10 of adjusted p-value).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns:
        - 'names': gene program names
        - 'logfoldchanges': log fold change values
        - 'pvals_adj': adjusted p-values
        Additional columns may be present but are not used.
    title_ct : str
        Cell type or condition name to include in plot title.
    top_n : int, default=10
        Number of top gene programs to display.
    color_map : str, default='Blues'
        Name of matplotlib colormap for coloring bars by significance.
    save_to : str, optional
        Path to save the figure. If None, figure is only displayed.

    Returns
    -------
    None
        Displays and optionally saves the plot.
    """
    # Prepare data
    df_sorted = df.reindex(df['logfoldchanges'].sort_values(ascending=False).index)
    df_top = df_sorted.head(top_n).copy()
    df_top['-log10(pvals_adj)'] = -np.log10(
        df_top['pvals_adj'].replace(0, np.nextafter(0, 1))
    )

    # Color mapping
    cmap = plt.get_cmap(color_map)
    norm = Normalize(
        vmin=df_top['-log10(pvals_adj)'].min(), vmax=df_top['-log10(pvals_adj)'].max()
    )
    colors = cmap(norm(df_top['-log10(pvals_adj)']))

    # Plot
    fig, ax = plt.subplots(figsize=(5, 0.15 * top_n + 2))
    ax.barh(
        y=df_top['names'],
        width=df_top['logfoldchanges'],
        color=colors,
        edgecolor='black',
    )

    ax.invert_yaxis()
    ax.set_xlabel('Log Fold Change')
    ax.set_ylabel('')
    ax.set_title(f'Top {top_n} GP in {title_ct} \nby log fold change')

    # Add colorbar reflecting -log10(pvals_adj)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('-log10(Adjusted p-value)', rotation=270, labelpad=15)

    plt.tight_layout()

    if save_to:
        plt.savefig(save_to)

    plt.show()


def plot_gp_scores_per_cell(
    adata,
    var_names,
    groupby,
    layer='Ms',
    color_map='viridis',
    n_convolve=20,
    min_max_scale=None,
    colorbar=None,
    context=None,
    font_scale=None,
    figsize=(10, 4),
    show=None,
    save=None,
    **kwargs,
):
    """
    Create heatmap of gene program scores across cells grouped by category.

    Generates a heatmap showing gene program scores for individual cells,
    sorted within groups and optionally smoothed with a moving average.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing gene program scores.
    var_names : list of str
        List of variable names (genes/gene programs) to plot. Must exist
        in adata.var_names.
    groupby : str
        Column name in adata.obs for grouping cells.
    layer : str, default='Ms'
        Layer name in adata.layers to use for scores. If not found, uses .X.
    color_map : str, default='viridis'
        Matplotlib colormap name for heatmap colors.
    n_convolve : int, default=20
        Window size for moving average smoothing. If None, no smoothing applied.
    min_max_scale : {0, 1, None}, optional
        Normalization method:
        - 0: normalize each gene (row) independently to [0, 1]
        - 1: normalize across all values to [0, 1]
        - None: no normalization
    colorbar : bool, optional
        Whether to show colorbar. Default depends on context.
    context : str, optional
        Seaborn plotting context ('paper', 'notebook', 'talk', 'poster').
    font_scale : float, optional
        Font scale multiplier for seaborn context.
    figsize : tuple, default=(10, 4)
        Figure size in inches (width, height).
    show : bool, optional
        Whether to display the plot. If None, shows by default.
    save : str, optional
        Path to save the figure. If None, figure is not saved.
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    None
        Displays and optionally saves the plot.

    Notes
    -----
    - Cells are sorted within each group by the first variable in var_names
    - Group boundaries are marked with vertical black lines
    - Group names are displayed below the heatmap
    - Invalid genes are skipped with a warning during smoothing
    - NaN and Inf values are replaced with 0
    """
    # Filter valid genes
    var_names = [g for g in var_names if g in adata.var_names]
    if not var_names:
        raise ValueError('No valid var_names found in adata.')

    group_vals = adata.obs[groupby].astype('category')
    group_categories = group_vals.cat.categories

    # Extract expression data
    X = (
        adata[:, var_names].layers[layer]
        if layer in adata.layers
        else adata[:, var_names].X
    )
    if issparse(X):
        X = X.toarray()

    df = pd.DataFrame(X, columns=var_names)
    df['group'] = group_vals.values

    sorted_idx = []
    for group in group_categories:
        sub_df = df[df['group'] == group].drop('group', axis=1).copy()
        if n_convolve:
            weights = np.ones(n_convolve) / n_convolve
            for gene in var_names:
                try:
                    sub_df[gene] = np.convolve(
                        sub_df[gene].values, weights, mode='same'
                    )
                except Exception as e:
                    print(f'Skipping gene {gene} due to error: {e}')
        sub_idx = sub_df.index[np.argsort(sub_df[var_names[0]].values)]
        sorted_idx.extend(sub_idx)

    df_sorted = df.loc[sorted_idx]
    df_sorted_expr = df_sorted.drop('group', axis=1)

    if min_max_scale == 0:
        df_sorted_expr = (df_sorted_expr.T - df_sorted_expr.T.min()) / (
            df_sorted_expr.T.max() - df_sorted_expr.T.min()
        )
        df_sorted_expr = df_sorted_expr.T

    elif min_max_scale == 1:
        df_sorted_expr = (df_sorted_expr - df_sorted_expr.min()) / (
            df_sorted_expr.max() - df_sorted_expr.min()
        )

    df_sorted = pd.concat([df_sorted_expr, df_sorted['group']], axis=1)

    numeric_columns = df_sorted.select_dtypes(include=[np.number]).columns
    if df_sorted[numeric_columns].isna().any().any():
        print('Warning: NaN values found, replacing with 0.')
        df_sorted[numeric_columns] = df_sorted[numeric_columns].fillna(0)
    if np.isinf(df_sorted[numeric_columns].values).any():
        print('Warning: Inf values found, replacing with 0.')
        df_sorted[numeric_columns] = df_sorted[numeric_columns].replace(
            [np.inf, -np.inf], 0
        )

    heat_data = df_sorted.drop('group', axis=1).T.values.astype(float)

    args = {}
    if font_scale:
        args = {'font_scale': font_scale}
        context = context or 'notebook'

    with sns.plotting_context(context=context, **args):
        fig, ax = plt.subplots(figsize=figsize)

        cax = ax.imshow(
            heat_data, cmap=color_map, aspect='auto', interpolation='nearest'
        )
        if colorbar:
            fig.colorbar(cax, ax=ax)

        current_pos = 0
        for i in range(len(df_sorted) - 1):
            if df_sorted.iloc[i]['group'] != df_sorted.iloc[i + 1]['group']:
                ax.axvline(x=current_pos + 1, color='black', lw=1)
            current_pos += 1

        current_pos = 0
        for group in group_categories:
            group_cells = df[df['group'] == group]
            group_len = len(group_cells)
            mid_pos = current_pos + group_len // 2
            ax.text(
                mid_pos,
                -0.75,
                group,
                ha='center',
                va='center',
                fontsize=10,
                color='black',
            )
            current_pos += group_len

        ax.set_yticks(np.arange(len(var_names)))
        ax.set_yticklabels(var_names, fontsize=8)
        ax.set_xticks([])  # Remove bottom x-axis ticks
        ax.set_ylabel('Genes')
        ax.set_xlabel(groupby)
        plt.tight_layout()

    if save:
        plt.savefig(save)

    if show or show is None:
        plt.show()

    plt.close()
