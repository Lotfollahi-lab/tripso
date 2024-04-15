import os
import random
import shutil
from typing import (
    Dict,
    List,
    Optional,
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import seaborn as sns
import torch
from captum.attr import GuidedGradCam
from datasets import load_from_disk
from pytorch_lightning.loggers import CSVLogger
from scib_metrics.benchmark import Benchmarker
from tqdm import tqdm

from ..Datamodules.datamodule import (
    EmbDataModule,
    iEmbDataModule,
    iTxDataModule,
    txDataModule,
)
from ..Models.gp_model import (
    gfBaseline,
    gpTransformerBase,
    gpTransformerGlobal,
    iGlobalWrapper,
    iGpWrapper,
)
from ..Trainers.trainer import EmbEvaluator, scGPL
from ..Utils.utils import (
    do_logistic_regression,
    find_genes_in_multiple_gp,
    find_latest_file,
    get_genes_in_single_gp,
    remove_single_data_points,
    summarize_attributions,
)

# for exporting pdfs
matplotlib.rcParams['pdf.fonttype'] = 42
torch.set_float32_matmul_precision('medium')

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
    add_remaining_var : bool
        Whether to initalize new transformer block covering non GP genes
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
        gene_counts_df: Optional[str] = None,
        n_blocks: Optional[int] = 1,
        gene_format: Optional[str] = 'symbol',
        tissue: Optional[str] = 'test',
        model_type: Optional[str] = 'Base',
        n_heads: Optional[int] = 8,
        gp_latent_size: Optional[int] = 256,
        gp_inputs: Optional[list] = None,
        batch_size: Optional[int] = 128,
        add_remaining_var: Optional[bool] = False,
        supervised_labels: Optional[Dict] = None,
        global_attn_heads: Optional[int] = 1,
        global_n_blocks: Optional[int] = 1,
        global_loss: Optional[str] = 'supervised',
        reconstruction_loss: Optional[str] = 'zinb',
    ):
        # check only one GPU
        assert torch.cuda.device_count() == 1, 'Please run evaluation on single GPU'

        # set seed for reproducibility
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Search for .ckpt files in the directory
        if model_type != 'Mean':
            latest_ckpt = find_latest_file(output_dir, tissue, model_type)
            print('Latest .ckpt file:', latest_ckpt)
            self.checkpoint_path = os.path.join(output_dir, latest_ckpt)

        gpdb = pd.read_csv(gpdb_path)

        if gene_format == 'symbol':
            do_ensembl_conversion = True
        elif gene_format == 'ensembl':
            do_ensembl_conversion = False
        self.do_ensembl_conversion = do_ensembl_conversion

        if gene_counts_df is not None:
            self.gene_counts_df = pd.read_csv(gene_counts_df)
        else:
            self.gene_counts_df = None

        if model_type == 'Base':
            self.model = gpTransformerBase(
                gp_inputs=gp_inputs,
                gene_counts_df=self.gene_counts_df,
                database=gpdb,
                do_ensembl_conversion=do_ensembl_conversion,
                n_blocks=n_blocks,
                num_heads=n_heads,
                gp_latent_size=gp_latent_size,
                add_remaining_var=add_remaining_var,
            )

        elif model_type == 'Global':
            self.model = gpTransformerGlobal(
                gene_counts_df=gene_counts_df,
                database=gpdb,
                do_ensembl_conversion=do_ensembl_conversion,
                n_blocks=n_blocks,
                num_heads=n_heads,
                gp_latent_size=gp_latent_size,
                gp_inputs=gp_inputs,
                add_remaining_var=add_remaining_var,
                supervised_labels=supervised_labels,
                global_attn_heads=global_attn_heads,
                global_n_blocks=global_n_blocks,
                global_loss=global_loss,
                reconstruction_loss=reconstruction_loss,
            )

            self.reconstruction_loss = reconstruction_loss

        elif model_type == 'Mean':
            self.model = gfBaseline(
                gp_inputs=gp_inputs,
                database=gpdb,
                do_ensembl_conversion=do_ensembl_conversion,
                gene_counts_df=self.gene_counts_df,
                # dummy variables to avoid errors if no defaults
                # but we won't use transformer blocks
                n_blocks=1,
                mgm_mask_ratio=1,
                num_heads=1,
                add_remaining_var=add_remaining_var,
            )

        else:
            raise ValueError('model_type must be one of Base, or Mean')

        if gp_inputs is None:
            gp_inputs = gpdb.columns.tolist()
        if isinstance(gp_inputs, str):
            gp_inputs = [gp_inputs]
        if add_remaining_var:
            gp_inputs.append('remaining_var')

        # Remove /
        gp_inputs = [g.replace('/', '_') for g in gp_inputs]

        self.gp_inputs = gp_inputs

        # change directory for saving outputs
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir
        self.tissue = tissue
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.gpdb = gpdb

        # for compatability with gpGlobal init
        self.global_loss = global_loss

        # Set up gpTransformer lightning module
        self.model_type = model_type
        return_classification_report = True if supervised_labels is not None else False
        self.gp_transformer = self._init_trainer(
            return_classification_report=return_classification_report
        )

    def _init_trainer(
        self,
        return_gene_embeddings=False,
        tokens_to_keep=None,
        gene_file_tag=None,
        return_attention=False,
        gp=None,
        return_classification_report=False,
        test_random_baseline=False,
        save_emb=False,
        split_label=None,
    ):
        if (
            self.model_type != 'Mean'
        ):  # no training required if just averaging geneformer embeddings
            gp_transformer = scGPL(
                self.model,
                self.model_type,
                return_gene_embeddings=return_gene_embeddings,
                tokens_to_keep=tokens_to_keep,
                gene_file_tag=gene_file_tag,
                return_attention=return_attention,
                gp=gp,
                return_classification_report=return_classification_report,
                global_loss=self.global_loss,
                test_random_baseline=test_random_baseline,
                save_emb=save_emb,
                split_label=split_label,
            ).load_from_checkpoint(self.checkpoint_path)
        else:
            gp_transformer = scGPL(
                self.model,
                self.model_type,
                tokens_to_keep=tokens_to_keep,
                gene_file_tag=gene_file_tag,
                return_gene_embeddings=return_gene_embeddings,
                output_dir=self.output_dir,
                return_classification_report=return_classification_report,
                save_emb=save_emb,
                split_label=split_label,
            )

        # reset attributes overwritten by loading from checkpoint
        gp_transformer.return_gene_embeddings = return_gene_embeddings
        gp_transformer.tokens_to_keep = tokens_to_keep
        gp_transformer.gene_file_tag = gene_file_tag
        gp_transformer.return_attention = return_attention
        gp_transformer.gp = gp
        gp_transformer.return_classification_report = return_classification_report
        gp_transformer.output_dir = self.output_dir
        gp_transformer.test_random_baseline = test_random_baseline
        gp_transformer.save_emb = save_emb
        gp_transformer.split_label = split_label

        return gp_transformer

    def generate_embeddings(self, split='train'):
        '''
        Save embeddings as Dataset
        '''

        gp_transformer = self._init_trainer(save_emb=True, split_label=split)

        txdata = txDataModule(
            folder=self.dataset_path,
            batch_size=self.batch_size,
            data_split_to_pass_to_val_step=split,
        )

        trainer = pl.Trainer(max_epochs=1, devices=-1, accelerator='auto', precision=16)

        trainer.validate(gp_transformer, txdata)

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
        filter_tag = f'_{filter_key}' if filter_key is not None else ''
        if task == 'classification':
            label_to_count = y_label
        else:
            label_to_count = None

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
            count_n_unique=label_to_count,
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
        )

        logger = CSVLogger(
            os.path.join(output_dir, 'evaluation_logs'),
            name=f'{emb_label}_{y_label.replace("_id", "")}{filter_tag}',
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=ckpt_dir,
            filename=f'{y_label.replace("_id", "")}_{emb_label}_{task}{filter_tag}',
            save_top_k=1,
            mode='min',
        )

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            callbacks=[checkpoint_callback],
            devices=-1,
            accelerator='auto',
            logger=logger,
            precision=16,
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

            for c in label_to_plot:
                sc.pp.neighbors(adata, use_rep='X')
                sc.tl.umap(adata)
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

    @staticmethod
    def benchmarking_with_scib(
        self,
        adata_path: str,
        batch_key: str,
        label_key: str,
        embs_to_benchmark: List,
        model_labels: List,
        emb_label: str,
    ):
        '''
        Run scIB benchmarking
        Based on https://github.com/YosefLab/scib-metrics/

        Parameters
        ----------
        adata_path : str
            Path to original adata
        embs_to_benchmark : list
            List of paths to embeddings to benchmark
            eg ['../output_different_hparam/embeddings/test_set',
                '../expimap/adata_expimap.h5ad']
        model_labels : list
            List of labels for each model
            eg ['gpTransformer', 'Expimap']
        emb_label : str
            Label for embeddings
            eg 'GEP_1'
            eg 'cell_token'

        '''
        # Load original gene expression data
        adata = sc.read_h5ad(adata_path)
        sc.pp.highly_variable_genes(
            adata, n_top_genes=2000, flavor='seurat_v3', batch_key='batch_key'
        )
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
        adata.obsm['Unintegrated'] = adata.obsm['X_pca']

        if isinstance(embs_to_benchmark, str):
            embs_to_benchmark = [embs_to_benchmark]

        if isinstance(model_labels, str):
            model_labels = [model_labels]

        for emb_path, model_name in zip(embs_to_benchmark, model_labels):
            if emb_path.endswith('.h5ad'):
                embx = sc.read_h5ad(emb_path)

                if emb_label != 'cell_token':
                    embx = embx[:, embx.var.str.contains(emb_label)]

            else:
                embx = load_from_disk(emb_path)
                x = np.array(embx[emb_label])
                y = pd.DataFrame(embx[[batch_key, label_key, 'idx']])
                embx = sc.AnnData(X=x, obs=y)

            adata = self._load_and_save_latent(adata, embx, model_name)

        bm = Benchmarker(
            adata,
            batch_key=batch_key,
            label_key=label_key,
            embedding_obsm_keys=model_labels,
            n_jobs=-1,
        )
        bm.benchmark()

        bm.plot_results_table(show=False, savedir=self.output_dir)

        shutil.move(
            os.path.join(self.output_dir, 'scib_results.svg'),
            os.path.join(self.output_dir, 'scib_results_minmax_scaling.svg'),
        )

        bm.plot_results_table(min_max_scale=False, show=False, savedir=self.output_dir)

    def evaluate_gene_embeddings(
        self,
        labels=['cell_type', 'condition'],
        data_to_model='gene_singleGP',  # or "gene_multiGP"
        min_cells=500,
        downsample_to_n_genes=50,
    ):
        """
        Run logistic regression to predict gene labels from embeddings

        Parameters
        ----------
        gp_features : list
            List of GP to use as features
            If "all", will use all GP
            If "concat", will concatenate all GP
        labels : list
            List of labels to predict

        min_cells : int
            Genes present in multiple GP must be included in at least min_cells

        downsample_to_n_genes : int
            Number of genes to downsample to for genes present
            in multiple GP

        """
        os.chdir(self.output_dir)

        # prepare output directories
        if isinstance(labels, str):
            labels = [labels]

        if data_to_model == 'gene_multiGP':
            txdata = txDataModule(folder=self.dataset_path, batch_size=self.batch_size)

            if not os.path.exists('gene_metrics'):
                os.makedirs('gene_metrics')

            genes_in_multiple_gp = find_genes_in_multiple_gp(
                gp_inputs=self.gp_inputs,
                gpdb=self.gpdb,
                token_df=self.gene_counts_df,
                do_ensembl_conversion=self.do_ensembl_conversion,
                min_cells=min_cells,
                downsample_to_n_genes=downsample_to_n_genes,
            )

            gp_transformer = self._init_trainer(
                return_gene_embeddings=True,
                tokens_to_keep=genes_in_multiple_gp,
                gene_file_tag='multipleGP',
            )
            trainer = pl.Trainer(
                max_epochs=1, devices=-1, accelerator='auto', precision=16
            )
            trainer.test(gp_transformer, txdata)

            adata = sc.read_h5ad('adata_gene_embedding_multipleGP.h5ad')

            print('Run logistic regression models - PER GENE')
            for g in adata.obs['ensembl'].unique():
                print(g)
                tdata = adata[adata.obs['ensembl'] == g]
                do_logistic_regression(
                    tdata,
                    'GP',
                    os.path.join(self.output_dir, 'gene_metrics'),
                    filename=f'gp_prediction_from_scgpl_{g}',
                    variable_to_track={'ensembl': g, 'gene': tdata.obs['gene'].iloc[0]},
                )

        elif data_to_model == 'gene_singleGP':
            txdata = txDataModule(
                folder=self.dataset_path, batch_size=self.batch_size, num_workers=4
            )

            if not os.path.exists('gene_metrics'):
                os.makedirs('gene_metrics')

            genes_in_single_gp = get_genes_in_single_gp(
                gpdb=self.gpdb,
                do_ensembl_conversion=self.do_ensembl_conversion,
                downsample_to_n_genes=downsample_to_n_genes,
            )

            gp_transformer = self._init_trainer(
                return_gene_embeddings=True,
                tokens_to_keep=genes_in_single_gp,
                gene_file_tag='singleGP',
            )

            trainer = pl.Trainer(
                max_epochs=1, devices=-1, accelerator='auto', precision=16
            )

            trainer.test(gp_transformer, txdata)

            adata_scgpl = sc.read_h5ad(
                os.path.join(self.output_dir, 'adata_gene_embedding_singleGP.h5ad')
            )

            do_logistic_regression(
                adata_scgpl,
                'GP',
                os.path.join(self.output_dir, 'gene_metrics'),
                filename='gp_prediction_from_scgpl',
                variable_to_track={'embedding_type': 'scGPL'},
            )

    def generate_attention_matrix(self, gp):
        """
        Get attention weights from gpTransformer
        """
        os.chdir(self.output_dir)

        if self.model.use_flash:
            raise ValueError('Attention weights not available with flash attentiokn')

        if (gp != 'cell_token') and (gp not in self.gp_inputs):
            raise ValueError(f'{gp} must be one of "cell_token" or {self.gp_inputs}')

        # Initialize trainer
        txdata = txDataModule(folder=self.dataset_path, batch_size=self.batch_size)

        gp_transformer = self._init_trainer(return_attention=True, gp=gp)

        trainer = pl.Trainer(max_epochs=1, devices=-1, accelerator='auto', precision=16)

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

        txdata = txDataModule(
            folder=self.dataset_path,
            batch_size=self.batch_size,
            adata_path=adata_path,
        )

        gp_transformer = self._init_trainer(test_random_baseline=True)
        trainer = pl.Trainer(max_epochs=1, devices=-1, accelerator='auto', precision=16)
        trainer.test(gp_transformer, txdata)


################################
# For attributions
################################


def calculate_gp_attribution_scores(
    gpdb_path,
    dataset_path,
    gp,
    data_split,
    n_blocks,
    num_heads,
    gp_latent_size,
    model_checkpoint,
    obs_key,
    obs_value,
    output_dir,
    total_n_cells=None,
    task='classification',
    gpdb_ref_path=None,
    do_ensembl_conversion=True,
    emb_dataset_path=None,
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
    # Set up dataloader
    # --------------------------

    gpdb = pd.read_csv(gpdb_path)

    txdata = iTxDataModule(
        folder=dataset_path,
        batch_size=1,
        return_tuple=True,
        gp=gp,
        gpdb=gpdb,
        do_ensembl_conversion=do_ensembl_conversion,
        filter_key=obs_key,
        filter_value=obs_value,
    )

    txdata.setup()
    dataloader = getattr(txdata, data_split + '_dataloader')()
    print('Number of cells', len(dataloader))

    if total_n_cells is None:
        total_n_cells = len(dataloader)

    # --------------------------
    # Build dictionary for class : id conversion
    # --------------------------

    y_label = obs_key + '_id'

    datax = load_from_disk(dataset_path)
    cols_to_remove = datax.column_names
    cols_to_remove.remove(obs_key)
    cols_to_remove.remove(y_label)
    datax = datax.remove_columns(cols_to_remove)
    conversion = datax.to_pandas().drop_duplicates()

    n_classes = len(conversion)
    conversion_dict = {k: v for k, v in zip(conversion[obs_key], conversion[y_label])}

    # --------------------------
    # Set up model
    # --------------------------

    # TO DO : can we get this as config file?
    model = gpTransformerBase(
        gene_counts_df=None,
        database=gpdb,
        do_ensembl_conversion=do_ensembl_conversion,
        n_blocks=n_blocks,
        num_heads=num_heads,
        gp_latent_size=gp_latent_size,
        gp_inputs=gp,
        add_remaining_var=False,
    )

    gp_transformer = scGPL(
        model,
        'Base',
        return_gene_embeddings=False,
        tokens_to_keep=None,
        gene_file_tag=None,
        return_attention=False,
        gp=None,
        return_classification_report=False,
    ).load_from_checkpoint(
        model_checkpoint,
        strict=False,
    )

    # Load classification layer
    # or train if not available
    if gp_transformer.global_loss != 'supervised':
        ckpt_dir = os.path.join(output_dir, 'evaluation_model_checkpoints')
        clf_ckpt = f'{y_label.replace("_id", "")}_{gp}_{task}'
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
                n_classes=n_classes,
                y_label=y_label,
                folder_path=emb_dataset_path,
                output_dir=output_dir,
                emb_label=gp,
                task=task,
                emb_dim=gp_latent_size,
                lr=1e-3,
                batch_size=128,
                num_workers=1,
                meta_labels=[y_label, y_label.replace('_id', '')],
                data_type='dataset',
                n_epochs=3,
                continuous_cov=[],
            )

            clf_layer = EmbEvaluator.load_from_checkpoint(
                os.path.join(ckpt_dir, f'{clf_ckpt}.ckpt')
            )

        task_index = None
    else:
        clf_layer = None

    imodel = iGpWrapper(
        gp_transformer,
        clf_layer,
        global_loss=gp_transformer.global_loss,
        task_index=task_index,
        gp_of_interest=gp,
    )

    imodel = imodel.to(device)

    # --------------------------
    # Run attribution
    # --------------------------

    # set up attribution
    gc = GuidedGradCam(imodel, imodel.gp_block.blocks[0].mlp)

    attribution_scores = {}
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
                target=conversion_dict[obs_value],
                additional_forward_args=input_ids[1],
            )

            attr_norm = summarize_attributions(attributions).detach().cpu().numpy()

            for i, t in enumerate(token_labels):
                if t in attribution_scores.keys():
                    attribution_scores[t] += [attr_norm[i]]
                    attribution_scores[f'{t}_abs'] += [np.abs(attr_norm[i])]
                else:
                    attribution_scores[t] = [attr_norm[i]]
                    attribution_scores[f'{t}_abs'] = [np.abs(attr_norm[i])]

        else:
            break

    for t in all_tokens:
        attribution_scores[t] = np.nanmean(attribution_scores[t])
        attribution_scores[f'{t}_abs'] = np.nanmean(attribution_scores[f'{t}_abs'])
        attribution_scores[f'{t}_std'] = np.nanstd(attribution_scores[t])
        attribution_scores[f'{t}_abs_std'] = np.nanstd(attribution_scores[f'{t}_abs'])

    rows = []

    for t in all_tokens:
        row = {
            'token': t,
            'attribution_score': attribution_scores[t],
            'abs_attribution_score': attribution_scores[f'{t}_abs'],
            'std_attribution_score': attribution_scores[f'{t}_std'],
            'abs_std_attribution_score': attribution_scores[f'{t}_abs_std'],
        }

        rows.append(row)

    attribution_df = pd.DataFrame(rows)

    # Add gene conversion
    gene_df = pd.DataFrame(imodel.gene_conversion)
    gene_df = gene_df.join(attribution_df.set_index('token'), on='token_original')

    # add GP labels
    if gpdb_ref_path is None:
        gpdb_ref_path = gpdb_path

    # add GP labels
    if gpdb_ref_path is None:
        gpdb_ref_path = gpdb_path

    gpdb_og = pd.read_csv(gpdb_ref_path)

    for ogp in gpdb_og.columns:
        if do_ensembl_conversion:
            gene_df[ogp] = np.where(gene_df['symbol'].isin(gpdb_og[ogp]), 1, 0)
        else:
            gene_df[ogp] = np.where(gene_df['ensembl'].isin(gpdb_og[ogp]), 1, 0)

    gene_df.to_csv(
        os.path.join(output_dir, f'{gp}_attribution_scores_{obs_value}.csv'),
        index=False,
    )


def calculate_cell_token_attribution_scores(
    gpdb_path,
    dataset_path,
    emb_dataset_path,
    data_split,
    n_blocks,
    num_heads,
    gp_latent_size,
    model_checkpoint,
    obs_key,
    obs_value,
    output_dir,
    global_loss,
    # for EmbEvaluator
    emb_label,
    task,
    save_plot=False,
    gp_inputs=None,
    use_embedding=False,
    pretrained_emb=None,
    reconstruction_loss=None,
    supervised_labels=None,
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

    print('emb_dataset_path', emb_dataset_path)
    emb_dm = iEmbDataModule(
        folder_path=emb_dataset_path,
        batch_size=1,
        gp_inputs=list(gpdb.columns),
        meta_labels=obs_key,
    )

    emb_dm.setup()

    dataloader = getattr(emb_dm, data_split + '_dataloader')()

    # --------------------------
    # Build dictionary for class : id conversion
    # --------------------------

    y_label = obs_key + '_id'

    datax = load_from_disk(dataset_path)
    cols_to_remove = datax.column_names
    cols_to_remove.remove(obs_key)
    cols_to_remove.remove(y_label)
    datax = datax.remove_columns(cols_to_remove)
    conversion = datax.to_pandas().drop_duplicates()

    conversion_dict = {k: v for k, v in zip(conversion[obs_key], conversion[y_label])}

    # --------------------------
    # Set up model
    # --------------------------

    model = gpTransformerGlobal(
        gene_counts_df=None,
        database=gpdb,
        do_ensembl_conversion=False,
        n_blocks=n_blocks,
        num_heads=num_heads,
        gp_latent_size=gp_latent_size,
        gp_inputs=list(gpdb.columns),
        add_remaining_var=False,
        global_loss=global_loss,
        reconstruction_loss=reconstruction_loss,
        supervised_labels=supervised_labels,
    )

    gp_transformer = scGPL(
        model,
        'Global',
        global_loss=reconstruction_loss,
        return_gene_embeddings=False,
        tokens_to_keep=None,
        gene_file_tag=None,
        return_attention=False,
        gp=None,
        return_classification_report=False,
    ).load_from_checkpoint(
        model_checkpoint,
        strict=False,
    )

    # Load classification layer
    # or train if not available
    ckpt_dir = os.path.join(output_dir, 'evaluation_model_checkpoints')
    clf_ckpt = f'{y_label.replace("_id", "")}_{emb_label}_{task}'
    if os.path.exists(os.path.join(ckpt_dir, f'{clf_ckpt}.ckpt')):
        clf_layer = EmbEvaluator.load_from_checkpoint(
            os.path.join(ckpt_dir, f'{clf_ckpt}.ckpt')
        )

    else:
        gpEval.evaluate_embeddings(
            y_label=y_label,
            folder_path=emb_dataset_path,
            output_dir=output_dir,
            emb_label=emb_label,
            task=task,
            emb_dim=gp_latent_size,
            lr=1e-3,
            batch_size=128,
            num_workers=1,
            meta_labels=[y_label, y_label.replace('_id', '')],
            data_type='dataset',
            n_epochs=3,
            continuous_cov=[],
        )

        clf_layer = EmbEvaluator.load_from_checkpoint(
            os.path.join(ckpt_dir, f'{clf_ckpt}.ckpt')
        )

    imodel = iGlobalWrapper(
        gp_transformer,
        clf_layer,
        use_embedding=use_embedding,
        pretrained_emb=pretrained_emb,
        vocab_size=len(gpdb.columns),
        embedding_dim=gp_latent_size,
    )
    imodel = imodel.to(device)

    # --------------------------
    # Run attribution
    # --------------------------

    # set up attribution
    gc = GuidedGradCam(imodel, imodel.global_block.encoder.blocks[0].mlp)

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
            'score_std': attribution_scores[f'{g}_std'],
            'score_abs_std': attribution_scores[f'{g}_abs_std'],
        }
        rows.append(row)

    # Convert the list of dictionaries to a DataFrame
    attribution_df = pd.DataFrame(rows)

    attribution_df.to_csv(
        os.path.join(output_dir, f'cell_token_attribution_scores_{obs_value}.csv'),
        index=False,
    )

    if save_plot:
        plt.figure()
        ax = sns.barplot(attribution_df, x='GP', y='scores_abs')

        # for i, bar in enumerate(ax.patches):
        #     x = bar.get_x() + bar.get_width() / 2
        #     y = bar.get_height()
        #     error = attribution_df['score_abs_std'].iloc[i]
        #     plt.errorbar(x, y, yerr=error, fmt='none', capsize=5, color='black')

        ax.set_title(obs_value.capitalize().replace('_', ' '))
        ax.set_ylabel('Absolute attribution score')
        ax.set_xlabel('Gene Program')
        plt.savefig(
            os.path.join(
                output_dir, f'cell_token_abs_attribution_scores_{obs_value}.pdf'
            )
        )
        plt.close()

        # Mean normalized values
        plt.figure()
        ax = sns.barplot(attribution_df, x='GP', y='scores')

        # for i, bar in enumerate(ax.patches):
        #     x = bar.get_x() + bar.get_width() / 2
        #     y = bar.get_height()
        #     error = attribution_df['score_std'].iloc[i]
        #     plt.errorbar(x, y, yerr=error, fmt='none', capsize=5, color='black')

        ax.set_title(obs_value.capitalize().replace('_', ' '))
        ax.set_ylabel('Attribution score')
        ax.set_xlabel('Gene Program')
        plt.savefig(
            os.path.join(output_dir, f'cell_token_attribution_scores_{obs_value}.pdf')
        )
        plt.close()
