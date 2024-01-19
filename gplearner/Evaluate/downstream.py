import os
import random
from typing import (
    List,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)

from ..Datamodules.datamodule import txDataModule
from ..Models.gp_model import gpTransformerBase  # , gfBaseline
from ..Trainers.trainer import scGPL
from ..Utils.utils import (
    do_logistic_regression,
    find_genes_in_multiple_gp,
    find_latest_file,
    get_genes_in_single_gp,
    remove_single_data_points,
    viz_gp,
)

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
    ):
        # check only one GPU
        assert torch.cuda.device_count() == 1, 'Please run evaluation on single GPU'

        # set seed for reproducibility
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)
        torch.manual_seed(seed)

        # Search for .ckpt files in the directory
        if model_type != 'Mean':
            latest_ckpt = find_latest_file(output_dir, tissue, model_type)
            print('Latest .ckpt file:', latest_ckpt)

        gpdb = pd.read_csv(gpdb_path)

        if gene_format == 'symbol':
            do_ensembl_conversion = True
        elif gene_format == 'ensembl':
            do_ensembl_conversion = False
        self.do_ensembl_conversion = do_ensembl_conversion

        if gene_counts_df is not None:
            self.gene_counts_df = pd.read_csv(gene_counts_df)

        if model_type == 'Base':
            self.model = gpTransformerBase(
                gp_inputs=gp_inputs,
                gene_counts_df=gene_counts_df,
                database=gpdb,
                do_ensembl_conversion=do_ensembl_conversion,
                n_blocks=n_blocks,
                num_heads=n_heads,
                gp_latent_size=gp_latent_size,
            )

        # elif model_type == 'Mean':
        #     model = gfBaseline(
        #         gp_inputs=gpdb.columns,
        #         database=gpdb,
        #         do_ensembl_conversion=do_ensembl_conversion,
        #         # dummy variables to avoid errors if no defaults
        #         # but we won't use transformer blocks
        #         n_blocks=1,
        #         mgm_mask_ratio=1,
        #         num_heads=1,
        #     )

        else:
            raise ValueError('model_type must be one of Base, or Mean')

        # Set up gpTransformer main module
        self.checkpoint_path = os.path.join(output_dir, latest_ckpt)

        self.model_type = model_type

        self.gp_transformer = self._init_trainer()

        if gp_inputs is None:
            gp_inputs = gpdb.columns
        self.gp_inputs = gp_inputs

        # change directory for saving outputs
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir
        self.tissue = tissue
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.gpdb = gpdb

    def _init_trainer(
        self,
        return_gene_embeddings=False,
        tokens_to_keep=None,
        gene_file_tag=None,
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
            ).load_from_checkpoint(self.checkpoint_path)
        else:
            gp_transformer = scGPL(
                self.model,
                self.model_type,
                tokens_to_keep=tokens_to_keep,
                gene_file_tag=gene_file_tag,
                return_gene_embeddings=return_gene_embeddings,
            )

        # reset attributes overwritten by loading from checkpoint
        gp_transformer.return_gene_embeddings = return_gene_embeddings
        gp_transformer.tokens_to_keep = tokens_to_keep
        gp_transformer.gene_file_tag = gene_file_tag

        return gp_transformer

    def generate_embeddings(self):
        """
        Generate embeddings for each cell
        """
        os.chdir(self.output_dir)
        txdata = txDataModule(folder=self.dataset_path, batch_size=self.batch_size)

        if os.path.exists('adata_gp_embedding.h5ad'):
            print(f'{self.output_dir}/adata_gp_embedding.h5ad already exists')
        else:
            trainer = pl.Trainer(
                max_epochs=1, devices=-1, accelerator='auto', precision=16
            )
            trainer.test(self.gp_transformer, txdata)

    def load_anndata(self):
        """
        Check that adata object exists
        """
        if os.path.exists('adata_gp_embedding.h5ad'):
            print(f'Loading adata from {self.output_dir}/adata_gp_embedding.h5ad')
            adata = sc.read_h5ad('adata_gp_embedding.h5ad')
        else:
            raise ValueError('No adata found. Please run generate_embeddings() first')

        return adata

    def visualize(self, label_to_plot, gp_to_plot=None):
        """
        UMAP of GP embeddings
        """
        adata = self.load_anndata()

        if isinstance(label_to_plot, str):
            label_to_plot = [label_to_plot]

        for c in label_to_plot:
            adata = remove_single_data_points(adata, c)

        for c in label_to_plot:
            sc.pl.umap(adata, color=c, save=f'_{self.tissue}_{c}.pdf')

        if gp_to_plot is None:
            gp_to_plot = self.gp_inputs
        for gp in gp_to_plot:
            viz_gp(gp, color_by=label_to_plot, adata=adata, save_to=self.tissue)

    def _evaluate_clustering_cells(self, adata):
        if 'leiden' not in adata.obs.columns:
            sc.tl.leiden(adata)

        print('Running cluster evaluation metrics...')
        ari_ct = adjusted_rand_score(adata.obs['leiden'], adata.obs['cell_type'])
        ari_cond = adjusted_rand_score(adata.obs['leiden'], adata.obs['condition'])
        sil = silhouette_score(adata.obsm['X_umap'], adata.obs['leiden'])
        db = davies_bouldin_score(adata.obsm['X_umap'], adata.obs['leiden'])

        output_df = pd.DataFrame(
            {
                'metric': [
                    'ARI_cell',
                    'ARI_env',
                    'Silhouette_leiden',
                    'Davies_Bouldain_leiden',
                ],
                'value': [ari_ct, ari_cond, sil, db],
            }
        )

        return output_df

    def feature_analysis(self, label_to_plot, rank_genes=True, cluster_latent=True):
        if isinstance(label_to_plot, str):
            label_to_plot = [label_to_plot]

        adata = self.load_anndata()

        if rank_genes:
            for c in label_to_plot:
                sc.tl.rank_genes_groups(adata, c)
                sc.pl.rank_genes_groups(
                    adata, n_genes=25, sharey=False, save=f'_{self.tissue}_by_{c}.pdf'
                )

        if cluster_latent:
            # make cluster metrics directory
            if not os.path.exists('cluster_metrics'):
                os.makedirs('cluster_metrics')

            df = self._evaluate_clustering_cells(adata)
            df.to_csv(
                os.path.join('cluster_metrics', 'latent_space_clustering_metrics.csv'),
                index=False,
            )

            # now for each gp:
            for gp in self.gp_inputs:
                df = self._evaluate_clustering_cells(
                    adata[:, adata.var['gp_idx'].str.startswith(gp)]
                )
                df.to_csv(
                    os.path.join(
                        'cluster_metrics', f'{gp}_latent_space_clustering_metrics.csv'
                    ),
                    index=False,
                )

    def logistic_regression(
        self,
        gp_features: Union[List, str] = 'all',
        labels=['cell_type', 'condition'],
        data_to_model='cell',  # "cell", "gene_singleGP" or "gene_multiGP"
        min_cells=500,
        downsample_to_n_genes=50,
    ):
        """
        Run logistic regression to predict labels from features

        Parameters
        ----------
        gp_features : list
            List of GP to use as features
            If "all", will use all GP
            If "concat", will concatenate all GP
        labels : list
            List of labels to predict
        model_cells : bool
            If True, will model cells
            If False, will model genes
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

        if data_to_model == 'cell':
            adata = self.load_anndata()

            if not os.path.exists('cell_metrics'):
                os.makedirs('cell_metrics')

            if gp_features == 'concat':
                for c in labels:
                    do_logistic_regression(
                        adata,
                        c,
                        os.path.join(self.output_dir, 'cell_metrics'),
                        f'{c}_prediction_from_concat_gp',
                    )

            else:
                if gp_features == 'all':
                    gp_features = self.gp_inputs
                    gp_features = list(gp_features)

                for gp in gp_features:
                    for c in labels:
                        do_logistic_regression(
                            adata=adata[:, adata.var['gp_idx'].str.startswith(gp)],
                            labels_var=c,
                            output_directory=os.path.join(
                                self.output_dir, 'cell_metrics'
                            ),
                            filename=f"{c}_prediction_{gp.replace('/', '_')}",
                            variable_to_track={'GP': gp},
                        )

        elif data_to_model == 'gene_mutliGP':
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
