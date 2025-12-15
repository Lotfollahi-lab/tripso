####################################################
# Build custom tokenizer to return only GP genes
####################################################

import logging
from collections import Counter
from typing import (
    Dict,
    List,
    Literal,
    Optional,
)

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from datasets import Dataset
from geneformer.tokenizer import TranscriptomeTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def rank_genes(gene_vector, gene_tokens):
    """
    Rank gene expression vector.
    """
    # sort by median-scaled gene values
    sorted_indices = np.argsort(-gene_vector)
    return gene_tokens[sorted_indices]


def sum_ensembl_ids(
    data_directory,
    collapse_gene_ids,
    gene_mapping_dict,
    gene_token_dict,
    custom_attr_name_dict,
    file_format='loom',
    chunk_size=512,
):
    """
    Map Ensembl IDs from gene mapping dictionary.
    If duplicate Ensembl IDs are found, sum counts together.
    Returns adata object with deduplicated Ensembl IDs.
    """

    data = sc.read_h5ad(str(data_directory))

    assert 'ensembl_id' in data.var.columns, "'ensembl_id' column missing from data.var"

    assert (
        'ensembl_id_collapsed' not in data.var.columns
    ), "'ensembl_id_collapsed' column already exists in data.var"
    assert 'n_counts' in data.obs.columns, "'n_counts' column missing from data.obs"

    if custom_attr_name_dict is not None:
        for label in custom_attr_name_dict:
            assert (
                label in data.obs.columns
            ), f'Attribute `{label}` not present in data.obs'

    # Get the ensembl ids that exist in data
    ensembl_ids = data.var.ensembl_id
    # Check for duplicate Ensembl IDs if collapse_gene_ids is False.
    # Comparing to gene_token_dict here, would not perform any mapping steps
    if not collapse_gene_ids:
        ensembl_id_check = [
            gene for gene in ensembl_ids if gene in gene_token_dict.keys()
        ]
        if len(ensembl_id_check) == len(set(ensembl_id_check)):
            return data_directory
        else:
            raise ValueError('Error: data Ensembl IDs non-unique.')

    # Get the genes that exist in the mapping dictionary and the value of those genes
    genes_in_map_dict = [
        gene for gene in ensembl_ids if gene in gene_mapping_dict.keys()
    ]
    vals_from_map_dict = [gene_mapping_dict.get(gene) for gene in genes_in_map_dict]

    # if the genes in the mapping dict and the value
    # of those genes are of the same length,
    # simply return the mapped values
    if len(set(genes_in_map_dict)) == len(set(vals_from_map_dict)):
        data.var['ensembl_id_collapsed'] = data.var.ensembl_id.str.upper().map(
            gene_mapping_dict
        )
        return data
    # Genes need to be collapsed
    else:
        data.var['ensembl_id_collapsed'] = data.var.ensembl_id.str.upper().map(
            gene_mapping_dict
        )
        data.var_names = data.var['ensembl_id_collapsed']
        data = data[:, ~data.var.index.isna()]
        dup_genes = [idx for idx, count in Counter(data.var_names).items() if count > 1]

        num_chunks = int(np.ceil(data.shape[0] / chunk_size))

        processed_genes = []
        for i in tqdm(range(num_chunks)):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, data.shape[0])
            data_chunk = data[start_idx:end_idx, :]

            processed_chunks = []
            for dup_gene in dup_genes:
                data_dup_gene = data_chunk[:, data_chunk.var_names == dup_gene]
                df = pd.DataFrame.sparse.from_spmatrix(
                    data_dup_gene.X,
                    index=data_dup_gene.obs_names,
                    columns=data_dup_gene.var_names,
                )
                df_sum = pd.DataFrame(df.sum(axis=1))
                df_sum.columns = [dup_gene]
                df_sum.index = data_dup_gene.obs.index
                processed_chunks.append(df_sum)

            processed_chunks = pd.concat(processed_chunks, axis=1)
            processed_genes.append(processed_chunks)
        processed_genes = pd.concat(processed_genes, axis=0)
        var_df = pd.DataFrame({'ensembl_id_collapsed': processed_genes.columns})
        var_df.index = processed_genes.columns
        processed_genes = sc.AnnData(X=processed_genes, obs=data.obs, var=var_df)

        data_dedup = data[:, ~data.var.index.isin(dup_genes)]  # Deduplicated data
        data_dedup = sc.concat([data_dedup, processed_genes], axis=1)
        data_dedup.obs = data.obs
        return data_dedup


class GPTokenizer(TranscriptomeTokenizer):
    def __init__(self, gp_genes, do_ensembl_conversion, **kwargs):
        super().__init__(**kwargs)
        self.gp_genes = gp_genes

        # convert gene names to token ids
        if do_ensembl_conversion:
            ensembl_id = [self.gene_mapping_dict.get(gene, None) for gene in gp_genes]
        else:
            ensembl_id = gp_genes

        gp_token_ids = [self.gene_token_dict.get(gene, None) for gene in ensembl_id]
        gp_token_ids = [token for token in gp_token_ids if token is not None]
        self.gp_tokens = set(gp_token_ids)

    def create_dataset(
        self,
        tokenized_cells,
        cell_metadata,
        use_generator=False,
        keep_uncropped_input_ids=False,
    ):
        print('Creating dataset.')
        # create dict for dataset creation

        tokenized_by_gp = []

        for i, genes in enumerate(tokenized_cells):
            # x = list(set(genes) & self.gp_tokens)
            x = [g for g in genes if g in self.gp_tokens]
            tokenized_by_gp += [x]

        dataset_dict = {'input_ids': tokenized_by_gp}

        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        # create dataset
        if use_generator:

            def dict_generator():
                for i in range(len(tokenized_cells)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}

            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            output_dataset = Dataset.from_dict(dataset_dict)

        # filter out cells with no genes
        output_dataset = output_dataset.filter(lambda x: len(x['input_ids']) > 0)

        def format_cell_features(example):
            # Store original uncropped input_ids in separate feature
            if keep_uncropped_input_ids:
                example['input_ids_uncropped'] = example['input_ids']
                example['length_uncropped'] = len(example['input_ids'])

            # Truncate/Crop input_ids to input size
            if self.special_token:
                example['input_ids'] = example['input_ids'][
                    0 : self.model_input_size - 2
                ]  # truncate to leave space for CLS and EOS token

                example['input_ids'] = np.insert(
                    example['input_ids'], 0, self.gene_token_dict.get('<cls>')
                )

                example['input_ids'] = np.insert(
                    example['input_ids'],
                    len(example['input_ids']),
                    self.gene_token_dict.get('<eos>'),
                )
            else:
                # Truncate/Crop input_ids to input size
                example['input_ids'] = example['input_ids'][0 : self.model_input_size]

            example['length'] = len(example['input_ids'])

            return example

        output_dataset_truncated = output_dataset.map(
            format_cell_features, num_proc=self.nproc
        )

        return output_dataset_truncated

    def tokenize_files(
        self, data_directory, file_format: Literal['loom', 'h5ad'] = 'h5ad'
    ):
        tokenized_cells = []
        cell_metadata: Optional[Dict[str, List]] = None
        if self.custom_attr_name_dict is not None:
            cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.values()
            }

        # loops through directories to tokenize .loom files
        file_found = 0
        # loops through directories to tokenize .loom or .h5ad files
        tokenize_file_fn = (
            self.tokenize_loom if file_format == 'loom' else self.tokenize_anndata
        )
        for file_path in data_directory.glob(f'*.{file_format}'):
            file_found = 1
            print(f'Tokenizing {file_path}')
            file_tokenized_cells, file_cell_metadata = tokenize_file_fn(file_path)
            tokenized_cells += file_tokenized_cells
            if self.custom_attr_name_dict is not None and cell_metadata is not None:
                for k in cell_attr:
                    cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[
                        k
                    ]
            else:
                cell_metadata = None

        if file_found == 0:
            logger.error(
                f'No .{file_format} files found in directory {data_directory}.'
            )
            raise
        return tokenized_cells, cell_metadata

    def tokenize_anndata(self, adata_file_path, target_sum=10_000):
        adata = sum_ensembl_ids(
            adata_file_path,
            self.collapse_gene_ids,
            self.gene_mapping_dict,
            self.gene_token_dict,
            self.custom_attr_name_dict,
            file_format='h5ad',
            chunk_size=self.chunk_size,
        )

        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        coding_miRNA_loc = np.where(
            [
                self.genelist_dict.get(i, False)
                for i in adata.var['ensembl_id_collapsed']
            ]
        )[0]
        norm_factor_vector = np.array(
            [
                self.gene_median_dict[i]
                for i in adata.var['ensembl_id_collapsed'][coding_miRNA_loc]
            ]
        )
        coding_miRNA_ids = adata.var['ensembl_id_collapsed'][coding_miRNA_loc]
        coding_miRNA_tokens = np.array(
            [self.gene_token_dict[i] for i in coding_miRNA_ids]
        )

        try:
            _ = adata.obs['filter_pass']
        except KeyError:
            var_exists = False
        else:
            var_exists = True

        if var_exists:
            filter_pass_loc = np.where([i == 1 for i in adata.obs['filter_pass']])[0]
        elif not var_exists:
            print(
                f"{adata_file_path} has no column attribute 'filter_pass';"
                'tokenizing all cells.'
            )
            filter_pass_loc = np.array([i for i in range(adata.shape[0])])

        tokenized_cells = []

        for i in range(0, len(filter_pass_loc), self.chunk_size):
            idx = filter_pass_loc[i : i + self.chunk_size]

            n_counts = adata[idx].obs['n_counts'].values[:, None]
            X_view0 = adata[idx, :].X
            X_view = X_view0[:, coding_miRNA_loc]
            X_norm = X_view / n_counts * target_sum / norm_factor_vector
            X_norm = sp.csr_matrix(X_norm)

            tokenized_cells += [
                rank_genes(X_norm[i].data, coding_miRNA_tokens[X_norm[i].indices])
                for i in range(X_norm.shape[0])
            ]

            # add custom attributes for subview to dict
            if self.custom_attr_name_dict is not None:
                for k in file_cell_metadata.keys():
                    file_cell_metadata[k] += adata[idx].obs[k].tolist()
            else:
                file_cell_metadata = None

        return tokenized_cells, file_cell_metadata
