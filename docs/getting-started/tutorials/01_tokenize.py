"""Cell Tokenization for Tripso Tutorial

This script tokenizes the processed dataset for model training. In our tokenization,
we will keep genes which are either part of the precomputed highly variable genes (HVGs)
or are included in our curated gene program database (gpdb_tf.csv).

Inputs:
    - data/processed/zeng.h5ad: Processed zeng dataset
    - gpdb_tf.csv: Gene program database with GP names as column names

Outputs:
    - data/processed/input_dataset/: Tokenized cell data ready for training
    - data/processed/zeng_genes.h5ad: zeng data with gene subset
"""

# =======================================================
# Load gene sets
# =======================================================

import os

import pandas as pd
import scanpy as sc

import tripso

# Set working directory to the tutorial folder
root_dir = '/path/to/your/folder/07_tutorial_zeng'
os.chdir(root_dir)


# =======================================================
print('---------- Tokenize zeng data ----------')
# =======================================================

zeng = sc.read_h5ad(os.path.join(root_dir, 'data/processed/zeng.h5ad'))
zeng_cols = list(zeng.obs.columns)

hvg = zeng[:, zeng.var['HVG_intersect3000']].var_names
gpdb = pd.read_csv('gpdb_tf.csv')

all_genes = set()

for i in gpdb.columns:
    all_genes.update(gpdb[i].dropna().values)

all_genes.update(hvg)
all_genes = list(all_genes)

print('Number of genes', len(all_genes))

tripso.pp_and_tokenize(
    root_dir=root_dir,
    adata_path=os.path.join(root_dir, 'data/processed/zeng.h5ad'),
    vars_to_keep=zeng_cols,
    cov_to_encode=['cell_type', 'age_group'],
    batch_keys='Study',
    subsample_by=None,
    name_tag='tf',
    save_gp_genes_object=True,
    calculate_hvg=False,
    input_size=4096,
    use_gp_tokenizer=True,
    gp_genes_union=all_genes,
    do_ensembl_conversion=True,  # convert gene symbols to ensembl IDs
    tissue='zeng',
)
