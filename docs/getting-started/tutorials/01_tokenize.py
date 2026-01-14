"""Cell Tokenization for Tripso Tutorial

This script tokenizes the processed dataset for model training. In our tokenization, 
we will keep genes which are either part of the precomputed highly variable genes (HVGs)
or are included in our curated gene program database (gpdb_tf.csv). 

Inputs:
    - data/processed/mnc.h5ad: Processed MNC dataset
    - gpdb_tf.csv: Gene program database with GP names as column names

Outputs:
    - data/processed/input_dataset/: Tokenized cell data ready for training
    - data/processed/mnc_genes.h5ad: MNC data with gene subset
"""

# =======================================================
# Load gene sets
# =======================================================

import pandas as pd
import scanpy as sc
import os
import gplearner

root_dir = '/lustre/scratch126/cellgen/lotfollahi/mm58/gplearner_reproducibility/06_tutorial_MNC'
os.chdir(root_dir)


# =======================================================
print('---------- Tokenize MNC data ----------')
# =======================================================

mnc = sc.read_h5ad(os.path.join(root_dir, 'data/processed/mnc.h5ad'))
mnc_cols = list(mnc.obs.columns)

hvg = mnc[:, mnc.var['highly_variable']].var_names
gpdb = pd.read_csv('gpdb_tf.csv')
    
all_genes = set()

for i in gpdb.columns:
    all_genes.update(gpdb[i].dropna().values)
    
all_genes.update(hvg)
all_genes = list(all_genes)

print('Number of genes', len(all_genes))

gplearner.pp_and_tokenize(root_dir=root_dir,
                          adata_path = os.path.join(root_dir, 'data/processed/mnc.h5ad'),
                          vars_to_keep = mnc_cols,
                          cov_to_encode = ['cell_type', 'source'],
                          batch_keys = 'donor',
                          subsample_by = None,
                          name_tag='tf',
                          save_gp_genes_object = True,
                          calculate_hvg = False,
                          input_size = 4096,
                          use_gp_tokenizer = True,
                          gp_genes_union = all_genes, 
                          do_ensembl_conversion = True, # convert gene symbols to ensembl IDs
                          tissue = 'mnc'
                          )


