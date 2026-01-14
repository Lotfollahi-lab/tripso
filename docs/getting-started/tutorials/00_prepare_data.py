"""Data Preparation for Tripso Tutorial

This script aims to prepare our novel Mononuclear Cell (MNC) dataset 
for tokenization. The key steps for compatibility with tokenizatoin are 
* adding Ensembl gene IDs to the variable annotations
* ensuring `n_counts` is in the obs columns
Other steps represent minor wrangling of metadata columns for consistency.

Inputs:
    - Raw MNC H5AD file with cell annotations

Outputs:
    - data/processed/mnc.h5ad: Processed MNC dataset with standardized metadata
"""

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os

from geneformer import ENSEMBL_DICTIONARY_FILE
ensembl_dict = pd.read_pickle(ENSEMBL_DICTIONARY_FILE)

##############################################
print(' ------- Wrangle MNC ------- ')
##############################################

mnc = sc.read_h5ad(
    '/lustre/scratch126/cellgen/lotfollahi/mm58/gplearner_reproducibility/04.5_HSC_post_qc/data/raw/MNC_RNA_79583cells_rm_stroma_annotated.h5ad'
)

# Reset raw counts
mnc.X = mnc.layers['counts']
# Wrangle obs columns

mnc_cols = [
    'runid_mrna_sample', 'sorting', 'biological_replicate_labID',
    'age', 'sex', 'tissue', 'age_general',
     'phase',
     'celltype', 'donor_tissue'
    # 'assignment_id', 'mrna_samples', 'runid',  'runid_prot_samples', 'prot_samples',  'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'pct_counts_in_top_20_genes', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'S_score', 'G2M_score', 'celltype_v1', 'leiden',
       ]

# fix tissue column

# separate young and aged in adult BM
# categorise age groups { end with PCW : 'Fetal, 0-15: Pediatric, 16-30 : Young Adult, 31-50:  Middle Age, 50+: Aged}
mnc.obs['age_group'] = None
mnc.obs.loc[(mnc.obs['age'].str.contains('PCW')) , 'age_group'] = 'Fetal'
mnc.obs.loc[(mnc.obs['age']=='0') , 'age_group'] = 'Cord Blood'
# replace all PCW rows with empty string, e.g. 14PCW -> ''
mnc.obs['age'] = mnc.obs['age'].str.replace(r'\d+PCW', '', regex=True)
mnc.obs['age'] = mnc.obs['age'].replace('', np.nan)
mnc.obs['age'] = mnc.obs['age'].astype(float)

# distinguish between aged bone marrow Aged (60+) and young (<60)
mnc.obs['tissue'] = pd.Categorical(mnc.obs['tissue'])
mnc.obs['tissue'] = mnc.obs['tissue'].cat.add_categories(['ABM_+60y', 'ABM_29-50y']) # 'PBM'
mnc.obs.loc[mnc.obs['age'] >= 60, 'tissue'] = 'ABM_+60y'
mnc.obs.loc[(mnc.obs['age'] < 60) & (mnc.obs['age'] >= 17), 'tissue'] = 'ABM_29-50y'

mnc.obs['tissue'] = mnc.obs['tissue'].cat.remove_unused_categories()


mnc.obs['tissue'] = mnc.obs['tissue'].cat.reorder_categories(
    ['YS', 'FL', 'FBM', 'CB', 'ABM_29-50y', 'ABM_+60y']
    )

mnc.obs = mnc.obs[mnc_cols]
mnc.obs = mnc.obs.rename(columns = {'celltype' : 'cell_type'})
mnc.obs['tissue'] = mnc.obs['donor_tissue'].str.split('_', n=2).str[1]
mnc.obs['donor'] = mnc.obs['donor_tissue'].str.split('_', n=2).str[0]
mnc.obs['source'] = 'in vivo'
mnc.obs['tissue_source'] = mnc.obs['source'].astype(str) + '_' + mnc.obs['tissue'].astype(str)

# Recalculate total counts
mnc.obs['n_counts'] = mnc.X.sum(axis = 1)

# add ensembl IDs
mnc.var = mnc.var.rename(columns = {'gene_ids' : 'ensembl_id'})

# Drop extra fields
del mnc.uns
del mnc.obsm
del mnc.obsp

# Save
os.makedirs('data/processed/', exist_ok=True)

mnc.obs['study'] = 'Isobe_MNC'
mnc.write_h5ad('data/processed/mnc.h5ad')
