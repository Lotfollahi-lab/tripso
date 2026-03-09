"""Data Preparation for Tripso Tutorial

This script aims to illustrate how to prepare data for running Tripso
for tokenization. The key steps for compatibility with tokenizatoin are
* adding Ensembl gene IDs to the variable annotations
* ensuring `n_counts` is in the obs columns
Other steps represent minor wrangling of metadata columns for consistency.

Here we will use the bone marrow dataset from Zeng et al.
Available at:
https://cellxgene.cziscience.com/collections/f6c50495-3361-40ed-a819-fb9644396ed9

Inputs:
    - Raw H5AD file with cell annotations

Outputs:
    - data/processed/zeng.h5ad: Processed dataset with standardized metadata
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc

from tripso import ENSEMBL_DICTIONARY_FILE

ensembl_dict = pd.read_pickle(ENSEMBL_DICTIONARY_FILE)

##############################################
print(' ------- Wrangle Zeng ------- ')
# data can be downloaded with
# curl -o zeng.h5ad https://datasets.cellxgene.cziscience.com/96c26450-ad18-4e43-8ec6-c84331bba832.h5ad
##############################################

zeng_as_downloaded = sc.read_h5ad('data/raw/zeng.h5ad')

# reset raw counts
zeng = sc.AnnData(
    X=zeng_as_downloaded.raw.X,
    obs=zeng_as_downloaded.obs.copy(),
    var=zeng_as_downloaded.raw.var.copy(),
)

print(zeng)

# drop duplicate ensembl ids
duplicated_vars = zeng.var_names[zeng.var_names.duplicated()]
zeng = zeng[:, ~zeng.var.index.isin(duplicated_vars)]

zeng_cols = [
    'AuthorCellType',
    'AuthorCellType_Broad',
    'cell_type',
    'Sorting',
    'Study',
    'donor_id',
    'sex',
    'development_stage',
]

zeng.obs = zeng.obs[zeng_cols]
zeng.obs = zeng.obs.rename(columns={'donor_id': 'donor'})

# Wrangle age


def assign_age_group(stage):
    # Handle named stages first
    if stage == 'young adult stage':
        return '18-20'
    if stage == 'prime adult stage':
        return '24-60'
    if stage == 'late adult stage':
        return '65+'

    # Extract numeric age from strings like "29-year-old stage"
    try:
        age = int(stage.split('-')[0])
    except (ValueError, AttributeError):
        return np.nan

    # Assign based on numeric age
    if 18 <= age <= 20:
        return '18-20'
    elif 24 <= age <= 60:
        return '24-60'
    elif age >= 65:
        return '65+'
    else:
        return np.nan


# Create harmonized age group column
zeng.obs['age_group'] = zeng.obs['development_stage'].apply(assign_age_group)

# Use gene symbols
zeng.var['ensembl_id'] = zeng.var.index.tolist()
zeng.var = zeng.var.set_index('gene_symbols')

# add back counts
zeng.obs['n_counts'] = zeng.X.sum(axis=1)

# Save to disk
if not os.path.exists('data/processed/'):
    os.makedirs('data/processed/')
zeng.write_h5ad('data/processed/zeng.h5ad')
