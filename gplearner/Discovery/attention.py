from typing import List, Union, Optional

import os

import numpy as np
import pandas as pd
import scanpy as sc

def rank_genes_by_attn_diff(
    attn_adata_fn: Union[str, os.PathLike],
    target_label: str,
    target_class: str,
    output_fn: Union[str, os.PathLike],
    subset_obs_column: Optional[str] = None,
    subset_obs_values: Union[str, List[str], None] = None,
    subset_adata_fn: Union[str, os.PathLike, None] = None,
):
    attn_adata = sc.read_h5ad(attn_adata_fn)

    # subset the anndata
    if subset_obs_column is not None and subset_obs_values is not None:
        
        if subset_adata_fn is None:
            subset_adata_fn = os.path.splitext(attn_adata_fn)[0] + f'_subset_{subset_obs_column}.h5ad'
            
        if isinstance(subset_obs_values, str):
            subset_obs_values = [subset_obs_values]
        
        attn_adata = attn_adata[attn_adata.obs[subset_obs_column].isin(subset_obs_values)]
        attn_adata.write_h5ad(subset_adata_fn)
        
    # get attention diff
    attn_target_class = attn_adata[attn_adata.obs[target_label] == target_class]
    attn_reverse = attn_adata[attn_adata.obs[target_label] != target_class]
    attn_df = pd.DataFrame({
        'names': attn_adata.var.index[1:],
        'scores_general': np.asarray(attn_adata.X.mean(0)).flatten()[1:], # without cls token
        'scores_target': np.asarray(attn_target_class.X.mean(0)).flatten()[1:], # without cls token
        'scores_diff': (
            np.asarray(attn_target_class.X.mean(0)).flatten()[1:]
            - np.asarray(attn_reverse.X.mean(0)).flatten()[1:]
        ), # without cls token
    })
    attn_df = attn_df.sort_values(by='scores_diff', ascending=False)
    attn_df.rename(columns={'names': 'gene'}, inplace=True)
    attn_df.to_csv(output_fn, index=False)