import os
from typing import (
    List,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
from anndata import AnnData


def rank_genes_by_attn_diff(
    attn_adata: AnnData,
    target_label: str,
    target_class: str,
    subset_obs_column: Optional[str] = None,
    subset_obs_values: Union[str, List[str], None] = None,
    save_fn: Union[str, os.PathLike, None] = None,
) -> pd.DataFrame:
    # subset the anndata
    if subset_obs_column is not None and subset_obs_values is not None:
        if isinstance(subset_obs_values, str):
            subset_obs_values = [subset_obs_values]

        attn_adata = attn_adata[
            attn_adata.obs[subset_obs_column].isin(subset_obs_values)
        ]

        if save_fn is not None:
            attn_adata.write_h5ad(save_fn)

    # get attention diff
    attn_target_class = attn_adata[attn_adata.obs[target_label] == target_class]
    attn_reverse = attn_adata[attn_adata.obs[target_label] != target_class]
    attn_df = pd.DataFrame(
        {
            'names': attn_adata.var.index[1:],
            'scores_general': np.asarray(attn_adata.X.mean(0)).flatten()[
                1:
            ],  # without cls token
            'scores_target': np.asarray(attn_target_class.X.mean(0)).flatten()[
                1:
            ],  # without cls token
            'scores_diff': (
                np.asarray(attn_target_class.X.mean(0)).flatten()[1:]
                - np.asarray(attn_reverse.X.mean(0)).flatten()[1:]
            ),  # without cls token
        }
    )
    attn_df = attn_df.sort_values(by='scores_diff', ascending=False)
    attn_df.rename(columns={'names': 'gene'}, inplace=True)

    return attn_df
