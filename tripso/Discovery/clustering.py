from typing import Optional, Sequence

import anndata
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def cluster(
    sim: np.ndarray,
    num_clusters: Optional[int] = None,
    num_cluster_candidates: Sequence[int] = np.arange(2, 10),
    seed: int = 0,
) -> np.ndarray:
    """
    Clusters variables according to a similarity matrix.

    Parameters
    ----------
    sim : np.ndarray
        Similarity matrix.
    num_clusters : int
        Number of clusters. If not specified, infers num_cluster
        using a silhouette score criterion.
    num_cluster_candidates : sequence of int
        Candidates for num_clusters, if num_clusters is
        not specified.
    seed : int
        Random seed for spectral clustering.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each variable.
    """
    if num_clusters is None:
        max_silhouette = 0
        for n_cluster in num_cluster_candidates:
            if n_cluster >= sim.shape[0]:
                continue
            clustering = SpectralClustering(
                n_clusters=n_cluster, affinity='precomputed', random_state=seed
            )
            labels = clustering.fit_predict(sim)
            silhouette = silhouette_score(1 - sim, labels, metric='precomputed')
            if silhouette > max_silhouette:
                num_clusters = n_cluster
                max_silhouette = silhouette
    clustering = SpectralClustering(
        n_clusters=num_clusters, affinity='precomputed', random_state=seed
    )
    labels = clustering.fit_predict(sim)
    return labels


def rerank_genes(
    score_df: pd.DataFrame,
    attention_adata: anndata.AnnData,
    col_name: str = 'scores_diff',
) -> pd.DataFrame:
    """
    Re-orders the gene ranking to reflect attention-correlated
    clusters of high-attention genes.

    Parameters
    ----------
    score_df : pandas.DataFrame
        Raw attention score dataframe.
    attention_adata : str or path
        Attention anndata object.
    col_name : str
        Column name in score_fn that indicates score.

    Returns
    -------
    score_df_reranked : pandas.DataFrame
        Re-ranked attention score dataframe.
    """
    score_df = score_df.sort_values(by=col_name, ascending=False)
    scores = np.array(score_df[col_name].tolist())
    genes = np.array(score_df['gene'].tolist())

    adata_genes = attention_adata.var.index.tolist()

    # Cluster based on score
    distance_score = np.abs(scores[:, None] - scores[None, :])
    similarity_score = 1 - distance_score
    labels_score = cluster(similarity_score)
    score_df['labels_score'] = pd.Series(labels_score)

    cluster_score_mean_scores = np.array(
        [scores[labels_score == c].mean() for c in np.unique(labels_score)]
    )
    top_score_cluster = np.argmax(cluster_score_mean_scores)

    # Cluster top score cluster based on score correlation
    top_score_cluster_scores = scores[labels_score == top_score_cluster]
    top_score_cluster_genes = genes[labels_score == top_score_cluster]

    gene_value_list = []

    for gene in tqdm(top_score_cluster_genes):
        X_gene = attention_adata.X[:, adata_genes.index(gene)]
        X_gene_dense = np.asarray(X_gene.todense())
        gene_value_list.append(X_gene_dense[:, 0])

    gene_value_mat = np.stack(gene_value_list, axis=-1)  # (n_cells, n_genes)

    corr_matrix = np.corrcoef(gene_value_mat, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix)
    np.fill_diagonal(corr_matrix, 1.0)
    similarity_corr = np.abs(corr_matrix)

    labels_corr = cluster(similarity_corr)

    # Assign labels_corr only to top_score_cluster genes
    # Initialize with NaN
    score_df['labels_corr'] = np.nan

    # Assign labels_corr only to genes in top_score_cluster
    # Find the indices in score_df where labels_score == top_score_cluster
    top_cluster_mask = score_df['labels_score'] == top_score_cluster
    score_df.loc[top_cluster_mask, 'labels_corr'] = labels_corr

    cluster_corr_mean_scores = np.array(
        [
            top_score_cluster_scores[labels_corr == c].mean()
            for c in np.unique(labels_corr)
        ]
    )
    cluster_corr_order = np.argsort(cluster_corr_mean_scores)[::-1]

    # Re-compile gene list
    reordered_genes = []
    for c in cluster_corr_order:
        reordered_genes.extend(top_score_cluster_genes[labels_corr == c])
    reordered_genes.extend(genes[labels_score != top_score_cluster])

    # Create dataframe
    score_df_reranked = score_df.copy()
    score_df_reranked['gene'] = pd.Categorical(
        score_df_reranked['gene'], categories=reordered_genes, ordered=True
    )
    score_df_reranked = score_df_reranked.sort_values('gene')

    return score_df_reranked
