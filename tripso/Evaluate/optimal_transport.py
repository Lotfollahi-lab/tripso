import anndata as ad
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.colors import (
    ListedColormap,
    to_hex,
    to_rgb,
)
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def compute_sinkhorn(adata, col, ref, target, epsilon=1e-3, tau_a=0.999, tau_b=0.999):
    # Data selection
    ref_adata_ = adata[adata.obs[col] == ref]
    query_adata_ = adata[adata.obs[col] == target]

    # Ensure the data is dense
    if issparse(ref_adata_.X):
        ref_adata_.X = ref_adata_.X.A
    if issparse(query_adata_.X):
        query_adata_.X = query_adata_.X.A

    # Extract embedding arrays
    x = jnp.array(ref_adata_.X)
    y = jnp.array(query_adata_.X)

    # Set up
    geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
    solver = sinkhorn.Sinkhorn()

    # define an unbalanced linear problem
    ot_prob = linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
    ot = solver(ot_prob)

    return ot


def wrangle_mapping(
    ot, label_source, label_target, order_source=None, order_target=None
):
    '''
    Given the OTT output, wrangle the mapping into a DataFrame where the rows
    represent source labels and the columns represent target labels.
    The DataFrame is sorted according to the provided order.
    '''
    # Check labels
    if order_source:
        if set(label_source) != set(order_source):
            raise ValueError(
                'Mismatch between source labels and categories.'
                f'\nProvided categories are {set(label_source)}, '
                f'but categories are {set(order_source)}'
            )

    if order_target:
        if set(label_target) != set(order_target):
            raise ValueError(
                'Mismatch between target labels and categories.'
                f'\nProvided categories are {set(label_target)}, '
                f'but categories are {set(order_target)}'
            )

    # Extract transport matrix
    ott_out = np.array(ot.matrix)

    # Convert into dataframe
    mapping = pd.DataFrame(ott_out, index=label_source, columns=label_target)

    # Reorder
    if order_source:
        mapping.index = pd.Categorical(
            mapping.index, categories=order_source, ordered=True
        )
        mapping = mapping.sort_index()

    if order_target:
        mapping = mapping.iloc[
            :,
            sorted(
                range(mapping.shape[1]),
                key=lambda i: order_target.index(mapping.columns[i])
                if mapping.columns[i] in order_target
                else float('inf'),
            ),
        ]

    return mapping


def compute_point_cloud_mapping(
    x: jnp.ndarray,
    y: jnp.ndarray,
    matrix: jnp.ndarray,
    adata_ref: ad.AnnData,
    adata_target: ad.AnnData,
    label_col: str,
    threshold=1e-7,
    closest_ref_idx=None,
    closest_query_idx=None,
    use_umap_coordinates=True,
):
    """
    Compute the lines representing the mapping between the 2 point clouds.
    Adapted from https://github.com/ott-jax/ott/blob/main/src/ott/tools/plot.py#L140
    """
    # Only plot the lines with a cost above the threshold.
    u, v = jnp.where(matrix > threshold)
    c = matrix[jnp.where(matrix > threshold)]

    if use_umap_coordinates:
        xy = jnp.concatenate([x[u], y[v]], axis=-1)
    else:
        xy = None

    # Check if we want to adjust transparency.
    scale_alpha_by_coupling = True

    # We can only adjust transparency if max(c) != min(c).
    if scale_alpha_by_coupling:
        min_matrix, max_matrix = jnp.min(c), jnp.max(c)
        scale_alpha_by_coupling = max_matrix != min_matrix

    result = []
    mapping_output = []

    for i in range(len(u)):
        strength = jnp.max(jnp.array(matrix.shape)) * c[i]
        if scale_alpha_by_coupling:
            normalized_strength = (c[i] - min_matrix) / (max_matrix - min_matrix)
            alpha = 0.7 * float(normalized_strength)
        else:
            alpha = 0.7

        # Matplotlib's transparency is sensitive to numerical errors.
        alpha = np.clip(alpha, 0.0, 1.0)

        # Add plotting data only if use_umap_coordinates is True
        if use_umap_coordinates and xy is not None:
            # from the ith row, first pick the elements at columns 0 and 2
            # then pick the elements at columns 1 and 3
            # nb this is because matplotlib expects the coordinates of a line to be
            # in the form ([x0, x1], [y0, y1])
            start, end = xy[i, [0, 2]], xy[i, [1, 3]]
            result.append((start, end, strength, alpha))

        # u = row coordinates of where matrix > threshold
        # v = column coordinates of where matrix > threshold
        matrix_ref_idx = int(u[i])
        matrix_target_idx = int(v[i])

        # Handle index mapping based on whether we're using centroids or direct mapping
        if closest_ref_idx is not None and closest_query_idx is not None:
            # We're using centroids, so map matrix indices to actual cell indices
            ref_idx = closest_ref_idx[matrix_ref_idx]
            target_idx = closest_query_idx[matrix_target_idx]
            # print(f"Centroid mapping: ref_idx={ref_idx}, target_idx={target_idx}")
        else:
            # Direct mapping - matrix indices correspond to cell indices
            ref_idx = matrix_ref_idx
            target_idx = matrix_target_idx
            # print(f"Direct mapping: ref_idx={ref_idx}, target_idx={target_idx}")

        # Get the global indices in adata_ref and adata_target
        adata_ref_idx = adata_ref.obs.index[ref_idx]
        adata_target_idx = adata_target.obs.index[target_idx]
        # print(f"adata_ref_idx: {adata_ref_idx}, adata_target_idx: {adata_target_idx}")

        ct1 = adata_ref.obs.iloc[ref_idx][label_col]
        ct2 = adata_target.obs.iloc[target_idx][label_col]
        # print(f"ct1: {ct1}, ct2: {ct2}")
        # print('')

        # Use the 'idx' column from obs if it exists, otherwise use the index
        if 'idx' in adata_ref.obs.columns:
            idx1 = adata_ref.obs.iloc[ref_idx]['idx']
        else:
            idx1 = adata_ref_idx

        if 'idx' in adata_target.obs.columns:
            idx2 = adata_target.obs.iloc[target_idx]['idx']
        else:
            idx2 = adata_target_idx

        coupling = np.asarray(c[i])
        ns = np.asarray(normalized_strength)
        mapping_output.append([idx1, ct1, idx2, ct2, coupling, ns])

    mapping_df = pd.DataFrame(mapping_output)
    mapping_df.columns = [
        'idx1',
        'source',
        'idx2',
        'target',
        'coupling',
        'normalized_strength',
    ]

    return result, mapping_df


def compute_centroid_mapping(
    adata,
    col,
    ref,
    target,
    label_col,
    num_clusters=10,
    resolution=1,
    threshold=1e-8,
    epsilon=1e-3,
    tau_a=0.999,
    tau_b=0.999,
    return_mapping=False,
    cluster_algo=None,
    cluster_col=None,
    use_umap_coordinates=True,
):
    # Data preparation
    X = adata[adata.obs[col] == ref].X
    Y = adata[adata.obs[col] == target].X

    adata_ref = adata[adata.obs[col] == ref].copy()
    adata_target = adata[adata.obs[col] == target].copy()

    # ----------------------------------------------------------------------
    # Optionally find cluster centroids
    # ----------------------------------------------------------------------

    if cluster_algo == 'knn':
        kmeans_ref = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
        kmeans_query = KMeans(n_clusters=num_clusters, random_state=0).fit(Y)

        # Assign labels to cells as 'knn_cluster' in adata_ref and adata_target
        # as type str to match leiden
        adata_ref.obs['knn_cluster'] = kmeans_ref.labels_.astype(str)
        adata_target.obs['knn_cluster'] = kmeans_query.labels_.astype(str)
        clusters_ref = adata_ref.obs['knn_cluster'].unique()
        clusters_target = adata_target.obs['knn_cluster'].unique()

        cluster_col = 'knn_cluster'

        # Get cluster centroids
        centroids_ref = kmeans_ref.cluster_centers_
        centroids_query = kmeans_query.cluster_centers_

    elif cluster_algo == 'leiden':
        cluster_col = 'leiden'
        sc.pp.neighbors(adata_ref, use_rep='X')
        sc.tl.leiden(adata_ref, resolution=resolution, key_added='leiden')

        sc.pp.neighbors(adata_target, use_rep='X')
        sc.tl.leiden(adata_target, resolution=resolution, key_added='leiden')

        clusters_ref = adata_ref.obs['leiden'].astype(int).unique()
        clusters_target = adata_target.obs['leiden'].astype(int).unique()

        centroids_ref = np.array(
            [
                X[adata_ref.obs['leiden'].astype(int) == cluster].mean(axis=0)
                for cluster in clusters_ref
            ]
        )
        centroids_query = np.array(
            [
                Y[adata_target.obs['leiden'].astype(int) == cluster].mean(axis=0)
                for cluster in clusters_target
            ]
        )

    elif cluster_algo == 'precomputed':
        if cluster_col not in adata.obs.columns:
            raise ValueError(
                'Please provide a `cluster_col` argument'
                'with the values of precomputed clusters.'
            )
        clusters_ref = adata_ref.obs[cluster_col].unique()
        clusters_target = adata_target.obs[cluster_col].unique()

        centroids_ref = np.array(
            [
                X[adata_ref.obs[cluster_col] == cluster].mean(axis=0)
                for cluster in clusters_ref
            ]
        )
        centroids_query = np.array(
            [
                Y[adata_target.obs[cluster_col] == cluster].mean(axis=0)
                for cluster in clusters_target
            ]
        )

    if cluster_algo is not None:
        # Find the actual points closest to centroids,
        # ensuring they belong to the respective clusters
        closest_ref_idx = []
        for i, centroid in enumerate(centroids_ref):
            cluster_points = X[adata_ref.obs[cluster_col] == str(clusters_ref[i])]
            cluster_indices = np.where(
                adata_ref.obs[cluster_col] == str(clusters_ref[i])
            )[0]
            closest_point_idx = cluster_indices[
                np.argmin(cdist(cluster_points, [centroid]))
            ]
            closest_ref_idx.append(closest_point_idx)

        closest_query_idx = []
        for i, centroid in enumerate(centroids_query):
            cluster_points = Y[adata_target.obs[cluster_col] == str(clusters_target[i])]
            cluster_indices = np.where(
                adata_target.obs[cluster_col] == str(clusters_target[i])
            )[0]
            closest_point_idx = cluster_indices[
                np.argmin(cdist(cluster_points, [centroid]))
            ]
            closest_query_idx.append(closest_point_idx)

        closest_ref_idx = np.array(closest_ref_idx, dtype=int)
        closest_query_idx = np.array(closest_query_idx, dtype=int)

        source_idx = adata.obs.loc[adata.obs[col] == ref].index[closest_ref_idx].values
        target_idx = (
            adata.obs.loc[adata.obs[col] == target].index[closest_query_idx].values
        )

        combined_indices = np.concatenate([source_idx, target_idx])
        adata_centroid = adata[adata.obs.index.isin(combined_indices)].copy()

    else:
        adata_centroid = adata
        closest_ref_idx = None
        closest_query_idx = None

    # ----------------------------------------------------------------------
    # Compute optimal transport
    # ----------------------------------------------------------------------

    ot_out = compute_sinkhorn(
        adata_centroid, col, ref, target, epsilon=epsilon, tau_a=tau_a, tau_b=tau_b
    )

    print('Sinkhorn algorithm converged?', ot_out.converged)

    # Only compute UMAP coordinates if use_umap_coordinates is True
    if use_umap_coordinates:
        ref_umap = adata_centroid[adata_centroid.obs[col] == ref].obsm['X_umap']
        target_umap = adata_centroid[adata_centroid.obs[col] == target].obsm['X_umap']

        # Ensure they are numpy arrays of numbers
        ref_umap_jnp = jnp.array(np.asarray(ref_umap))
        target_umap_jnp = jnp.array(np.asarray(target_umap))
    else:
        # Create dummy arrays for the function call (they won't be used for plotting)
        ref_umap_jnp = jnp.array([[0, 0]])  # Dummy array
        target_umap_jnp = jnp.array([[0, 0]])  # Dummy array

    point_map_centroid, mapping_df = compute_point_cloud_mapping(
        x=ref_umap_jnp,
        y=target_umap_jnp,
        matrix=ot_out.matrix,
        adata_ref=adata[adata.obs[col] == ref],
        adata_target=adata[adata.obs[col] == target],
        label_col=label_col,
        threshold=threshold,
        closest_ref_idx=closest_ref_idx,
        closest_query_idx=closest_query_idx,
        use_umap_coordinates=use_umap_coordinates,
    )

    mapping_df['coupling'] = mapping_df['coupling'].astype(float)

    if return_mapping:
        return (
            point_map_centroid,
            closest_ref_idx,
            closest_query_idx,
            mapping_df,
            ot_out,
        )
    else:
        return point_map_centroid, closest_ref_idx, closest_query_idx, mapping_df


# --------------------------------------------
#  Heatmaps
# --------------------------------------------


def make_contingency_table(
    df,
    by='source',
    labels_source=None,
    labels_target=None,
    use_label_order=False,
    fig_size=(6, 5),
    save_to=None,
):
    """
    Processes the dataframe to keep only the row
    with the largest 'coupling' for each 'idx1',
    then computes the crosstabulation of 'source' and 'target',

    Parameters:
    df (pd.DataFrame): The input DataFrame with columns
        ['idx1', 'source', 'idx2', 'target', 'coupling', 'normalized_strength']

    by (str): Specifies whether to group by 'source' or 'target'

    labels_source (list): The list of source labels to include in the crosstabulation

    labels_target (list): The list of target labels to include in the crosstabulation

    use_label_order (bool): Specifies whether to use the order of
        labels_source and labels_target

    fig_size (tuple): The size of the figure

    save_to (str): The path to save the figure

    """
    if by == 'source':
        idx_col = 'idx1'
    elif by == 'target':
        idx_col = 'idx2'
    else:
        raise ValueError("by must be either 'source' or 'target'")

    required_cols = {'idx1', 'idx2', 'source', 'target', 'coupling'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'DataFrame is missing required columns: {sorted(missing)}')

    work = df.copy()
    work.columns = work.columns.str.strip()
    for c in ['idx1', 'idx2', 'source', 'target']:
        work[c] = work[c].astype(str).str.strip()
    work['coupling'] = pd.to_numeric(work['coupling'], errors='coerce')
    work = work.dropna(subset=['coupling'])

    # Keep strongest row for exact idx1-idx2 duplicates.
    work = work.sort_values('coupling', ascending=False).drop_duplicates(
        subset=['idx1', 'idx2'], keep='first'
    )

    # Keep strongest partner per selected index.
    retained = work.sort_values('coupling', ascending=False).drop_duplicates(
        subset=[idx_col], keep='first'
    )

    crosstab = pd.crosstab(retained['source'], retained['target'])

    if labels_source:
        missing_sources = set(labels_source) - set(crosstab.index)
        for source in missing_sources:
            crosstab.loc[source] = 0
        crosstab.index = pd.CategoricalIndex(
            crosstab.index, categories=labels_source, ordered=True
        )
        crosstab = crosstab.sort_index(axis=0)

    if labels_target:
        missing_targets = set(labels_target) - set(crosstab.columns)
        for target in missing_targets:
            crosstab[target] = 0
        if use_label_order:
            crosstab.columns = pd.CategoricalIndex(
                crosstab.columns, categories=labels_target, ordered=True
            )
            crosstab = crosstab.sort_index(axis=1)

    # # if no explicit order, reorder columns to maximize diagonal.
    # if not use_label_order and linear_sum_assignment is not None and not crosstab.empty:
    #     cost_matrix = -crosstab.values
    #     _, col_ind = linear_sum_assignment(cost_matrix)
    #     ordered_targets = crosstab.columns[col_ind]
    #     crosstab = crosstab.loc[:, ordered_targets]

    plt.figure(figsize=fig_size)
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Target')
    plt.ylabel('Source')
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to)

    plt.show()

    return crosstab


def plot_mapping_heatmap(
    mapping_df, order_source=None, order_target=None, normalize=None, fig_size=(6, 5)
):
    """
    Plots a heatmap of the given data, with an option to normalize rows or columns.

    Parameters:
    heatmap_data (pd.DataFrame): The data to plot in the heatmap.
    normalize (str): Specifies whether to normalize rows, columns, or none.
                     Options are 'rows', 'columns', or 'none'.
    """
    # Pivot
    heatmap_data = mapping_df.pivot_table(
        index='source', columns='target', values='coupling', aggfunc='sum'
    ).fillna(0)

    # Order source
    if order_source:
        heatmap_data.index = pd.CategoricalIndex(
            heatmap_data.index, categories=order_source, ordered=True
        )
        heatmap_data = heatmap_data.sort_index(axis=0)

    # Order target
    if order_target:
        heatmap_data.columns = pd.CategoricalIndex(
            heatmap_data.columns, categories=order_target, ordered=True
        )
        heatmap_data = heatmap_data.sort_index(axis=1)

    # Ensure the data is numeric
    heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')
    heatmap_data = heatmap_data.astype(float)

    # Normalize rows or columns if specified
    if normalize == 'row':
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)
    elif normalize == 'column':
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)

    # Create the heatmap
    plt.figure(figsize=fig_size)
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='viridis', cbar=True)

    # Add labels and title
    if normalize:
        plt.title(f'Transport plan, \nnormalized by {normalize}')
    else:
        plt.title('Transport plan')
    plt.xlabel('Target labels')
    plt.ylabel('Source labels')
    plt.show()


def plot_gp_assignment_heatmap(
    leiden_to_pred: pd.DataFrame,
    predefined_order_row: list,
    predefined_order_column=None,
    x_label='Leiden Clusters',
    y_label='Reference cell types',
    fig_size=(10, 8),
    save_to=None,
    show_unmapped=True,
):
    """
    Plots a heatmap where columns are Leiden cluster indices,
    rows are the assigned categories in a predefined order,
    and the values of the heatmap are the number of embeddings where
    Leiden cluster j is assigned to class i.

    Parameters:
    - leiden_to_pred: pd.DataFrame, where columns are embedding names,
        values are assigned classes,
        and the index are Leiden clusters.
    - predefined_order_row: list of assigned categories
        in the order you want them to appear on the y-axis.
    - predefined_order_column: list of Leiden cluster indices
        in the order you want them to appear on the x-axis.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - fig_size: Tuple defining the size of the figure.
    - save_to: Path to save the resulting figure. If None, does not save.
    - show_unmapped: Boolean indicating whether to include the 'Unmapped'
        category in the plot.
    """

    # Ensure 'Unmapped' is included in the predefined order if show_unmapped is True
    if show_unmapped and 'Unmapped' not in predefined_order_row:
        predefined_order_row.append('Unmapped')

    # Transpose DataFrame to ensure Leiden clusters are columns and embeddings are rows
    transposed = leiden_to_pred.T

    # Fill NaN values in the DataFrame with 'Unmapped'
    transposed = transposed.fillna('Unmapped')

    # Count the number of times each class appears for each Leiden cluster
    category_counts = (
        transposed.apply(lambda col: col.value_counts()).fillna(0).astype(int)
    )

    # Reindex to ensure the order of categories on the y-axis
    category_counts = category_counts.reindex(predefined_order_row, axis=0).fillna(0)
    category_counts = category_counts.astype(int)

    # Optionally drop the 'Unmapped' category if show_unmapped is False
    if not show_unmapped:
        category_counts = category_counts.drop('Unmapped', axis=0, errors='ignore')

    # Match predefined order
    if predefined_order_column:
        # Reindex the columns to match the predefined order
        category_counts = category_counts.loc[:, predefined_order_column]
    else:
        # Optimize the order of the Leiden clusters to maximize the diagonal
        num_rows, num_cols = category_counts.shape

        # Pad the cost matrix with dummy rows if necessary
        if num_cols > num_rows:
            padding = np.zeros((num_cols - num_rows, num_cols))
            padded_cost_matrix = np.vstack((-category_counts.values, padding))
        else:
            padded_cost_matrix = -category_counts.values

        # Apply linear sum assignment
        row_ind, col_ind = linear_sum_assignment(padded_cost_matrix)

        # Reorder the columns of category_counts based on the optimal assignment
        category_counts = category_counts.iloc[:, col_ind]

    # Create the heatmap
    plt.figure(figsize=fig_size)
    sns.heatmap(
        category_counts,
        annot=True,
        fmt='d',
        cmap='viridis',
        cbar=True,
        linewidths=0.5,
    )

    # Set labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Heatmap of Assigned Categories')

    if save_to:
        plt.savefig(save_to)

    plt.show()


# --------------------------------------------
#  Assign class labels to clusters
# --------------------------------------------


def summarize_sinkhorn_mapping(
    df,
    groupby,
    input_df,
    cluster_col_name='leiden',
    full_input_cluster_list=None,
    aggregate_by='coupling_strength',
):
    '''

    Inputs
    - df: pd.DataFrame, with columns
        ['idx1', 'source', 'idx2', 'target', 'coupling', 'normalized_strength', 'gp']
        where 'gp' is the embedding name
    - groupby: str, the column name for the index column = the group to summarize by
        For example, if groupby = 'target',
        then we group by the classes from the target distribtuion, and look for the
        classes from the source distribution with the highest values
    - input_df: pd.DataFrame, where the index is cell indices,
        and the columns are the input clusters labels
    - cluster_col_name: name of the column within input_df that has the labels
        we are interested in
    - full_input_cluster_list: list
        list of all of the input clusters
        this allows us to retrieve clusters that are not  mapped to any population
        across any GP
        Note this is a brute force approach, check the clusters were actually included
        in the OT computation

    Outputs
    - cluster_to_pred: pd.DataFrame, where columns are embedding names,
        values are assigned classes,

    '''
    cluster_to_pred = pd.DataFrame(
        index=sorted(list(input_df[cluster_col_name].unique())),
    )

    # if we group by target, use the source column as a "prediction"
    if groupby == 'target':
        col_name = 'source'

    # if we group by source, use the target column as a "prediction"
    elif groupby == 'source':
        col_name = 'target'

    for gp in df['gp'].unique():
        df1 = df[df['gp'] == gp]

        if aggregate_by == 'coupling_strength' and 'coupling' in df1.columns:
            topn = df1.loc[df1.groupby(groupby)['coupling'].idxmax()]
        elif aggregate_by == 'num_pairs':
            topn = (
                df1.groupby(['gp', 'source', 'target']).size().reset_index(name='count')
            )
            topn = topn.loc[topn.groupby(groupby)['count'].idxmax()]
        else:
            topn = df1.drop_duplicates(subset=[groupby, 'source', 'target'])

        cluster_to_pred = cluster_to_pred.join(
            topn[['source', 'target']]
            .set_index(groupby)
            .rename(columns={col_name: gp}),
            how='left',
        )

    if full_input_cluster_list:
        for cluster in full_input_cluster_list:
            if cluster not in cluster_to_pred.index:
                # Create a new row with NaN values
                new_row = pd.DataFrame(
                    [[np.nan] * len(cluster_to_pred.columns)],
                    columns=cluster_to_pred.columns,
                    index=[cluster],
                )
                # Append the new row to the dataframe
                cluster_to_pred = pd.concat([cluster_to_pred, new_row])

    return cluster_to_pred


def get_largest_assignment_mapping(
    leiden_to_pred: pd.DataFrame, unseen_threshold: int
) -> dict:
    """
    Returns a dictionary mapping the indices (row labels eg leiden clusters)
    to the value in the column where it has the largest assignment.
    ** If the number of non-zero category counts for a cluster is less than
    the unseen_threshold, the mapping for that cluster will be set to 'Unseen_{idx}'.

    ** If the most common value is NaN, the mapping will be set to the second
    most common value with the suffix 'like'.

    ** If there are multiple columns tied for the maximum value, the mapping will be
    set to the concatenated values of the tied columns.

    Parameters:
    - leiden_to_pred: pd.DataFrame, where columns are embedding names (eg GP),
        values are assigned classes (eg cell types), and the index are input classes
        (eg Leiden clusters)
    - unseen_threshold: int, the minimum number of non-zero category counts
        required for a cluster to be mapped to the column with the largest assignment.
        IE if one cluster is mapped to no classes for > unseen_threshold embeddings,
        it will be mapped to 'Unseen_{idx}'.

    Returns:
    - A dictionary where keys are the indices and values are the corresponding column
        where the index has the largest assignment or 'Unseen_{idx}'
    """

    # Count the number of times each class appears for each Leiden cluster
    df_leiden_col = leiden_to_pred.T

    mapping = {}

    for idx in df_leiden_col.columns:
        value_counts = df_leiden_col[idx].value_counts(dropna=False)

        # Count the number of non-zero, non-NaN entries in the column
        non_zero_counts = (df_leiden_col[idx].notna() & (df_leiden_col[idx] != 0)).sum()

        # If total number of non-zero,
        # non-NaN values is less than threshold, assign 'Unseen_{idx}'
        if non_zero_counts < unseen_threshold:
            mapping[idx] = f'Unseen_{idx}'
            continue

        # If NaN is the most common value, assign 'Maybe_2nd most common value'
        most_common_value, _ = value_counts.idxmax(), value_counts.max()
        if pd.isna(most_common_value):  # Check if NaN is the most common
            second_most_common_value = (
                value_counts.index[1] if len(value_counts) > 1 else None
            )
            mapping[idx] = (
                f'{second_most_common_value}_like'
                if second_most_common_value is not None
                else f'Unseen_{idx}'
            )
            continue

        # Get all indices tied for the maximum count value
        max_count = value_counts.max()
        chosen_gp = value_counts[value_counts == max_count].index.tolist()

        # If there are multiple columns tied for the maximum value, concatenate them
        if len(chosen_gp) > 1:
            chosen_gp = '_'.join(
                map(str, chosen_gp)
            )  # Convert each entry to string before concatenation

        if isinstance(chosen_gp, list) and len(chosen_gp) == 1:
            chosen_gp = chosen_gp[0]  # If it's a single item list, extract the item

        mapping[idx] = chosen_gp

    return mapping


# --------------------------------------------
#  Plotting
# --------------------------------------------


def blend_colors(color1, color2):
    """
    Blend two colors to get the midpoint color.
    Args:
        color1 (str): Color name or hex code for the first color.
        color2 (str): Color name or hex code for the second color.
    Returns:
        str: Hex code of the blended color.
    """
    rgb1 = to_rgb(color1)
    rgb2 = to_rgb(color2)
    blended_rgb = [(c1 + c2) / 2 for c1, c2 in zip(rgb1, rgb2)]
    return to_hex(blended_rgb)


def lighten_color(color, factor=0.5):
    """
    Lighten a color by blending it with white.
    Args:
        color (str): Color name or hex code for the color to lighten.
        factor (float): A value between 0 and 1,
        where 1 means no change and 0 means fully white.
    Returns:
        str: Hex code of the lightened color.
    """
    rgb = to_rgb(color)
    white = (1, 1, 1)
    lightened_rgb = [c * factor + (1 - factor) * w for c, w in zip(rgb, white)]
    return to_hex(lightened_rgb)


def encode_colors(data_obs, variable, colormap, custom_order=None):
    if variable is not None and variable in data_obs:
        if custom_order is not None:
            # Ensure the variable has the desired order
            data_obs[variable] = pd.Categorical(
                data_obs[variable], categories=custom_order, ordered=True
            )

        # Create custom mapping based on the Categorical codes
        labels = data_obs[variable].cat.codes  # This respects the custom order
        unique_labels = data_obs[variable].cat.categories  # Ordered categories
        cmap = ListedColormap(colormap(np.linspace(0.2, 1, len(unique_labels))))
        return labels, unique_labels, cmap, cmap.colors

    return None, None, None, None


def prepare_adata(adata, col, col_value, label_var, label_order):
    """Subset and reorder categories"""
    adata = adata[adata.obs[col] == col_value].copy()

    adata.obs[label_var] = pd.Categorical(
        adata.obs[label_var], categories=label_order, ordered=True
    )
    return adata


def plot_umap_scatter(ax, umap_coords, colors, cmap, label, alpha=0.5, size=20):
    """Scatter plot for UMAP coordinates."""
    if umap_coords is not None:
        ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=colors if colors is not None else 'gray',
            cmap=cmap if cmap else None,
            alpha=alpha,
            s=size,
            label=label,
        )


def plot_point_connections(ax, point_map, set_alpha):
    """Plot connections between points using a point map."""
    if point_map is None:
        return  # Skip plotting if no point map data is available

    for coords in point_map:
        start_i, end_i, _, alpha_i = coords
        ax.plot(
            start_i,
            end_i,
            color='k',
            alpha=alpha_i if set_alpha else 0.8,
            linestyle='--',
        )


def build_legend_elements(categories, cluster_colors, prefix):
    """Build legend elements for plotting."""
    if categories is None or cluster_colors is None:
        return []
    return [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor=cluster_colors[i],
            markersize=10,
            label=f'{prefix}: {cat}',
        )
        for i, cat in enumerate(categories)
    ]


def plot_umap_with_transport(
    ot_out,
    adata,
    col,
    ref,
    target,
    ref_label,
    ref_label_order,
    target_label,
    target_label_order,
    ref_colormap=plt.cm.Blues,
    target_colormap=plt.cm.Reds,
    fig_size=(12, 8),
    threshold=1e-7,
    set_alpha=True,
    save_path=None,
    use_umap_coordinates=True,
):
    # Prepare data
    ref_adata = prepare_adata(adata, col, ref, ref_label, ref_label_order)
    target_adata = prepare_adata(adata, col, target, target_label, target_label_order)

    # Map colors
    ref_colors, ref_categories, ref_cmap, ref_cluster_colors = encode_colors(
        ref_adata.obs, ref_label, ref_colormap, ref_label_order
    )
    (
        target_colors,
        target_categories,
        target_cmap,
        target_cluster_colors,
    ) = encode_colors(
        target_adata.obs, target_label, target_colormap, target_label_order
    )

    # Extract UMAP coordinates
    ref_umap = ref_adata.obsm['X_umap']
    target_umap = target_adata.obsm['X_umap']

    # Compute point map
    point_map, _ = compute_point_cloud_mapping(
        x=jnp.array(ref_umap),
        y=jnp.array(target_umap),
        matrix=ot_out.matrix,
        adata_ref=ref_adata,
        adata_target=target_adata,
        label_col=ref_label,
        threshold=threshold,
        closest_ref_idx=None,
        closest_query_idx=None,
        use_umap_coordinates=use_umap_coordinates,
    )

    # Plot
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_facecolor('white')
    plot_umap_scatter(ax, ref_umap, ref_colors, ref_cmap, 'Source')
    plot_umap_scatter(ax, target_umap, target_colors, target_cmap, 'Target')
    plot_point_connections(ax, point_map, set_alpha)

    # Legend
    legend_elements = build_legend_elements(
        ref_categories, ref_cluster_colors, 'Source'
    )
    legend_elements += build_legend_elements(
        target_categories, target_cluster_colors, 'Target'
    )
    if legend_elements:
        ax.legend(handles=legend_elements, loc='best', fontsize=12)

    # Final touches
    ax.set_xlabel('UMAP1', fontsize=14)
    ax.set_ylabel('UMAP2', fontsize=14)
    ax.set_title('Optimal Transport Mapping', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_umap_with_centroids(
    adata,
    col,
    ref,
    target,
    ref_label,
    ref_label_order,
    target_label,
    target_label_order,
    ref_colormap=plt.cm.Blues,
    target_colormap=plt.cm.Reds,
    fig_size=(12, 8),
    threshold=1e-7,
    set_alpha=True,
    save_path=None,
    point_map=None,
    closest_ref_idx=None,
    closest_query_idx=None,
    show_legend=True,
):
    # Prepare data
    ref_adata = prepare_adata(adata, col, ref, ref_label, ref_label_order)
    target_adata = prepare_adata(adata, col, target, target_label, target_label_order)

    # Map colors
    ref_colors, ref_categories, ref_cmap, ref_cluster_colors = encode_colors(
        ref_adata.obs, ref_label, ref_colormap, ref_label_order
    )
    (
        target_colors,
        target_categories,
        target_cmap,
        target_cluster_colors,
    ) = encode_colors(
        target_adata.obs, target_label, target_colormap, target_label_order
    )

    # Extract UMAP coordinates
    ref_umap = ref_adata.obsm['X_umap']
    target_umap = target_adata.obsm['X_umap']

    # Plot
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_facecolor('white')
    plot_umap_scatter(ax, ref_umap, ref_colors, ref_cmap, 'Source')
    plot_umap_scatter(ax, target_umap, target_colors, target_cmap, 'Target')
    plot_point_connections(ax, point_map, set_alpha)

    # Highlight selected centroids and add labels
    if closest_ref_idx is not None:
        ref_highlight_coords = ref_umap[closest_ref_idx, :]
        ref_labels = ref_adata.obs[ref_label].iloc[closest_ref_idx].values

        if ref_cluster_colors is not None:
            ref_highlight_color = np.array(
                [
                    ref_cluster_colors[ref_categories.tolist().index(label)]
                    for label in ref_labels
                ]
            )
        else:
            ref_highlight_color = 'gold'

        ax.scatter(
            ref_highlight_coords[:, 0],
            ref_highlight_coords[:, 1],
            c=ref_highlight_color,
            edgecolor='black',
            s=300,
            marker='o',
            label='Selected Reference Points',
        )

        # Add small text labels below the centroid points
        for i, (x, y) in enumerate(ref_highlight_coords):
            ax.text(
                x,
                y - 0.08,  # Shift the text a bit below the centroid
                ref_labels[i],
                color='black',
                fontsize=12,
                ha='center',
                va='top',
            )

    if closest_query_idx is not None:
        query_highlight_coords = target_umap[closest_query_idx, :]
        query_labels = target_adata.obs[target_label].iloc[closest_query_idx].values

        if target_cluster_colors is not None:
            query_highlight_color = np.array(
                [
                    target_cluster_colors[target_categories.tolist().index(label)]
                    for label in query_labels
                ]
            )

        else:
            query_highlight_color = 'lime'

        ax.scatter(
            query_highlight_coords[:, 0],
            query_highlight_coords[:, 1],
            c=query_highlight_color,
            edgecolor='black',
            s=300,
            marker='o',
            label='Selected Query Points',
        )

        # Add small text labels below the centroid points
        for i, (x, y) in enumerate(query_highlight_coords):
            ax.text(
                x,
                y - 0.08,  # Shift the text a bit below the centroid
                query_labels[i],
                color='black',
                fontsize=12,
                ha='center',
                va='top',
            )

    # Legend
    if show_legend:
        legend_elements = build_legend_elements(
            ref_categories, ref_cluster_colors, 'Source'
        )
        legend_elements += build_legend_elements(
            target_categories, target_cluster_colors, 'Target'
        )
        if legend_elements:
            ax.legend(handles=legend_elements, loc='best', fontsize=12)

    # Final touches
    ax.set_xlabel('UMAP1', fontsize=14)
    ax.set_ylabel('UMAP2', fontsize=14)
    ax.set_title('Optimal Transport Mapping', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_num_pairs_by_gp(
    df, col, value, sort_by=None, sort_by_row=None, highlight=None, **kwargs
):
    '''
    Make a heatmap where rows are source/target cell types,
    columns are GPs, and values are counts of pairs mapped to each population.

    sort_by:
        - 'sum': sort GP columns by total sum of mapped cells
        - 'max': sort GP columns by maximum cells mapped to any single population

    sort_by_row:
        - str: name of the row (mapped cell type) to sort GP columns by

    highlight:
        - list: GP labels to highlight in red on the x-axis

    **kwargs:
        arguments passed to sns.heatmap (cmap, annot, vmin, vmax, etc.)
    '''

    count_pairs = df.groupby(['source', 'target', 'gp']).size().reset_index(name='n')

    # Pivot the table to get 'source' as rows, 'gp' as columns, and 'n' as values
    if col == 'target':
        view_col = 'target'
        index_col = 'source'
    else:
        view_col = 'source'
        index_col = 'target'

    heatmap_data = count_pairs[count_pairs[view_col] == value].pivot_table(
        index=index_col, columns='gp', values='n', aggfunc='sum', fill_value=0
    )

    # Choose sorting method
    if sort_by == 'sum':
        sorted_columns = heatmap_data.sum().sort_values(ascending=False).index
    elif sort_by == 'max':
        sorted_columns = heatmap_data.max().sort_values(ascending=False).index
    elif sort_by_row:
        if sort_by_row in heatmap_data.index:
            sorted_columns = (
                heatmap_data.loc[sort_by_row].sort_values(ascending=False).index
            )
        else:
            raise ValueError(f"'{sort_by_row}' is not a valid row name.")
    else:
        sorted_columns = heatmap_data.columns  # Default order

    # Reorder the heatmap data
    heatmap_data = heatmap_data[sorted_columns]

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, cbar=True, linewidths=0.5, ax=ax, **kwargs)

    # Color specified GP labels in red
    if highlight is not None:
        for i, label in enumerate(heatmap_data.columns):
            if label in highlight:
                ax.get_xticklabels()[i].set_color('red')

    plt.xlabel('GP')
    plt.ylabel('Mapped cell type')
    plt.title(f'Number of {value} cells mapped to each population, by GP')

    plt.show()
