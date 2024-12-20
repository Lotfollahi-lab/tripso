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
                f'\nProvided categories are {set(label_source)},'
                f'but categories are {set(order_source)}'
            )

    if order_target:
        if set(label_target) != set(order_target):
            raise ValueError(
                'Mismatch between target labels and categories.'
                f'\nProvided categories are {set(label_target)},'
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
    adata: ad.AnnData,
    label_col: str,
    threshold=1e-7,
):
    """
    Compute the lines representing the mapping between the 2 point clouds.
    From
    https://github.com/ott-jax/ott/blob/main/src/ott/tools/plot.py#L136

    Modified to extract cell labels from AnnData object
    """
    # Only plot the lines with a cost above the threshold.
    u, v = jnp.where(matrix > threshold)
    c = matrix[jnp.where(matrix > threshold)]
    xy = jnp.concatenate([x[u], y[v]], axis=-1)

    # Check if we want to adjust transparency.
    scale_alpha_by_coupling = True

    # We can only adjust transparency if max(c) != min(c).
    if scale_alpha_by_coupling:
        min_matrix, max_matrix = jnp.min(c), jnp.max(c)
        scale_alpha_by_coupling = max_matrix != min_matrix

    result = []
    mapping_output = []

    for i in range(xy.shape[0]):
        strength = jnp.max(jnp.array(matrix.shape)) * c[i]
        if scale_alpha_by_coupling:
            normalized_strength = (c[i] - min_matrix) / (max_matrix - min_matrix)
            alpha = 0.7 * float(normalized_strength)
        else:
            alpha = 0.7

        # Matplotlib's transparency is sensitive to numerical errors.
        alpha = np.clip(alpha, 0.0, 1.0)

        start, end = xy[i, [0, 2]], xy[i, [1, 3]]
        result.append((start, end, strength, alpha))

        # Use adata UMAP coordinates to get labels
        start_i = np.asarray(start)
        end_i = np.asarray(end)
        start_adata = adata[
            (adata.obsm['X_umap'][:, 0] == start_i[0])
            & (adata.obsm['X_umap'][:, 1] == end_i[0])
        ]
        end_adata = adata[
            (adata.obsm['X_umap'][:, 0] == start_i[1])
            & (adata.obsm['X_umap'][:, 1] == end_i[1])
        ]

        ct1 = start_adata.obs[label_col].values[0]
        ct2 = end_adata.obs[label_col].values[0]

        idx1 = start_adata.obs['idx'].values[0]
        idx2 = end_adata.obs['idx'].values[0]

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
        num_clusters = num_clusters
        kmeans_ref = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
        kmeans_query = KMeans(n_clusters=num_clusters, random_state=0).fit(Y)

        # Get cluster centroids
        centroids_ref = kmeans_ref.cluster_centers_
        centroids_query = kmeans_query.cluster_centers_

    elif cluster_algo == 'leiden':
        # Leiden clustering for reference condition
        sc.pp.neighbors(adata_ref, use_rep='X')  # use the existing data in .X
        sc.tl.leiden(adata_ref, resolution=resolution, key_added='leiden')

        # Leiden clustering for target condition
        sc.pp.neighbors(adata_target, use_rep='X')  # use the existing data in .X
        sc.tl.leiden(adata_target, resolution=resolution, key_added='leiden')

        # Get unique cluster identifiers
        clusters_ref = adata_ref.obs['leiden'].astype(int).unique()
        clusters_target = adata_target.obs['leiden'].astype(int).unique()

        # Calculate centroids for each cluster in the reference and target
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
        # Get unique cluster identifiers
        if cluster_col not in adata.obs.columns:
            raise ValueError(
                'Please provide a `cluster_col` argument'
                'with the values of precomputed clusters.'
            )
        clusters_ref = adata_ref.obs[cluster_col].unique()
        clusters_target = adata_target.obs[cluster_col].unique()

        # Calculate centroids for each cluster in the reference and target
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
        # Find the actual points closest to centroids
        closest_ref_idx = [
            np.argmin(cdist(X, [centroid])) for centroid in centroids_ref
        ]
        closest_query_idx = [
            np.argmin(cdist(Y, [centroid])) for centroid in centroids_query
        ]

        # Ensure indices are integers (avoid issues with .iloc)
        closest_ref_idx = np.array(closest_ref_idx, dtype=int)
        closest_query_idx = np.array(closest_query_idx, dtype=int)

        # Extract indices from the 'source' condition in the AnnData object
        source_idx = (
            adata.obs.loc[adata.obs[col] == ref, 'idx'].iloc[closest_ref_idx].values
        )
        target_idx = (
            adata.obs.loc[adata.obs[col] == target, 'idx']
            .iloc[closest_query_idx]
            .values
        )

        # Create a subset of the AnnData object containing these indices
        combined_indices = np.concatenate([source_idx, target_idx])
        adata_centroid = adata[adata.obs['idx'].isin(combined_indices)].copy()

    # ----------------------------------------------------------------------
    # Match syntax for centroid-based analysis
    # ----------------------------------------------------------------------

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

    # Extract UMAP coordinates
    ref_umap = adata_centroid[adata_centroid.obs[col] == ref].obsm['X_umap']
    target_umap = adata_centroid[adata_centroid.obs[col] == target].obsm['X_umap']

    ref_umap_jnp = jnp.array(ref_umap)
    target_umap_jnp = jnp.array(target_umap)

    # Apply the modified mapping for point cloud alignment
    point_map_centroid, mapping_df = compute_point_cloud_mapping(
        ref_umap_jnp,
        target_umap_jnp,
        ot_out.matrix,
        adata,
        label_col,
        threshold=threshold,
    )

    # check dtype of output
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
        point_map_centroid, closest_ref_idx, closest_query_idx, mapping_df


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

    # Step 1: Keep the row with the largest 'coupling' for each 'idx1'
    df_largest_coupling = df.loc[df.groupby(idx_col)['coupling'].idxmax()]

    # Step 2: Create a crosstabulation of 'source' and 'target'
    crosstab = pd.crosstab(df_largest_coupling['source'], df_largest_coupling['target'])

    # Step 3: Ensure all values from order_source and order_target
    # are included in the crosstab
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

    # Step 4: Reorganize the crosstab to maximize the diagonal
    # using the Hungarian algorithm
    if not use_label_order:
        cost_matrix = (
            -crosstab.values
        )  # We negate the matrix since we want to maximize the diagonal
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Reorder columns based on the result of the Hungarian algorithm
        ordered_targets = crosstab.columns[col_ind]

        crosstab = crosstab.loc[:, ordered_targets]

    # Step 5: Plot the crosstabulation table as a heatmap
    plt.figure(figsize=fig_size)
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Target')
    plt.ylabel('Source')
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to)

    plt.show()

    # Step 6: Perform chi-square test
    if labels_target:
        crosstab = crosstab.loc[:, crosstab.columns.isin(labels_target)]
    if labels_source:
        crosstab = crosstab.loc[crosstab.index.isin(labels_source), :]


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
    predefined_order: list,
    x_label='Leiden Clusters',
    y_label='Reference cell types',
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
    - predefined_order: list of assigned categories
        in the order you want them to appear on the y-axis.
    """

    # Transpose DataFrame to ensure Leiden clusters are columns and embeddings are rows
    transposed = leiden_to_pred.T

    # Count the number of times each class appears for each Leiden cluster
    category_counts = (
        transposed.apply(lambda col: col.value_counts()).fillna(0).astype(int)
    )

    # Reindex to ensure the order of categories on the y-axis
    category_counts = category_counts.reindex(predefined_order, axis=0).fillna(0)
    category_counts = category_counts.astype(int)

    # Optimize the order of the Leiden clusters to maximize the diagonal
    cost_matrix = (
        -category_counts.values
    )  # We negate to convert maximization to minimization problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reorder the columns of category_counts based on the optimal assignment
    reordered_category_counts = category_counts.iloc[:, col_ind]

    # Create the heatmap
    plt.figure(figsize=(10, 8))  # Adjust size as necessary
    sns.heatmap(
        reordered_category_counts,
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

    plt.show()


# --------------------------------------------
#  Assign class labels to clusters
# --------------------------------------------


def summarize_sinkhorn_mapping(df, idx_col, input_df, cluster_col_name='leiden'):
    '''

    Inputs
    - df: pd.DataFrame, with columns
        ['idx1', 'source', 'idx2', 'target', 'coupling', 'normalized_strength', 'gp']
        where 'gp' is the embedding name
    - idx_col: str, the column name for the index column = the group to summarize by
        For example, if idx_col = 'idx1', then we look for the target class
        with the highest value for each value of idx1
    - input_df: pd.DataFrame, where the index is cell indices,
        and the columns are the input clusters labels

    Outputs
    - cluster_to_pred: pd.DataFrame, where columns are embedding names,
        values are assigned classes,

    '''
    cluster_to_pred = pd.DataFrame(
        index=sorted(list(input_df[cluster_col_name].unique())),
    )

    for gp in df['gp'].unique():
        df1 = df[df['gp'] == gp]
        topn = df1.loc[df1.groupby(idx_col)['coupling'].idxmax()]
        topn['target'] = topn['target'].astype(int)

        cluster_to_pred = cluster_to_pred.join(
            topn[['source', 'target']]
            .set_index('target')
            .rename(columns={'source': gp}),
            how='left',
        )

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
    point_map = compute_point_cloud_mapping(
        jnp.array(ref_umap), jnp.array(target_umap), ot_out.matrix, threshold
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
