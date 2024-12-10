import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scipy.stats import chi2_contingency
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


def make_contigency_table(mapping, axis='row', fig_size=(10, 6)):
    """
    Analyze a mapping DataFrame by assigning labels from source A to source B
    (row-wise or column-wise), creating a crosstabulation, and performing a
    chi-square test, while visualizing the crosstab as a heatmap.

    Parameters:
        mapping (pd.DataFrame): The DataFrame where indices represent source A
        and columns represent source B.
        axis (str): Analysis direction, either 'row' or 'column'. Default is 'row'.

    Returns:
        None
    """
    if axis not in ['row', 'column']:
        raise ValueError("Axis must be either 'row' or 'column'")

    # Determine the axis to work on
    if axis == 'row':
        max_labels = mapping.idxmax(
            axis=1
        )  # Get the label of source B with max value for each row
        crosstab = pd.crosstab(index=mapping.index, columns=max_labels)
    else:
        max_labels = mapping.idxmax(
            axis=0
        )  # Get the label of source A with max value for each column
        crosstab = pd.crosstab(index=max_labels, columns=mapping.columns)

    # Plot the crosstabulation table as a heatmap
    plt.figure(figsize=fig_size)
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
    plt.title('Crosstab Heatmap')
    plt.xlabel('Target')
    plt.ylabel('Source')
    plt.show()

    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(crosstab)

    print('\nChi-square test results:')
    print(f'Chi-square statistic: {chi2}')
    print(f'Degrees of freedom: {dof}')
    print(f'P-value: {p_value}')


def view_mapping(mapping, normalize_by=None, fig_size=(6, 4)):
    '''
    Visualize the mapping DataFrame as a heatmap.
    '''

    # Reshape and aggregate using pivot_table
    summary_df = mapping.stack().reset_index()  # Reshape to long format
    summary_df.columns = ['index', 'column', 'value']
    summary_df = summary_df.groupby(['index', 'column']).sum()
    summary_df = summary_df.unstack(fill_value=0)  # Aggregate and reshape back
    summary_df.columns = summary_df.columns.droplevel(
        0
    )  # Drop the multi-level column index

    # Optionally normalize
    if normalize_by == 'row':
        summary_df = summary_df.div(summary_df.sum(axis=1), axis=0)
        title = 'Aggregated transport plan \nNormalized by row'
    if normalize_by == 'column':
        summary_df = summary_df.div(summary_df.sum(axis=0), axis=1)
        title = 'Aggregated transport plan \nNormalized by column'

    # Create a heatmap
    plt.figure(figsize=fig_size)
    sns.heatmap(summary_df, annot=True, fmt='.1f', cmap='viridis', cbar=True)

    # Add labels and title
    plt.title(title)
    plt.xlabel('Target labels')
    plt.ylabel('Source labels')
    plt.show()


def compute_point_cloud_mapping(
    x: jnp.ndarray,
    y: jnp.ndarray,
    matrix: jnp.ndarray,
    threshold=1e-7,
):
    """Compute the lines representing the mapping between the 2 point clouds."""
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

    return result


def compute_centroid_mapping(adata, col, ref, target, num_clusters=10, threshold=1e-8):
    # Data preparation
    X = adata[adata.obs[col] == ref].X
    Y = adata[adata.obs[col] == target].X

    # Clustering: Find centroids
    num_clusters = num_clusters
    kmeans_ref = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    kmeans_query = KMeans(n_clusters=num_clusters, random_state=0).fit(Y)

    # Get cluster centroids
    centroids_ref = kmeans_ref.cluster_centers_
    centroids_query = kmeans_query.cluster_centers_

    # Find the actual points closest to centroids
    closest_ref_idx = [np.argmin(cdist(X, [centroid])) for centroid in centroids_ref]
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
        adata.obs.loc[adata.obs[col] == target, 'idx'].iloc[closest_query_idx].values
    )

    # Create a subset of the AnnData object containing these indices
    combined_indices = np.concatenate([source_idx, target_idx])
    adata_centroid = adata[adata.obs['idx'].isin(combined_indices)].copy()

    # Output the shape of the new AnnData object
    ot_out = compute_sinkhorn(adata_centroid, col, ref, target)

    print('Converged?', ot_out.converged)

    # Extract full UMAP coordinates
    ref_umap = adata_centroid[adata_centroid.obs[col] == ref].obsm['X_umap']
    target_umap = adata_centroid[adata_centroid.obs[col] == target].obsm['X_umap']

    ref_umap_jnp = jnp.array(ref_umap)
    target_umap_jnp = jnp.array(target_umap)

    # Apply the modified mapping for point cloud alignment
    point_map_centroid = compute_point_cloud_mapping(
        ref_umap_jnp, target_umap_jnp, ot_out.matrix, threshold=threshold
    )

    return point_map_centroid, closest_ref_idx, closest_query_idx


# --------------------------------------------
#  Plotting
# --------------------------------------------


# Helper to encode colors
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

    # Highlight selected centroids
    if closest_ref_idx is not None:
        ref_highlight_coords = ref_umap[closest_ref_idx, :]
        ref_labels = ref_adata.obs[ref_label][closest_ref_idx]

        if ref_cluster_colors is not None:
            ref_highlight_color = [
                ref_cluster_colors[ref_categories.tolist().index(label)]
                for label in ref_labels
            ]
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

    if closest_query_idx is not None:
        query_highlight_coords = target_umap[closest_query_idx, :]
        query_labels = target_adata.obs[target_label][closest_query_idx]

        if target_cluster_colors is not None:
            query_highlight_color = [
                target_cluster_colors[target_categories.tolist().index(label)]
                for label in query_labels
            ]
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
