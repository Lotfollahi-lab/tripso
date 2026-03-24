"""Novel Gene Program Discovery via Attention-Based Clustering - Tripso Tutorial

This script extracts and clusters novel gene programs from the GPFinder module's attention
weights. By analyzing how the HVG (highly variable genes) module attends to different genes,
we can identify coherent gene programs that emerge from the data without prior annotation.

The workflow involves:
1. Extracting attention weights from the HVG gene program module
2. Filtering genes by attention sparsity
3. Computing gene-gene similarity from attention patterns
4. Clustering genes to discover novel gene programs
5. Exporting discovered programs as a new gene program database

Inputs:
    - data/processed/input_dataset/: Tokenized cell data
    - gpdb_with_hvg.csv: Extended gene program database with HVG
    - output_gpfinder/: Trained GPFinder model checkpoints

Outputs:
    - output_gpfinder/attention/: Attention weight matrices for train/val/test splits
        (h5ad anndata files with cells x genes attention scores)
    - output_gpfinder/gpdb_clusters_from_attention.csv: Discovered gene programs
        organized by cluster, ready for downstream analysis
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc

import tripso
from tripso.Discovery.clustering import cluster

# =======================================================
# Setup directories and paths
# =======================================================

root_dir = 'path/to/your/folder/07_tutorial_zeng'
data_dir = 'path/to/your/folder/07_tutorial_zeng/data/processed/input_dataset'

# =======================================================
# Extract attention weights from GPFinder model
# =======================================================

# Load gene program databases
GPDB_OLD = os.path.join(root_dir, 'gpdb_tf.csv')  # Original curated GPs
GPDB_NEW = os.path.join(root_dir, 'gpdb_with_hvg.csv')  # Extended with HVG
gpdb_new = pd.read_csv(GPDB_NEW)

GPFINDER_DIR = os.path.join(root_dir, 'output_gpfinder')

# Initialize evaluation object for GPFinder model
gp_downstream = tripso.gpEval(
    dataset_path=data_dir,
    gpdb_path=GPDB_NEW,  # Use extended database with HVG
    output_dir=GPFINDER_DIR,
    tissue='zeng',
    model_type='Global',  # GPFinder uses Global architecture
    batch_size=128,
)

# Extract attention weights from HVG module for all data splits
# Attention weights reveal which genes the model considers important together
for t in ['train', 'test', 'val']:
    gp_downstream.generate_attention_matrix(
        gp_for_forward='HVG',  # Use HVG module for forward pass
        gp_for_downstream='HVG',  # Extract HVG attention weights
        genes_to_keep=gpdb_new['HVG'].dropna().tolist(),  # Keep all HVG genes
        precision='16-mixed',  # Use mixed precision to match training
        split=t,  # Process each data split
    )


# =======================================================
# Discover novel gene programs via clustering
# =======================================================

# Load attention weights from test set for clustering analysis
# We use test set to ensure discovered programs are not overfit to training data
# Additionally, for the tutorial, we focus on the test set only
# Although in practice, one might combine all splits to increase numbers for rare populations
attn = sc.read_h5ad(os.path.join(GPFINDER_DIR, 'attention/HVG_attention_test_set.h5ad'))

# Remove CLS (classification) token used by model architecture
attn = attn[:, attn.var.index != 'cls']
print('Initial attention shape:', attn.shape)

# Convert to dense matrix for correlation computation
# Attention matrix: cells x genes (how much each cell attends to each gene)
X_dense = attn.X.todense()
n_cells = attn.n_obs

# =======================================================
# Filter genes by attention quality
# =======================================================

# Calculate attention statistics for each gene
gene_nonzero_frac = np.array((X_dense > 0).sum(axis=0)).ravel() / n_cells  # Sparsity

# Keep genes that are attended to in >10% of cells
# This filters out genes with sparse or uninformative attention patterns
keep_mask = gene_nonzero_frac > 0.1
genes_to_keep = np.array(attn.var.index)[keep_mask]

# Apply filtering
attn = attn[:, genes_to_keep]
X_dense = X_dense[:, keep_mask]
genes = np.array(attn.var.index)
print('Attention shape after filtering:', attn.shape)

# =======================================================
# Compute gene-gene similarity from attention patterns
# =======================================================

# Calculate correlation between genes based on their attention patterns across cells
# Genes with similar attention patterns likely belong to the same program
corr_matrix = np.corrcoef(np.asarray(X_dense), rowvar=False)  # genes x genes
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)  # Replace NaN with 0
np.fill_diagonal(corr_matrix, 1.0)  # Ensure diagonal is 1

# Transform correlation [-1, 1] to similarity [0, 1] for clustering
# This preserves the magnitude of correlation while ensuring non-negative values
similarity_corr = (corr_matrix + 1.0) / 2.0

# =======================================================
# Cluster genes into novel gene programs
# =======================================================

# Determine range of cluster numbers to test
max_k = max(2, len(genes) // 30)
num_cluster_candidates = list(range(2, max_k + 1))

# Perform clustering to identify novel gene programs
# The algorithm will automatically select optimal number of clusters
labels_corr = cluster(
    similarity_corr,  # Gene-gene similarity matrix
    num_cluster_candidates=num_cluster_candidates,  # Range of k values to test
    seed=0,  # For reproducibility
)

# =======================================================
# Export discovered gene programs
# =======================================================

# Build gene program database: each cluster represents a novel gene program
# Format matches the input gpdb structure for downstream compatibility
gpdb_cluster = {}
unique_clusters = np.unique(labels_corr)
for c in unique_clusters:
    gpdb_cluster[f'gp_{c}'] = pd.Series(genes[labels_corr == c])

# Save discovered gene programs to disk
cluster_df = pd.DataFrame(gpdb_cluster)
cluster_df.to_csv(
    os.path.join(GPFINDER_DIR, 'gpdb_clusters_from_attention.csv'),
    index=False,
)

print(f'\nDiscovered {len(unique_clusters)} novel gene programs')
print(
    f"Gene programs saved to: {os.path.join(GPFINDER_DIR, 'gpdb_clusters_from_attention.csv')}"
)
