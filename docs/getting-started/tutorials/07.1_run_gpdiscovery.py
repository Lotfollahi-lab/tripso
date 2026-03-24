"""GPFinder Training for Novel Gene Program Discovery - Tripso Tutorial

This script demonstrates how to train the Tripso GPFinder discovery module
to infer data-driven gene programs beyond the curated set used in the Base model.
The GPFinder module incorporates the pretrained Base model while adding
an additional HVG (highly variable genes) gene program. We will see how to cluster
the attention weights of this module to enable discovery of new biological programs
in the next script.

The workflow involves:
1. Creating an expanded gene program database file that includes HVG
2. Configuring model architecture matching the Base model
3. Training the new HVG block with reconstruction loss

Inputs:
    - data/processed/input_h5ad/zeng_gp_genes.h5ad: Raw gene expression data
    - data/processed/input_dataset/: Tokenized cell data
    - gpdb_tf.csv: Original curated gene program database
    - output_base/: Path to directory containing pretrained Base model checkpoints
    - data/processed/zeng.h5ad: Processed dataset with HVG annotations

Outputs:
    - gpdb_with_hvg.csv: Extended gene program database including HVG
    - output_gpfinder/: Trained GPFinder model checkpoints and logs
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc

import tripso
from tripso.Train.training_flexi import run_training_from_select_gps

# =======================================================
# Setup directories and paths
# =======================================================

root_dir = 'path/to/your/folder/07_tutorial_zeng'
data_dir = 'path/to/your/folder/07_tutorial_zeng/data/processed/input_dataset'

output_dir = os.path.join(root_dir, 'output_base')
gpdb_path = os.path.join(root_dir, 'gpdb_tf.csv')

# =======================================================
# Prepare expanded gene program database with HVG
# =======================================================

# Load highly variable genes from processed dataset
zeng = sc.read_h5ad(os.path.join(root_dir, 'data/processed/zeng.h5ad'))
hvg = zeng[:, zeng.var['HVG_intersect3000']].var_names

# Load original curated gene program database
gpdb = pd.read_csv(gpdb_path)

# Collect all genes from curated GPs and HVG
all_genes = set()

for i in gpdb.columns:
    all_genes.update(gpdb[i].dropna().values)

all_genes.update(hvg)
all_genes = list(all_genes)

# Create new gene program database with HVG added as a separate program
gpdb_new = {}

for gp in gpdb.columns:
    # Pad each GP column to match HVG length with NaN values
    gpdb_new[gp] = list(gpdb[gp].dropna().values) + [
        np.nan for _ in range(len(hvg) - len(gpdb[gp].dropna().values))
    ]

# Add HVG as a new gene program column
gpdb_new['HVG'] = list(hvg)

# Save expanded database to disk
gpdb_new = pd.DataFrame(gpdb_new)
gpdb_new.to_csv(os.path.join(root_dir, 'gpdb_with_hvg.csv'), index=False)

# =======================================================
# Configure model architecture and training parameters
# =======================================================

# Basic training hyperparameters
tissue = 'zeng'
model_type = 'Base'
n_heads = 8
n_blocks = 2
weight_decay = 1e-4
mgm = 0.25  # Masked gene modeling probability
n_epochs = 20
batch_size = 128
lr_scheduler = 'ReduceLROnPlateau'
lr = 1e-4

# Gene encoder (foundation model) configuration
# This configuration matches the Base model architecture to ensure compatibility
config_dict = {
    'hidden_size': 512,  # Dimension of hidden representations
    'num_hidden_layers': 2,  # Number of transformer layers
    'num_attention_heads': 8,  # Number of attention heads
    'tokenization_input_size': 4096,  # Max sequence length of the Geneformer model used for tokenization
    'mlm_masking_prob': 0.25,  # Masking probability for MLM task
    'use_pos_emb': 'sin_cos',  # Positional embedding type
    'use_l2_norm': False,  # Whether to use L2 normalization
    'tokenization_vocab_size': 20275,  # Total vocabulary size for tokenization (from Geneformer)
    'torch_dtype': 'bf16',  # Use bfloat16 precision
    'use_flash': True,  # Enable flash attention
    'max_seq_len': len(all_genes),  # Maximum sequence length
}

# =======================================================
# Train GPFinder model
# =======================================================

# Define paths for original and extended gene program databases
GPDB_OLD = os.path.join(root_dir, 'gpdb_tf.csv')  # Original curated GPs
GPDB_NEW = os.path.join(root_dir, 'gpdb_with_hvg.csv')  # Extended with HVG

GPFINDER_DIR = os.path.join(root_dir, 'output_gpfinder')

# Fine-tune from Base model with expanded GP database
# The model will learn to represent the new HVG program alongside existing curated GPs
run_training_from_select_gps(
    # Data inputs
    adata_path=os.path.join(root_dir, 'data/processed/input_h5ad/zeng_gp_genes.h5ad'),
    dataset_path=data_dir,
    gpdb_path=GPDB_NEW,  # Use expanded database with HVG
    gpdb_old=GPDB_OLD,  # Original database for initialization
    # Model configuration
    output_dir=GPFINDER_DIR,
    tissue='zeng',
    model_type_old='Base',  # Base model to load from
    model_type='Global',  # Global architecture with reconstruction
    global_loss='reconstruction',  # Use reconstruction loss for fine-tuning
    reconstruction_loss='nb',  # Negative binomial loss for count data
    global_training='finetune',  # Necessary to update weights of HVG block
    path_to_base_model=os.path.join(root_dir, 'output_base'),
    # Gene program configuration
    gp_inputs_new=None,  # Use all GPs (including HVG)
    gp_inputs_old=list(gpdb.columns),  # Original curated GPs
    # Gene encoder configuration
    fm_encoder_pkg='from_scratch',  # Train gene encoder from scratch
    bert_config=config_dict,  # Use config matching Base model
    use_gene_embeddings='gf-12L-95M-i4096',  # Initialize with pre-extracted Geneformer embeddings
    # Training parameters
    sampler='weighted',  # Weighted sampling for balanced training
    sample_by='age_group',  # Sample by age group for balance (this is the covariate we will look into at the next step)
    seed=0,  # Random seed for reproducibility
    all_genes=all_genes,  # Complete gene list = effective vocabulary size
    lr=lr,  # Learning rate
    n_epochs=10,  # Number of training epochs
    batch_size=128,  # Batch size
    precision='bf16-mixed',  # Mixed precision training
    # Gene program module parameters
    gp_latent_size=256,  # Dimension of GP latent space
    n_heads=8,  # Number of attention heads
    calc_gp_loss=False,  # Don't calculate GP masking loss
)
