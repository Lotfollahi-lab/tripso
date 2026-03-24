"""Train Base and Global Models for Tripso Tutorial

This script trains both the Base and Global models on the tokenized zeng dataset.
The Base model learns gene program representations using masked gene modeling,
while the Global model learns a global cell-level representation
through a gene expression reconstruction task. Training is
performed sequentially, with the Global model initialized from the Base model,
but the gene program module blocks being frozen while the Global model is trained.

The `use_gene_embeddings` argument is set to 'gf-12L-95M-i4096' to initialize the gene embeddings
with the pre-extracted Geneformer embeddings from the GF-12L-95M-i4096 model.
You can replace this with the path to your favorite gene embedding file (.pt or .npy)
or set it to False to train the gene embeddings from scratch.

Inputs:
    - data/processed/input_dataset/: Tokenized cell data
    - gpdb_tf.csv: Gene program database
    - data/processed/zeng_genes.h5ad: zeng raw gene expression data (for Global model)

Outputs:
    - output_base/: Trained Base model checkpoints and logs
    - output_global/: Trained Global model checkpoints and logs
"""

import os

import pandas as pd
import scanpy as sc

import tripso

# Directory paths for loading/saving
root_dir = 'path/to/your/folder/07_tutorial_zeng'
data_dir = 'path/to/your/folder/07_tutorial_zeng/data/processed/input_dataset'

output_dir = os.path.join(root_dir, 'output_base')
gpdb_path = os.path.join(root_dir, 'gpdb_tf.csv')


# define list of genes to use
zeng = sc.read_h5ad(os.path.join(root_dir, 'data/processed/zeng.h5ad'))
hvg = zeng[:, zeng.var['HVG_intersect3000']].var_names

gpdb = pd.read_csv(gpdb_path)

all_genes = set()

for i in gpdb.columns:
    all_genes.update(gpdb[i].dropna().values)

all_genes.update(hvg)
all_genes = list(all_genes)

# define model training arguments
tissue = 'zeng'
model_type = 'Base'
n_heads = 8
n_blocks = 2
weight_decay = 1e-4
mgm = 0.25
n_epochs = 20
batch_size = 128
lr_scheduler = 'ReduceLROnPlateau'
lr = 1e-4


########################################################
# Base model
########################################################

# Configuration from the provided config file
config_dict = {
    'hidden_size': 512,
    'num_hidden_layers': 2,
    'num_attention_heads': 8,
    'tokenization_input_size': 4096,
    'mlm_masking_prob': 0.25,
    'use_pos_emb': 'sin_cos',
    'use_l2_norm': False,
    'tokenization_vocab_size': 20275,
    'torch_dtype': 'bf16',
    'use_flash': True,
    'max_seq_len': len(all_genes),
}

# train model
tripso.train(
    dataset_path=data_dir,
    gpdb_path=gpdb_path,
    output_dir=output_dir,
    batch_size=batch_size,
    mgm=mgm,
    tissue=tissue,
    model_type=model_type,
    n_heads=n_heads,
    n_epochs=n_epochs,
    # use_flash = True,
    n_blocks=n_blocks,
    weight_decay=weight_decay,
    lr_scheduler=lr_scheduler,
    sampler='weighted',
    sample_by='Sorting',
    precision='bf16-mixed',
    fm_encoder_pkg='from_scratch',
    bert_config=config_dict,
    lr=lr,
    use_l2_norm=False,
    seed=0,
    all_genes=all_genes,
    gp_latent_size=256,
    use_gene_embeddings='gf-12L-95M-i4096',
    use_flash=True,
    use_pos_emb='sin_cos',
    warmup=10,
    init_sparsity=0,
)


########################################################
# Global model
########################################################

model_type = 'Global'
global_loss = 'reconstruction'
reconstruction_loss = 'nb'
n_epochs = 5
batch_size = 64
global_attn_heads = 8

path_to_base_model = os.path.join(root_dir, 'output_base')

output_dir = os.path.join(root_dir, 'output_global')

# train model
tripso.train(
    dataset_path=data_dir,
    gpdb_path=gpdb_path,
    output_dir=output_dir,
    batch_size=batch_size,
    mgm=mgm,
    tissue=tissue,
    model_type=model_type,
    n_heads=n_heads,
    n_epochs=n_epochs,
    n_blocks=n_blocks,
    global_loss=global_loss,
    reconstruction_loss=reconstruction_loss,
    global_training='sequential',
    global_attn_heads=global_attn_heads,
    adata_path=os.path.join(root_dir, 'data/processed/input_h5ad/zeng_gp_genes.h5ad'),
    path_to_base_model=path_to_base_model,
    lr=1e-3,
    sampler='weighted',
    sample_by='Sorting',
    weight_decay=weight_decay,
    precision='bf16-mixed',
    fm_encoder_pkg='from_scratch',
    bert_config=config_dict,
    use_gene_embeddings='gf-12L-95M-i4096',
    seed=0,
    all_genes=all_genes,
    gp_latent_size=256,
    global_attn_dropout=0.2
    # resume_training = True
)
