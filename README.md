 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![python](https://img.shields.io/badge/Python-3.10-brightgreen)

# GPformer: learning representations of single cell gene program activity 

## 0. Introduction & Scope

We introduce **GPformer**, a self-supervised approach for learning gene program activity at single cell resolution.  


### Projects

Currently available:

- [Modules](gplearner/Models/) :
  - Base model for learning individual GP representations
  - Global model for learning cell representations based on gene expression reconstruction or supervised tasks 

### Discussion Board

This repository is accompanied by a discussion board intended for active communication with and among the community.
Please feel free to ask your questions there, share valuable insights and give us feedback on our material.

### Disclaimer

Please note that the contents of this repository are still in the experimental early
stages and may be subject to significant changes, bugs, and limitations.

We are continuously working on improving the repository and welcome any
feedback or contributions. Thank you for your understanding.

## 1. Usage

First, clone the repo and change to the project directory.

```shell
git clone https://github.com/{repository name}
```

The relevant use-cases and source codes are located in `library`.
Currently, we support **python >= 3.10**.
It is recommended to install the required dependencies in a separate environment, e.g.
via `conda`.
A simpler alternative is a virtual environment, which is created and activated with:

```shell
python -m venv .venv
source .venv/bin/activate
```

Dependencies are then installed via `pip`.

```shell
pip install -r requirements.txt
```

The current project is structured like a python package, which has the advantage of
being able to **install** it and thus reuse modules or functions without worrying about
absolute filepaths.
An editable version of the package is also installed over `pip`:

```shell
pip install -e .
```

Example usage:
```
import gplearner
import os
import pandas as pd
from gplearner.Evaluate.downstream import calculate_gp_attribution_scores


# Directory paths for loading/saving 
root_dir = 'path/to/directory'
data_dir = os.path.join(root_dir, 'data/input_dataset')

output_dir = os.path.join(root_dir, "output_base")
gpdb_tag = "progeny" # identifier for gene program database
gpdb_path = os.path.join(root_dir, f'gpdb_{gpdb_tag}.csv')

# define model training arguments
tissue = "lung"
model_type = "Base"
n_heads = 8
n_blocks = 2
weight_decay = 1e-4
mgm = 0.75
n_epochs = 15
batch_size = 256
gp_latent_size = 256
lr_scheduler = 'CosineLRwithWarmUp'

# load data and preprocess
gplearner.pp_and_tokenize(root_dir=root_dir,
                          adata_path = os.path.join(root_dir, 'data/lung.h5ad'),
                          vars_to_keep = ['celltype', 'lineage', 'disease'],
                          cov_to_encode = ['celltype', 'lineage', 'disease'],
                          batch_keys = 'dataset',
                          subsample_by = None,
                          name_tag=gpdb_tag,
                          #save_gp_genes_object = True
                          )


# train model
gplearner.train(
    dataset_path=data_dir,
    gpdb_path=gpdb_path,
    output_dir=output_dir,
    batch_size=batch_size,
    mgm=mgm,
    tissue=tissue,
    model_type=model_type,
    n_heads=n_heads,
    n_epochs=n_epochs,
    gp_latent_size=gp_latent_size,
    use_weighted_sampler = True,
    sample_by = 'celltype_id',
    use_flash = False,
    n_blocks = n_blocks,
    weight_decay = weight_decay,
    lr_scheduler = lr_scheduler
)

########################################################
# Step 2: learning global cell token
########################################################

# define model training arguments
model_type = "Global"
global_loss = 'reconstruction'
reconstruction_loss = 'nb'
n_epochs = 8
batch_size = 128
gp_latent_size = 256
global_attn_heads = 8

path_to_base_model = os.path.join(root_dir, 'output_base')

output_dir = os.path.join(root_dir, "output_global")


# train model
gplearner.train(
    dataset_path=data_dir,
    gpdb_path=gpdb_path,
    output_dir=output_dir,
    batch_size=batch_size,
    mgm=mgm,
    tissue=tissue,
    model_type=model_type,
    n_heads=n_heads,
    n_epochs=n_epochs,
    gp_latent_size=gp_latent_size,
    global_loss = global_loss,
    reconstruction_loss = reconstruction_loss,
    global_training = 'sequential',
    global_attn_heads = global_attn_heads,
    adata_path = os.path.join(root_dir, 'data/input_h5ad/lung_gp_genes.h5ad'),
    path_to_base_model = path_to_base_model,
    lr = 1e-3,
    use_weighted_sampler = True,
    sample_by = 'celltype_id',
    use_flash = False,
    n_blocks = n_blocks,
    weight_decay = weight_decay
)

########################################################
# Step 3: Visualize 
########################################################

# downstream evaluation
gp_downstream = gplearner.gpEval(
    dataset_path=data_dir,
    gpdb_path=gpdb_path,
    output_dir=output_dir,
    tissue=tissue,
    model_type=model_type,
    n_heads=n_heads,
    n_blocks=n_blocks,
    global_attn_heads = global_attn_heads,    
    global_loss = global_loss,
    reconstruction_loss = reconstruction_loss,
)

# Generate embeddings for train and test set
for s in ['train', 'val', 'test']:
    gp_downstream.generate_embeddings(split = s)

# Generate UMAP for visualization
gp_downstream.visualize(label_to_plot=["celltype", "lineage", 'Binary Stage'])
gp_downstream.visualize(gp_to_plot = 'cell_token',
                        label_to_plot=["celltype", 'Binary Stage'])


# Using <GP> cls to classify output labels
gpdb = pd.read_csv(gpdb_path)
gp_inputs = list(gpdb.columns)

for gp in gp_inputs:
    gp = gp.replace('/', '_')
    gp_downstream.evaluate_embeddings(
        y_label = 'celltype',
        folder_path = os.path.join(output_dir, 'embeddings'),
        emb_label = gp,
        output_dir = output_dir,
        use_weighted_sampler = True,
        sample_by = 'celltype',
        encode_covariate = True
        )
    
    gp_downstream.evaluate_embeddings(
        y_label = 'lineage',
        folder_path = os.path.join(output_dir, 'embeddings'),
        emb_label = gp,
        output_dir = output_dir,
        use_weighted_sampler = True,
        sample_by = 'celltype',
        encode_covariate = True
        )
    
# Using cell token
gp_downstream.evaluate_embeddings(
    y_label = 'celltype',
    emb_label = 'cell_token',
    folder_path = os.path.join(output_dir, 'embeddings'),
    output_dir = output_dir,
    use_weighted_sampler = True,
    sample_by = 'celltype',
    encode_covariate = True
    )

# Evaluate count reconstruction
gp_downstream.test_random_baseline(adata_path = os.path.join(root_dir, 'data/input_h5ad/lung_gp_genes.h5ad'))

# Calculate gene -> GP attributions

model_checkpoint = find_latest_file(output_dir, tissue, model_type)

calculate_gp_attribution_scores(
    gpdb_path = gpdb_path,
    dataset_path = data_dir,
    gp = gp,
    data_split = 'test',
    n_blocks = 2,
    num_heads = 8,
    gp_latent_size = 256,
    model_checkpoint = model_checkpoint,
    obs_key = 'lineage',
    obs_value = 'Epithelial',
    output_dir = output_dir,
    total_n_cells=2000,
    emb_dataset_path = os.path.join(output_dir, 'embeddings'),
    model_type='Global',
    global_loss = 'reconstruction',
)

# Calculate GP -> cell attributions

for dis in ['Control', 'Disease']:
    calculate_cell_token_attribution_scores(
        gpdb_path = gpdb_path,
        dataset_path = data_dir,
        emb_dataset_path = os.path.join(output_dir, 'embeddings'),
        data_split = 'test',
        n_blocks = 1,
        num_heads = 8,
        gp_latent_size = 256,
        model_checkpoint = model_checkpoint,
        obs_key = 'disease_status',
        obs_value = dis,
        output_dir = output_dir,
        global_loss = global_loss,
        save_plot = True,
        supervised_labels = supervised_labels_dict,
        # for EmbEvaluator
        emb_label = 'cell_token',
        task = 'classification',
    )


```

## 2. Contributing

New ideas and improvements are always welcome. Feel free to open an issue or contribute
over a pull request.
Our repository has a few automatic checks in place that ensure a compliance with PEP8 and static
typing.
It is recommended to use `pre-commit` as a utility to adhere to the GitHub actions hooks
beforehand.
First, install the package over pip and then set a hook:
```shell
pip install pre-commit
pre-commit install
```

To ensure code serialization and keeping the memory profile low, `.ipynb` are blacklisted
in this repository.
A notebook can be saved to the repo by converting it to a serializable format via
`jupytext`, preferably `py:percent`:

```shell
jupytext --to py:percent <notebook-to-convert>.ipynb
```

The result is a python file, which can be committed and later on be converted back to `.ipynb`.
A notebook-python file from jupytext shall carry the suffix `_nb.py`.


## Citation

If you use our repository or code in your research, please cite us:

```

```
