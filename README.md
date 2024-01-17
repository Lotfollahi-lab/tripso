 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![python](https://img.shields.io/badge/Python-3.10-brightgreen)

<p align="center">
<img src=assets/  alt="Mo's Lab logo"/>
</p>

# gpleaner: learning representations of single cell gene program activity 

## 0. Introduction & Scope

Introducing **gpLearner** 


### Projects

Currently available:

- [Modules](gplearner/Models/) : base model for learning individual GP representations
-  

### Discussion Board

This repository is accompanied by a discussion board intended for active communication with and among the community.
Please feel free to ask your questions there, share valuable insights and give us feedback on our material.

### Disclaimer

Please note that the contents of this repository are still in the experimental early
stages and may be subject to significant changes, bugs, and limitations.
We are continuously working on improving the **lotfollibrary** repository and welcome any
feedback or contributions. Thank you for your understanding.

## 1. Usage

First, clone the repo and change to the project directory.

```shell
git clone https://github.com/amirvhd/lotfollibrary.git
```

The relevant use-cases and source codes are located in `lotfollibrary`.
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

The `lotfollibrary` project is structured like a python package, which has the advantage of
being able to **install** it and thus reuse modules or functions without worrying about
absolute filepaths.
An editable version of `lotfollibrary` is also installed over `pip`:

```shell
pip install -e .
```

The project contains some jupyter notebooks, which were converted to python files
due to better handling in the repository.
These files end with `_nb.py` and can be converted back to a `.ipynb` file with
`jupytext`:

```shell
jupytext --to ipynb --execute <your_file>_nb.py
```

The `--execute` flag triggers executing every cell during conversion.
Alternatively, you can run the `_nb.py` files like every other python script.

Example usage:
```
import gplearner
import os

# Directory paths for loading/saving 
root_dir="/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/scgpl_reproducibility/examples/synthetic"
data_dir=os.path.join(root_dir, "data/input_dataset")
output_dir=os.path.join(root_dir, "output_TEST")

# define model training arguments
tissue = "synth"
model_type = "Base"
n_heads = 8
mgm = 0.15
n_epochs = 20
batch_size = 128
gene_format = "ensembl"
gp_latent_size = 256

# load data and preprocess
gplearner.pp_and_tokenize(root_dir = root_dir,
                          vars_to_keep = ["cell_type", "condition", "n_counts"],
                          subsample_by = ["cell_type", "condition"],
                          n_cells_per_class = 20_000,
                          n_splits = 2,
                          name_tag = "synth",
                          )

# train model
gplearner.train(
    dataset_path = data_dir,
    gpdb_path = os.path.join(root_dir, 'gpdb_synth.csv'),
    output_dir = output_dir,
    batch_size = batch_size,
    mgm = mgm,
    tissue = tissue,
    model_type = model_type, 
    n_heads = n_heads,
    n_epochs = n_epochs,
    gene_format = gene_format,
    gp_latent_size = gp_latent_size
)

# downstream evaluation
gp_downstream = gplearner.gpEval(
    dataset_path = data_dir,
    gpdb_path = os.path.join(root_dir, 'gpdb_synth.csv'),
    output_dir = output_dir,
    tissue = tissue,
    model_type = model_type,
    n_heads = n_heads,
    gene_format = gene_format
)

gp_downstream.generate_embeddings() 

# Generate UMAP for visualization
gp_downstream.visualize(label_to_plot = ["cell_type", "condition"]) # ouput = UMAP

# Quantative metrics:
# scanpy ranked genes and clusterability
# (could add scIB style metrics here)
gp_downstream.feature_analysis(label_to_plot = ["cell_type", "condition"],
                               rank_genes=True, 
                               cluster_latent=True
                               )

# Using <GP> cls to classify output labels
gp_downstream.logistic_regression(data_to_model = 'cell', labels = ['cell_type', 'condition'])
gp_downstream.logistic_regression(data_to_model = 'cell', 
                                  labels = ['cell_type', 'condition'],
                                  gp_features = "concat"
                                  )

# classifying gene embeddings to GP
gp_downstream.logistic_regression(data_to_model = 'gene_singleGP')
gp_downstream.logistic_regression(data_to_model = 'gene_mutliGP')


# Not yet implemented:
# gplearner.evaluate.visualize_attention()
# gplearner.evaluate.analyze_attention()


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
