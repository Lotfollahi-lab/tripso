<div align="center">
  <img src="tripso_logo_cropped.png" alt="Tripso Logo" width="400"/>
</div>

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![python](https://img.shields.io/badge/Python-3.10-brightgreen)


Welcome to the Tripso github repo! This document will walk you through installing  and the main steps for model training and analysis.

## Tutorial Overview

The tutorials are organized in a recommended sequence:

**Data Preprocessing**
- [00_prepare_data.py](docs/getting-started/tutorials/00_prepare_data.py) - Prepare your single-cell dataset by filtering cells, normalizing counts, and selecting highly variable genes for model input

**Model Training**
- [01_tokenize.py](docs/getting-started/tutorials/01_tokenize.py) - Convert preprocessed gene expression data into tokenized format compatible with the  model
- [02_run_tripso.py](docs/getting-started/tutorials/02_run_tripso.py) - Train the  model to learn gene programs from your tokenized dataset

**Downstream Analysis & Evaluation**

Embedding Analysis
- [03_run_tripso_eval.py](docs/getting-started/tutorials/03_run_tripso_eval.py) - Extract and save GP embeddings
- [04_visualize_embeddings.ipynb](docs/getting-started/tutorials/04_visualize_embeddings.ipynb) - Visualize learned GP-specific embeddings using UMAP to assess model quality and biological interpretability

Gene Program Importance
- [05_calculate_gp_importance_scores.ipynb](docs/getting-started/tutorials/05_calculate_gp_importance_scores.ipynb) - Calculate GP importance scores via ablation analysis and perform differential GP analysis across cell types to identify programs that distinguish cell populations

Gene-to-GP Importance
- [06.1_generate_gp_gene_cosine_similarity.py](docs/getting-started/tutorials/06.1_generate_gp_gene_cosine_similarity.py) - Extract and save gene to GP cosine simialrity scores
- [06.2_visualize_gene_cosine_similarity.ipynb](docs/getting-started/tutorials/06.2_visualize_gene_cosine_similarity.ipynb) - Analyze gene-to-GP cosine similarity to identify which genes are most strongly associated with each gene program and how these associations differ between cell types

Gene program discovery
- [07.1_run_gpdiscovery.py](docs/getting-started/tutorials/07.1_run_gpdiscovery.py) - Training Tripso's GP discovery module
- [07.2_extract_novel_gp.py](docs/getting-started/tutorials/07.2_extract_novel_gp.py) - Extract the GP discovery module attention weights and derive data-driven gene programs
- [07.3_visualize_novel_gp.ipyb](docs/getting-started/tutorials/07.3_visualize_novel_gp.ipynb) - Visualize the inferred gene programs in the original transcriptomic data.


---------------------

## Installation

Follow these steps to install Tripso and its dependencies.

### Step 1: Create and activate your virtual environment


```bash
python -m venv /path/to/your/env
source /path/to/your/env/bin/activate

```

### Step 2: Install PyTorch

Install PyTorch with the appropriate CUDA version for your system (see the [PyTorch installation page](https://pytorch.org/get-started/locally/) )

Example for CUDA 12.8 on linux:
```bash
pip install torch torchvision
```


### Step 3: Set up Geneformer

Tripso reuses the vocabulary and tokenizer from Geneformer. Follow the installation instructions on the [Geneformer HuggingFace page](https://huggingface.co/ctheodoris/Geneformer).

### Step 4: Clone and Install

Clone the  repository:

```bash
git clone git@github.com:Lotfollahi-lab/tripso.git
```

**Note:** If you don't have access to the Lotfollahi Lab GitHub organization, please message me [mm58@sanger.ac.uk](mailto:mm58@sanger.ac.uk)

Install  and its dependencies:

```bash
cd tripso
pip install -r requirements.txt
pip install -e .
```

### Step 6: Verify Installation

Test that  is correctly installed:

```bash
python -c "import tripso; print(' successfully installed!')"
```

---



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


## Citation

If you use our repository or code in your research, please cite us:

```

```
