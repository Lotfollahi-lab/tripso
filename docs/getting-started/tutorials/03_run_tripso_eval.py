"""Model Evaluation and Gene Program Importance Analysis for Tripso Tutorial

This script performs downstream evaluation of the trained Global model. It generates
cell and gene program embeddings, and calculates gene program importance scores.
In the following notebooks, we will see examples of how to interpret
and analyze these outputs.

Inputs:
    - data/processed/input_dataset/: Tokenized cell data
    - gpdb_tf.csv: Gene program database
    - output_global/: Trained Global model

Outputs:
    - output_global/embeddings/: Cell and GP embeddings for train/val/test splits
        (huggingface dataset class, embeddings can be extracted and imported to anndata)
    - output_global/ablation/: Gene program importance scores and cosine similarities
        (h5ad anndata files)

"""

import os

import tripso
from tripso.Evaluate.downstream import gpAblationEval

# Directory paths for loading/saving
root_dir = '/path/to/your/folder/07_tutorial_zeng'
data_dir = '/path/to/your/folder/07_tutorial_zeng/data/processed/input_dataset'

gpdb_path = os.path.join(root_dir, 'gpdb_tf.csv')

output_dir = os.path.join(root_dir, 'output_global')
model_type = 'Global'

########################################################
# Downstream evaluation
########################################################


gp_downstream = tripso.gpEval(
    dataset_path=data_dir,
    gpdb_path=gpdb_path,
    output_dir=output_dir,
    tissue='zeng',
    model_type=model_type,
    seed=0,
)


for t in ['train', 'val', 'test']:
    gp_downstream.generate_embeddings(split=t, precision='16-mixed')


########################################################
# generate GP importance scores
#######################################################


# Directory paths for loading/saving
output_dir = os.path.join(root_dir, 'output_global/ablation')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# downstream evaluation
gp_downstream = gpAblationEval(
    dataset_path=data_dir,
    gpdb_path=gpdb_path,
    output_dir=output_dir,
    model_type='Global',
    tissue='zeng',
    main_ckpt_dir=os.path.join(root_dir, 'output_global'),
    seed=0,
    compute_cosine=True,
)

# Generate GP importance scores
for t in ['train', 'val', 'test']:
    gp_downstream.generate_embeddings(split=t, precision='16-mixed')
