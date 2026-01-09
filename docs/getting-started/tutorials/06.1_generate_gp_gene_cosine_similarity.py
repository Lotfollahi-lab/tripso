import gplearner
import os
import pandas as pd
import numpy as np
from datasets import load_from_disk
from gplearner.Utils.utils import dataset_pivot_longer
import shutil

# Directory paths for loading/saving 
root_dir = '/lustre/scratch126/cellgen/lotfollahi/mm58/gplearner_reproducibility/06_tutorial_MNC' 
output_dir = os.path.join(root_dir, "output_global")
target_dir = os.path.join(output_dir, 'gene_embeddings_analysis')
gpdb_path = os.path.join(root_dir, 'gpdb_tf.csv')

gpdb = pd.read_csv(gpdb_path)

def write_gene_embeddings(gp):
    gp_genes = gpdb[gp].dropna().tolist()
    
    data_dir = os.path.join(root_dir, 'data/processed/input_dataset')

    gp_downstream = gplearner.gpEval(
        dataset_path=data_dir,
        gpdb_path=gpdb_path,
        output_dir=output_dir,
        tissue='HSC',
        model_type='Global',
    )
    
    dc = ['22_pDC', '24_cDC1', '25_cDC2',]
    
    for split in ['train', 'val', 'test']:
        gp_downstream.generate_gene_embeddings(
            gp_for_forward=gp,
            gp_for_downstream = gp,
            split=split,
            obs_key='cell_type',
            obs_value=dc,
            genes_to_keep=gp_genes,
            precision = '16-mixed',
            return_gene_cosim='gene_to_gp',
            )
    
    # move saved embeddings to target directory
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    emb_file_name = f"{gp}_gene_embeddings_{'_'.join(dc)}"

    shutil.move(os.path.join(output_dir, emb_file_name),
                os.path.join(target_dir, f"{gp}_dendritic_cells"))

if __name__ == "__main__":    
    for gp in ['GP_IRF1', 'GP_IKZF1']: 
        write_gene_embeddings(gp)