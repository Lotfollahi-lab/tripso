# Automatic filtering of public databases for preparation into scGPL pipeline
# we are looking for genes which are well-represented
# in a tokenized dataset (eg Geneformer)

import os

import pandas as pd
from datasets import load_from_disk

from ..Utils.gp_curation_utils import (
    Ontology,
    filter_by_size,
    rm_overlapping_gp,
)
from ..Utils.utils import (
    count_genes_per_cell,
    intersection_heatmap,
    load_gmt,
    make_similarity_matrix,
)

############################################
# Main Function
############################################


def make_gpdb(
    dataset_path,
    output_path,
    gp_inputs,
    use_ontology,
    n_cells_to_count,
    threshold_value,
    overlap_threshold,
    max_gp_len,
    name_tag,
):
    """
    Main function for building gene program database

    :param dataset_path: path to input tokenized dataset
    :param output_path: path to output directory
    :param gp_inputs: path to gene program databases (prior knowledge)

    :param use_ontology: whether to use ontology to expand gene programs
        option to only use gene sets and ignore their relationship in tree

    :param n_cells_to_count: number of cells to use for counting genes
        this is quite slow so if dataset is large, use a subset

    :param threshold_value: threshold for number of genes
        which must be expressed in 50% of cells

    :param overlap_threshold: threshold for max gene overlap between gene programs

    :param max_gp_len: maximum number of genes per gene program

    :param name_tag: tag to add to output files
    """

    ####################
    # Dataset
    ####################

    # Load
    dataset = load_from_disk(dataset_path)
    subset = dataset.shuffle(seed=0).select(range(n_cells_to_count))

    # Build counter object representing number of genes per cell

    if os.path.exists(os.path.join(output_path, 'genes_per_cell.csv')):
        token_df = pd.read_csv(os.path.join(output_path, 'genes_per_cell.csv'))
    else:
        token_df = count_genes_per_cell(subset)
        token_df.to_csv(os.path.join(output_path, 'genes_per_cell.csv'), index=False)

    # get list of genes which are expressed in at least 50% of the cells
    genes50 = token_df[token_df['prop'] > 0.5]['gene'].tolist()
    print('Number of genes expressed in at least 0.5 of cells:', len(genes50))

    # list of genes which are expressed in at least 10% of the cells
    genes10 = token_df[token_df['prop'] > 0.1]['gene'].tolist()

    ####################
    # Gene Programs
    ####################

    # Load gene program databases
    if isinstance(gp_inputs, str):
        gp_inputs = [gp_inputs]

    db = []

    for f in gp_inputs:
        if not os.path.exists(f):
            raise ValueError(f'Gene program database {f} does not exist')
        else:
            if f.endswith('.gmt'):
                db_in = load_gmt(f)
                db.append(db_in)
            elif f.endswith('.csv'):
                db_in = pd.read_csv(f, index_col=0).T
                db.append(db_in)
            elif f.endswith('.txt'):
                db_in = pd.read_csv(f, sep='\t', index_col=0).T
                db.append(db_in)
            else:
                raise ValueError(f"Format {f.split('.')[-1]} not recognized")

    # Bind into single dataframe
    db = pd.concat(db, axis=1)

    ###########################################
    # Dataset-specific Gene Programs
    ###########################################

    if use_ontology:
        # Build ontology
        onto = Ontology(
            filename='/lustre/scratch126/cellgen/team292/mm58/'
            'geneformer_endometrium/pathways_db/go.obo'
        )
        onto.add_genes(db)

        # Define threshold and initial genes to keep dictionary
        gp_to_keep = {}
        id_to_keep = set()
        col_to_keep = set()

        # Iterate through the ontology nodes
        for node_id in onto.ont.keys():
            onto.move_up_and_filter(
                genes50, threshold_value, node_id, gp_to_keep, id_to_keep, col_to_keep
            )

        # the columns of the database to keep are stored in col_to_keep
        gpdb = db[list(col_to_keep)]

    else:
        # filter gene sets meeting minimum criteria
        # without using ontology structure
        gpdb = filter_by_size(
            db,
            genes50,
            genes10,
            threshold_value,
            max_gp_len=max_gp_len,
            threshold_rare=10,
        )

    # Now remove gene sets with high overlap
    # (keep the one with the highest average proportion of cells expressing GP genes)
    gpdb = rm_overlapping_gp(gpdb, token_df, threshold=overlap_threshold)

    # Save
    gpdb.to_csv(os.path.join(output_path, f'gpdb_{name_tag}.csv'), index=False)

    # Visualize
    intersection_heatmap(
        gpdb, save_to=os.path.join(output_path, f'gpdb_heatmap_{name_tag}.png')
    )

    # Export similarity matrix based on overlap
    make_similarity_matrix(
        gpdb, save_to=os.path.join(output_path, f'gp_similarity_matrix_{name_tag}.npy')
    )
