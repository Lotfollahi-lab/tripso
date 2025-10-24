#!/usr/bin/env python3
"""
Example script demonstrating the new compute_delta_nb_loss functionality in gpAblation.

This script shows how to use the gpAblationEval class with the new compute_delta_nb_loss parameter
to compute the change in count reconstruction loss as a result of GP perturbation.
"""

import os
from gplearner.Evaluate.downstream import gpAblationEval

def run_ablation_with_delta_nb_loss():
    """
    Example of running GP ablation with delta NB loss computation.
    """
    
    # Example paths - adjust these to your actual data
    dataset_path = "/path/to/your/input_dataset"
    gpdb_path = "/path/to/your/gpdb.csv"
    output_dir = "/path/to/your/output"
    main_ckpt_dir = "/path/to/your/trained/model"
    adata_path = "/path/to/your/gene_expression.h5ad"  # NEW: Gene expression data for count reconstruction
    
    # Initialize gpAblationEval with compute_delta_nb_loss=True
    gp_downstream = gpAblationEval(
        dataset_path=dataset_path,
        gpdb_path=gpdb_path,
        output_dir=output_dir,
        model_type='Global',
        tissue='your_tissue',
        main_ckpt_dir=main_ckpt_dir,
        compute_delta_nb_loss=True,  # New parameter for delta NB loss computation
        adata_path=adata_path  # NEW: Pass gene expression data for count reconstruction
    )
    
    # Generate embeddings with delta NB loss computation
    gp_downstream.generate_embeddings(split='test', precision='16-mixed')
    
    print("Delta NB loss ablation completed!")
    print(f"Results saved to: {os.path.join(output_dir, 'with_gp_ablation')}")
    print("Output file: {split_label}_set_delta_nb_loss.h5ad")

def run_ablation_with_cosine_similarity():
    """
    Example of running GP ablation with cosine similarity computation (original functionality).
    """
    
    # Example paths - adjust these to your actual data
    dataset_path = "/path/to/your/input_dataset"
    gpdb_path = "/path/to/your/gpdb.csv"
    output_dir = "/path/to/your/output"
    main_ckpt_dir = "/path/to/your/trained/model"
    
    # Initialize gpAblationEval with compute_cosine=True (original functionality)
    gp_downstream = gpAblationEval(
        dataset_path=dataset_path,
        gpdb_path=gpdb_path,
        output_dir=output_dir,
        model_type='Global',
        tissue='your_tissue',
        main_ckpt_dir=main_ckpt_dir,
        compute_cosine=True  # Original parameter for cosine similarity computation
    )
    
    # Generate embeddings with cosine similarity computation
    gp_downstream.generate_embeddings(split='test', precision='16-mixed')
    
    print("Cosine similarity ablation completed!")
    print(f"Results saved to: {os.path.join(output_dir, 'with_gp_ablation')}")
    print("Output file: {split_label}_set.h5ad")

def run_ablation_with_raw_embeddings():
    """
    Example of running GP ablation with raw embeddings (original functionality).
    """
    
    # Example paths - adjust these to your actual data
    dataset_path = "/path/to/your/input_dataset"
    gpdb_path = "/path/to/your/gpdb.csv"
    output_dir = "/path/to/your/output"
    main_ckpt_dir = "/path/to/your/trained/model"
    
    # Initialize gpAblationEval without compute_cosine or compute_delta_nb_loss
    # This will save raw embeddings
    gp_downstream = gpAblationEval(
        dataset_path=dataset_path,
        gpdb_path=gpdb_path,
        output_dir=output_dir,
        model_type='Global',
        tissue='your_tissue',
        main_ckpt_dir=main_ckpt_dir
        # No compute_cosine or compute_delta_nb_loss parameters
    )
    
    # Generate embeddings with raw embeddings
    gp_downstream.generate_embeddings(split='test', precision='16-mixed')
    
    print("Raw embeddings ablation completed!")
    print(f"Results saved to: {os.path.join(output_dir, 'with_gp_ablation')}")
    print("Output: HuggingFace dataset with control and perturbed embeddings")

if __name__ == "__main__":
    print("GP Ablation Examples")
    print("===================")
    print()
    print("This script demonstrates three different modes of GP ablation:")
    print("1. Delta NB Loss computation (NEW)")
    print("2. Cosine similarity computation (original)")
    print("3. Raw embeddings (original)")
    print()
    print("IMPORTANT: For delta NB loss computation, you MUST provide:")
    print("- adata_path: Path to gene expression data (.h5ad file)")
    print("- The gene expression data must match the tokenized dataset exactly")
    print("- This is required for count reconstruction loss computation")
    print()
    print("To run any of these examples, uncomment the corresponding function call below:")
    print()
    print("# run_ablation_with_delta_nb_loss()")
    print("# run_ablation_with_cosine_similarity()")
    print("# run_ablation_with_raw_embeddings()")
