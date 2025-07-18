import os
from typing import Optional, List, Dict, Union
import pathlib  # For Path conversion in write_h5ad

# If torch import fails, ensure torch is installed in your environment
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from torch.utils.data import Dataset
from datasets import concatenate_datasets
import pickle

from gplearner.Trainers.trainer import gpBase
from gplearner.Evaluate.downstream import gpEval
from gplearner.Utils.utils import build_gp_input_matrix

class GeneAblation(gpBase):
    """
    Ablation trainer for masking individual gene embeddings in the model input.
    Inherits from gpBase. Analogous to gpAblation but for gene-level masking.
    """
    def __init__(self, compute_cosine: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.compute_cosine = compute_cosine
        self.save_raw_embeddings = not compute_cosine
        self.cosine_adata = None

    def build_perturbed_input_matrix(self, z, gene_labels, gene_pert_index=None):
        # z: (batch, n_genes, dim)
        # gene_labels: (batch, n_genes)
        # Mask the gene at gene_pert_index (across all cells in batch)
        attn_mask = torch.ones_like(gene_labels, dtype=z.dtype)
        if gene_pert_index is not None:
            attn_mask[:, gene_pert_index] = 0
            z[:, gene_pert_index] = 0
        return z, gene_labels, attn_mask

    def test_step(self, batch, batch_idx):
        # Get the relevant GP index and name
        if hasattr(self, 'gp') and self.gp is not None and hasattr(self.model, 'gp_inputs'):
            gp_idx = self.model.gp_inputs.index(self.gp)
            gp_name = self.gp
        else:
            gp_idx = 0  # default to first GP if not specified
            gp_name = self.model.gp_inputs[0]

        # Invert the gene_token_path dictionary for token ID to gene name mapping
        with open(self.model.multi_gp_encoder.gene_token_path, 'rb') as f:
            gene_token_dict = pickle.load(f)
        token_to_gene = {v: k for k, v in gene_token_dict.items()}

        # Get gene tokens for this GP
        gp_tokens = getattr(self.model.multi_gp_encoder, f'gp{gp_idx}_tokens')
        gp_tokens_lookup = getattr(self.model.multi_gp_encoder, f'gp{gp_idx}_tokens_lookup')

        # Get gene embeddings and input_ids from batch
        gene_emb_dict = self.model.gf_wrapper(batch, masking=False)
        gene_emb = gene_emb_dict['gene_emb']  # shape: (batch, seq_len, embed_dim)
        input_ids = batch['input_ids']

        # Build the input matrix for this GP
        emb_pad, tokens_pad, num_genes_per_cell, attn_mask = build_gp_input_matrix(
            gene_emb, input_ids, gp_tokens
        )
        tokens_pad = gp_tokens_lookup[tokens_pad].long()

        # Save the control (non-ablated) CLS embedding for this GP
        encoder_output_control = self.model.multi_gp_encoder.encoder[gp_idx](
            emb_pad,
            attn_mask=attn_mask,
            gene_labels=tokens_pad,
            masking=False,
            return_attention=False,
        )
        emb_dict = {}
        emb_dict['control'] = encoder_output_control['cls'].detach().cpu()

        if self.compute_cosine:
            cosine_dict = {}
            control_array = emb_dict['control'].numpy()

        # Map ablation positions to gene names using tokens_pad from the first batch row
        gene_token_ids = tokens_pad[0].cpu().numpy()  # shape: (n_genes,)
        # Get the list of GP genes and their tokens
        gp_genes = list(self.model.gpdb[gp_name])
        gp_gene_tokens = [gene_token_dict[g] for g in gp_genes if g in gene_token_dict]
        # For each GP gene, ablate if present, else use control
        for gene, gene_token in zip(gp_genes, gp_gene_tokens):
            # Find all positions in tokens_pad[0] that correspond to this gene's token
            positions = np.where(gene_token_ids == gene_token)[0]
            if len(positions) > 0:
                ablated_emb = emb_pad.clone()
                ablated_attn_mask = attn_mask.clone()
                for pos in positions:
                    ablated_emb[:, pos, :] = 0
                    ablated_attn_mask[:, pos + 1] = 1
                encoder_output = self.model.multi_gp_encoder.encoder[gp_idx](
                    ablated_emb,
                    attn_mask=ablated_attn_mask,
                    gene_labels=tokens_pad,
                    masking=False,
                    return_attention=False,
                )
                ablated_result = encoder_output['cls'].detach().cpu()
            else:
                # Gene not present, so ablation result is just the control
                ablated_result = encoder_output_control['cls'].detach().cpu()
            if self.compute_cosine:
                gene_array = ablated_result.numpy()
                cos_sim = 1 - np.diag(
                    np.dot(control_array, gene_array.T) /
                    (np.linalg.norm(control_array, axis=1) * np.linalg.norm(gene_array, axis=1) + 1e-8)
                )
                cosine_dict[gene] = cos_sim
            else:
                emb_dict[f'{gene}_perturb'] = ablated_result

        if self.save_raw_embeddings:
            for k, v in batch.items():
                if k != 'input_ids':
                    emb_dict[k] = v
            emb = Dataset.from_dict(emb_dict)
            if self.emb_dataset is None:
                self.emb_dataset = emb
            else:
                self.emb_dataset = concatenate_datasets([self.emb_dataset, emb])
        elif self.compute_cosine:
            meta_dict = {}
            for k, v in batch.items():
                if k != 'input_ids':
                    if isinstance(v, torch.Tensor):
                        meta_dict[k] = v.cpu().numpy()
                    else:
                        meta_dict[k] = v
            if len(cosine_dict) > 0 and len(next(iter(cosine_dict.values()))) > 0:
                cosine_df = pd.DataFrame(cosine_dict)
                if cosine_df.shape[0] == pd.DataFrame(meta_dict).shape[0]:
                    adata = sc.AnnData(
                        X=cosine_df.values,
                        obs=pd.DataFrame(meta_dict),
                        var=pd.DataFrame(index=pd.Index(list(cosine_dict.keys()))),
                    )
                    if self.cosine_adata is None:
                        self.cosine_adata = adata
                    else:
                        if hasattr(sc, 'concat'):
                            self.cosine_adata = sc.concat([self.cosine_adata, adata])
                        else:
                            self.cosine_adata = ad.concat([self.cosine_adata, adata])
                else:
                    print(f"[GeneAblation] Batch {batch_idx}: Skipping AnnData creation due to shape mismatch: cosine_df {cosine_df.shape}, meta_dict {pd.DataFrame(meta_dict).shape}")
            else:
                print(f"[GeneAblation] Batch {batch_idx}: No ablated genes found. Skipping AnnData creation.")
        return None

    def on_test_epoch_end(self):
        output_path = os.path.join(self.output_dir, 'with_gene_ablation')
        os.makedirs(output_path, exist_ok=True)
        output_name = os.path.join(output_path, f'{self.split_label}_set')
        if self.save_raw_embeddings and self.emb_dataset is not None:
            self.emb_dataset.save_to_disk(output_name)
            self.emb_dataset = None
        elif self.cosine_adata is not None:
            # Use pathlib.Path for write_h5ad filename
            self.cosine_adata.write_h5ad(pathlib.Path(output_name + '.h5ad'))
            self.cosine_adata = None
        return None

class geneAblationEval(gpEval):
    def __init__(self, main_ckpt_dir, compute_cosine=False, gp=None, *args, **kwargs):
        self.main_ckpt_dir = os.path.join(main_ckpt_dir, 'checkpoints/last.ckpt')
        self.compute_cosine = compute_cosine
        self.gp = gp
        super().__init__(*args, **kwargs)

    def _init_trainer(self, split_label=None, **kwargs):
        load_kwargs = dict(hparam_save='ignore_model', map_location='cpu')
        if self.gp is not None:
            load_kwargs['gp'] = self.gp
        gp_transformer = GeneAblation.load_from_checkpoint(
            self.main_ckpt_dir, **load_kwargs
        )

        # reset attributes overwritten by loading from checkpoint
        gp_transformer.save_emb = True
        gp_transformer.split_label = split_label
        gp_transformer.output_dir = self.output_dir
        gp_transformer.compute_cosine = self.compute_cosine
        gp_transformer.save_raw_embeddings = not self.compute_cosine

        # Extract model
        self.model = gp_transformer.model
        # If user specified gp, use that, else use model's default
        if self.gp is not None:
            self.gp = self.gp
        else:
            self.gp = getattr(gp_transformer, 'gp', None)

        # Extract pretrained encoder config
        self.fm_encoder_pkg = gp_transformer.model.fm_encoder_pkg

        if self.fm_encoder_pkg == 'geneformer':
            self.fm_encoder_name = gp_transformer.model.fm_encoder_name
            self.max_len = (
                gp_transformer.model.gf_wrapper.gf.config.max_position_embeddings
            )
        elif self.fm_encoder_pkg == 'from_scratch':
            self.fm_encoder_name = gp_transformer.model.fm_encoder_pkg
            # TO DO --> flexibly account for different model sizes
            self.max_len = 4096

        return gp_transformer 