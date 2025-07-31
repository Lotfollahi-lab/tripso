import os
from typing import Optional, List, Dict, Union
import pathlib
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
from gplearner.Metrics.metrics import wasserstein

class GeneAblationEMD(gpBase):
    """
    Ablation trainer for masking individual gene embeddings in the model input.
    Computes EMD (Earth Mover's Distance) between control and ablated embeddings.
    Inherits from gpBase.
    """
    def __init__(self, compute_emd: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.compute_emd = compute_emd
        self.save_raw_embeddings = not compute_emd
        self.emd_adata = None

    def build_perturbed_input_matrix(self, z, gene_labels, gene_pert_index=None):
        attn_mask = torch.ones_like(gene_labels, dtype=z.dtype)
        if gene_pert_index is not None:
            attn_mask[:, gene_pert_index] = 0
            z[:, gene_pert_index] = 0
        return z, gene_labels, attn_mask

    def test_step(self, batch, batch_idx):
        if hasattr(self, 'gp') and self.gp is not None and hasattr(self.model, 'gp_inputs'):
            gp_idx = self.model.gp_inputs.index(self.gp)
            gp_name = self.gp
        else:
            gp_idx = 0
            gp_name = self.model.gp_inputs[0]

        with open(self.model.multi_gp_encoder.gene_token_path, 'rb') as f:
            gene_token_dict = pickle.load(f)
        token_to_gene = {v: k for k, v in gene_token_dict.items()}

        gp_tokens = getattr(self.model.multi_gp_encoder, f'gp{gp_idx}_tokens')
        gp_tokens_lookup = getattr(self.model.multi_gp_encoder, f'gp{gp_idx}_tokens_lookup')

        gene_emb_dict = self.model.gf_wrapper(batch, masking=False)
        gene_emb = gene_emb_dict['gene_emb']
        input_ids = batch['input_ids']

        emb_pad, tokens_pad, num_genes_per_cell, attn_mask = build_gp_input_matrix(
            gene_emb, input_ids, gp_tokens
        )
        tokens_pad = gp_tokens_lookup[tokens_pad].long()

        encoder_output_control = self.model.multi_gp_encoder.encoder[gp_idx](
            emb_pad,
            attn_mask=attn_mask,
            gene_labels=tokens_pad,
            masking=False,
            return_attention=False,
        )
        emb_dict = {}
        emb_dict['control'] = encoder_output_control['cls'].detach().cpu()

        if self.compute_emd:
            emd_dict = {}
            control_array = emb_dict['control'].numpy()

        gene_token_ids = tokens_pad[0].cpu().numpy()
        gp_genes = list(self.model.gpdb[gp_name])
        gp_gene_tokens = [gene_token_dict[g] for g in gp_genes if g in gene_token_dict]
        for gene, gene_token in zip(gp_genes, gp_gene_tokens):
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
                ablated_result = encoder_output_control['cls'].detach().cpu()
            if self.compute_emd:
                gene_array = ablated_result.numpy()
                # Compute 1-Wasserstein (EMD) between control and ablated for each cell
                emd_scores = np.array([
                    wasserstein(torch.tensor(control_array[i]), torch.tensor(gene_array[i]), method='exact', power=1)
                    for i in range(control_array.shape[0])
                ])
                emd_dict[gene] = emd_scores
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
        elif self.compute_emd:
            meta_dict = {}
            for k, v in batch.items():
                if k != 'input_ids':
                    if isinstance(v, torch.Tensor):
                        meta_dict[k] = v.cpu().numpy()
                    else:
                        meta_dict[k] = v
            if len(emd_dict) > 0 and len(next(iter(emd_dict.values()))) > 0:
                emd_df = pd.DataFrame(emd_dict)
                if emd_df.shape[0] == pd.DataFrame(meta_dict).shape[0]:
                    adata = sc.AnnData(
                        X=emd_df.values,
                        obs=pd.DataFrame(meta_dict),
                        var=pd.DataFrame(index=pd.Index(list(emd_dict.keys()))),
                    )
                    if self.emd_adata is None:
                        self.emd_adata = adata
                    else:
                        if hasattr(sc, 'concat'):
                            self.emd_adata = sc.concat([self.emd_adata, adata])
                        else:
                            self.emd_adata = ad.concat([self.emd_adata, adata])
                else:
                    print(f"[GeneAblationEMD] Batch {batch_idx}: Skipping AnnData creation due to shape mismatch: emd_df {emd_df.shape}, meta_dict {pd.DataFrame(meta_dict).shape}")
            else:
                print(f"[GeneAblationEMD] Batch {batch_idx}: No ablated genes found. Skipping AnnData creation.")
        return None

    def on_test_epoch_end(self):
        output_path = os.path.join(self.output_dir, 'with_gene_ablation_emd')
        os.makedirs(output_path, exist_ok=True)
        output_name = os.path.join(output_path, f'{self.split_label}_set')
        if self.save_raw_embeddings and self.emb_dataset is not None:
            self.emb_dataset.save_to_disk(output_name)
            self.emb_dataset = None
        elif self.emd_adata is not None:
            self.emd_adata.write_h5ad(pathlib.Path(output_name + '.h5ad'))
            self.emd_adata = None
        return None

class geneAblationEMDEval(gpEval):
    def __init__(self, main_ckpt_dir, compute_emd=True, gp=None, *args, **kwargs):
        self.main_ckpt_dir = os.path.join(main_ckpt_dir, 'checkpoints/last.ckpt')
        self.compute_emd = compute_emd
        self.gp = gp
        super().__init__(*args, **kwargs)

    def _init_trainer(self, split_label=None, **kwargs):
        load_kwargs = dict(hparam_save='ignore_model', map_location='cpu')
        if self.gp is not None:
            load_kwargs['gp'] = self.gp
        gp_transformer = GeneAblationEMD.load_from_checkpoint(
            self.main_ckpt_dir, **load_kwargs
        )

        gp_transformer.save_emb = True
        gp_transformer.split_label = split_label
        gp_transformer.output_dir = self.output_dir
        gp_transformer.compute_emd = self.compute_emd
        gp_transformer.save_raw_embeddings = not self.compute_emd

        self.model = gp_transformer.model
        if self.gp is not None:
            self.gp = self.gp
        else:
            self.gp = getattr(gp_transformer, 'gp', None)

        self.fm_encoder_pkg = gp_transformer.model.fm_encoder_pkg

        if self.fm_encoder_pkg == 'geneformer':
            self.fm_encoder_name = gp_transformer.model.fm_encoder_name
            self.max_len = (
                gp_transformer.model.gf_wrapper.gf.config.max_position_embeddings
            )
        elif self.fm_encoder_pkg == 'from_scratch':
            self.fm_encoder_name = gp_transformer.model.fm_encoder_pkg
            self.max_len = 4096

        return gp_transformer 