import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..Utils.utils import build_gp_input_matrix
from .gp_model import (
    gpTransformerBase,
    gpTransformerGlobal,
    gpWrapper,
)

####################################
# Baseline : averaging GP embeddings
####################################


class AverageNonZero(nn.Module):
    def __init__(self, cls_tag='cls'):
        super().__init__()
        self.cls_tag = cls_tag

    def forward(self, x, return_gene_embeddings=False, *args, **kwargs):
        if return_gene_embeddings:
            output = {
                'cls': torch.zeros((1, 1)),
                'gene_embeddings': x,
                'logits_lm': [],
                'gene_labels': [],
            }
            return output

        # extra argument only for compatibility with gpTransformerEncoder
        # also for compatibility: extract tensor if necessary
        if isinstance(x, dict):
            x = x['z']

        # Count the non-zero values along the last dimension
        non_zero_count = torch.sum(x != 0, dim=1, keepdim=True)

        # Calculate the sum of non-zero values along the last dimension
        non_zero_sum = torch.sum(x * (x != 0), dim=1)

        # Avoid division by zero by setting count to 1 where it's zero
        non_zero_count[non_zero_count == 0] = 1

        # Compute the mean of non-zero values
        x = non_zero_sum / non_zero_count.squeeze()

        # output
        output = {self.cls_tag: x, 'logits_lm': [], 'gene_labels': []}

        return output


class gpAverager(gpWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = nn.ModuleList(
            [AverageNonZero() for i in range(len(self.gp_inputs))]
        )

    def get_last_self_attn(self, gf_emb, input_dataset, gp_idx):
        # Extract embeddings for the gene program of interest
        emb_pad, tokens_pad, _, attn_mask = build_gp_input_matrix(
            gf_emb['gene_emb'],  # geneformer embeddings
            input_dataset['input_ids'],
            getattr(self, f'gp{gp_idx}_tokens'),
            gp_idx=gp_idx,
        )

        # Encode tokens
        tokens_pad = (
            tokens_pad.cpu()
            .apply_(
                lambda x: getattr(self, f'gp{gp_idx}_tokens_encoded')[x]
                if x in getattr(self, f'gp{gp_idx}_tokens_encoded').keys()
                else -100
            )
            .to(emb_pad.device)
        )

        # Get cell embedding: average non zero genes
        o = self.encoder[gp_idx](emb_pad)
        cell = o['cls']
        # Reshape to dimensions of gene tensor
        cell = cell.unsqueeze(1).expand_as(emb_pad)

        # get cosine similarity between gene and cell embedding
        # for each gene in the GP
        cosim = F.cosine_similarity(emb_pad, cell, dim=-1)

        # set to 0 for padding tokens
        cosim[tokens_pad == -100] = 0

        # For the padding tokens, attention will be 0
        # so we can randomly reassign gene tokens to help with ranking
        all_gp_tokens = set(getattr(self, f'gp{gp_idx}_tokens_encoded').values())

        holder = []

        for i in range(tokens_pad.shape[0]):
            # because we've not done any masking,
            # all the -100 tokens will be at the end
            x = tokens_pad[i, :]
            labeled_genes_idx = x != -100

            values_to_fill_in = all_gp_tokens - set(
                x[labeled_genes_idx].cpu().numpy().tolist()
            )
            new_labels = torch.tensor(list(values_to_fill_in)).to(x.device)

            new_padded = torch.concat([x[labeled_genes_idx], new_labels], dim=0)
            # bring back cls to first position
            new_padded = torch.cat([new_padded[-1].unsqueeze(0), new_padded[:-1]])

            holder.append(new_padded)

        tokens_pad = torch.stack(holder).long().to(cosim.device)

        # Reorder attention matrix so genes are in the same order in each cell
        # Create an index tensor to sort tokens_pad
        _, indices = torch.sort(tokens_pad, dim=1)

        # Apply sorting to the corresponding rows in x
        cosim = torch.gather(cosim, 1, indices)
        tokens_pad = torch.gather(tokens_pad, 1, indices)

        output = {
            'attn': cosim.detach().cpu().numpy(),
        }

        return output


class gfBaseline(gpTransformerBase):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.multi_gp_encoder = gpAverager(
            database=self.gpdb,
            do_ensembl_conversion=self.do_ensembl_conversion,
            gp_latent_size=self.gp_latent_size,
            n_blocks=self.n_blocks,
            num_heads=self.num_heads,
            mgm_mask_ratio=self.mgm_mask_ratio,
            gp_inputs=self.gp_inputs,
            use_flash=False,
            model_type='Mean',
            learn_new_gp=False,
            # MAY NEED TO UPDATE THIS
            mean_emb_dict=None,
            use_pos_emb=False,
            use_diffl=False,
            use_flex=False,
            fm_model_input_size=self.fm_model_input_size,
        )

    def get_last_self_attn(self, input_dataset, gp):
        warnings.warn(
            'Using model type : Mean'
            'Attention matrices are not available for this model type.'
            'Instead, we return the cosine similarity between GP embeddings'
            'and the mean GP embedding for that GP.'
            'but note that this is not a true attention matrix.'
        )
        gp_idx = self.gp_inputs.index(gp)
        # input is tokenized dataset
        emb_out = self.gf_wrapper(input_dataset)

        # Extract attention matrix for our GP of interest
        output = self.multi_gp_encoder.get_last_self_attn(
            emb_out, input_dataset, gp_idx=gp_idx
        )

        return output


class gfGlobal(gpTransformerGlobal):
    def __init__(
        self,
        database,
        do_ensembl_conversion,
        **kwargs,
    ):
        super().__init__(
            database=database, do_ensembl_conversion=do_ensembl_conversion, **kwargs
        )

        self.multi_gp_encoder = gpAverager(
            database=self.gpdb,
            do_ensembl_conversion=self.do_ensembl_conversion,
            gp_latent_size=self.gp_latent_size,
            n_blocks=self.n_blocks,
            mgm_mask_ratio=self.mgm_mask_ratio,
            gene_token_path=self.gene_token_path,
            gene_name_path=self.gene_name_path,
            gp_inputs=self.gp_inputs,
            use_flash=False,
            model_type='Mean',
            learn_new_gp=False,
            num_heads=1,
            use_pos_emb=False,
            fm_model_input_size=self.fm_model_input_size,
            use_diffl=False,
            use_flex=False,
        )

        self.cell_token_learner = AverageNonZero(cls_tag='cell_token')

    def get_cell_token_attention(self, input_dataset):
        warnings.warn(
            'Using model type : Mean'
            'Attention matrices are not available for this model type.'
            'Instead, we return the cosine similarity between GP embeddings'
            'and the mean GP embedding for that GP.'
            'but note that this is not a true attention matrix.'
        )

        output = super().forward(input_dataset, return_gp_cls=True)

        # get cosine similarity between cell and GP embedding
        # for each gene in the GP
        cosim = F.cosine_similarity(output['gp_cls'], output['cell_token'], dim=-1)

        # set to 0 for padding tokens
        cosim[output['gp_labels'] == -100] = 0

        # For the padding tokens, attention will be 0
        all_gp = set([i for i in range(len(self.gp_inputs))])

        holder = []

        gp_labels = output['gp_labels']

        for i in range(gp_labels.shape[0]):
            # because GP are ranked by number of genes per cell
            # all the -100 tokens will be at the end
            x = gp_labels[i, :]
            labeled_gp_idx = x != -100

            values_to_fill_in = all_gp - set(x[labeled_gp_idx].cpu().numpy().tolist())
            new_labels = torch.tensor(list(values_to_fill_in)).to(x.device)

            new_padded = torch.concat([x[labeled_gp_idx], new_labels], dim=0)
            # bring back cls to first position
            new_padded = torch.cat([new_padded[-1].unsqueeze(0), new_padded[:-1]])

            holder.append(new_padded)

        gp_labels = torch.stack(holder).long().to(gp_labels.device)

        # Reorder attention matrix so GP are in the same order in each cell
        # Create an index tensor to sort tokens_pad
        _, indices = torch.sort(gp_labels, dim=1)

        # Apply sorting to the corresponding rows in x
        cosim = torch.gather(cosim, 1, indices)
        gp_labels = torch.gather(gp_labels, 1, indices)

        output = {
            'attn': cosim.detach().cpu().numpy(),
        }

        return output
