####################################
# Load packages
####################################

import os
import pickle
from typing import Dict, Optional

# imports
import numpy as np
import torch
import torch.nn as nn
from geneformer import ENSEMBL_DICTIONARY_FILE, TOKEN_DICTIONARY_FILE
from peft import PeftConfig, get_peft_model
from transformers import BertConfig, BertForMaskedLM

from ..Modules.modules import (
    Mlp,
    PromptEncoder,
    gpTransformerEncoder,
    gpTransformerEncoderWithPrompt,
)
from ..Utils.geneformer_utils import EmbExtractor, get_gf_repo
from ..Utils.utils import (
    bin_gene_expression,
    build_gp_input_matrix,
    get_gp_tokens,
)

####################################
# Geneformer
####################################


class OneHotWrapper(nn.Module):
    def __init__(self, gene_names):
        super().__init__()
        self.ref_gene_names = gene_names
        self.gene_to_index = {gene: idx for idx, gene in enumerate(gene_names)}

    def forward(self, input_dataset, inference):
        b, s = input_dataset['input_ids'].shape
        e = len(self.ref_gene_names)

        # Convert gene names in input_ids to indices based on all_genes
        gene_indices = input_dataset['input_ids']

        # remove padding tokens
        gene_indices = gene_indices[:, : len(self.ref_gene_names)]

        b, s = gene_indices.shape
        e = len(self.ref_gene_names)

        # Initialize embedding tensor with zeros
        emb = torch.zeros((b, s, e), dtype=torch.float32).to(gene_indices.device)

        # Use advanced indexing to set the appropriate positions
        batch_indices = torch.arange(b).view(b, 1).expand(b, s)
        sequence_indices = torch.arange(s).view(1, s).expand(b, s)
        emb[batch_indices, sequence_indices, gene_indices] = input_dataset['norm_exp']

        return emb


class gfWrapper(nn.Module):
    def __init__(
        self,
        geneformer_model,
        fm_layer_to_quant,
        peft_config_path,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):
        super().__init__()

        # Initialize geneformer model for getting geneformer embeddings
        model = BertForMaskedLM.from_pretrained(
            geneformer_model,
            output_attentions=False,
            output_hidden_states=True,
            # attn_implementation = 'sdpa',
            # load_in_8bit=True,
        )

        if peft_config_path:
            # Load the PEFT configuration from the checkpoint
            peft_config = PeftConfig.from_pretrained(peft_config_path)
            self.gf = get_peft_model(model, peft_config)
        else:
            self.gf = model

        # Freeze weights for geneformer model
        for name, param in self.gf.named_parameters():
            param.requires_grad = False

        self.gf_emb_extractor = EmbExtractor(
            emb_layer=fm_layer_to_quant,
            token_dictionary_file=token_dictionary_file,
        )

    def forward(self, input_dataset, masking):
        # input is tokenized dataset

        emb_out = self.gf_emb_extractor.extract_embs(
            model=self.gf,
            input_data=input_dataset,
            # turn off Geneformer dropout
            inference=True,
        )

        gene_output = {}
        gene_output['gene_emb'] = emb_out

        return gene_output


class GeneWrapper(nn.Module):
    def __init__(
        self,
        all_genes,
        do_ensembl_conversion,
        gene_name_path,
        gene_token_path,
        config_dict,
        gp_latent_size,
        use_gf_embeddings=None,
        condition_on_length=False,
    ):
        super().__init__()

        # Intiialize lookup table for vocab
        # Set word embeddings to Geneformer embeddings
        if use_gf_embeddings == 'gf-12L-95M-i4096':
            geneformer = BertForMaskedLM.from_pretrained(
                '/lustre/scratch126/cellgen/team361/mm58/Geneformer/gf-12L-95M-i4096'
            )
            gene_emb_weight = geneformer.bert.embeddings.word_embeddings.weight.data

            self.gene_embeddings = nn.Embedding.from_pretrained(
                gene_emb_weight,
                padding_idx=0,
                # freeze=config_dict['freeze_word_emb'],
            )

            if '16' in str(config_dict['torch_dtype']):
                self.gene_embeddings.half()

        elif isinstance(use_gf_embeddings, str):
            self.gene_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(np.load(use_gf_embeddings)),
                padding_idx=0,
            )

        else:
            self.gene_embeddings = nn.Embedding(
                config_dict['tokenization_vocab_size'],
                config_dict['hidden_size'],
                padding_idx=0,
            )

        # Look up table for re-encoding tokens to max vocab size
        if isinstance(all_genes, dict):
            hvg_tokens = get_gp_tokens(
                all_genes['hvg'],
                do_ensembl_conversion,
                'GP and HVG genes union',
                gene_token_path,
                gene_name_path,
            )

            gp_tokens = get_gp_tokens(
                all_genes['gp_genes'],
                do_ensembl_conversion,
                'GP genes',
                gene_token_path,
                gene_name_path,
            )

            gene_tokens = list(hvg_tokens) + list(gp_tokens)
            no_mask_tokens = [0, 1, 2, 3] + list(gp_tokens)
        else:
            gene_tokens = get_gp_tokens(
                all_genes,
                do_ensembl_conversion,
                'GP and HVG genes union',
                gene_token_path,
                gene_name_path,
            )

            gene_tokens = sorted(list(gene_tokens))

            # [0, 1, 2, 3] based on Geneformer vocab
            no_mask_tokens = [0, 1, 2, 3]

        gene_tokens_tensor = torch.tensor(gene_tokens, dtype=torch.int32)

        self.register_buffer('gene_tokens', gene_tokens_tensor)

        lookup_tensor = torch.full(
            (config_dict['tokenization_vocab_size'],), -100, dtype=torch.int32
        )

        indices = torch.arange(gene_tokens_tensor.shape[0], dtype=torch.int32)

        lookup_tensor[gene_tokens_tensor.long()] = indices

        self.register_buffer('gene_tokens_lookup', lookup_tensor)

        # Initialize transformer encoder model for getting gene embeddings

        self.model = gpTransformerEncoder(
            n_gp_tokens=len(gene_tokens),
            embed_dim=config_dict['hidden_size'],
            depth=config_dict['num_hidden_layers'],
            num_heads=config_dict['num_attention_heads'],
            mlm_masking_prob=config_dict['mlm_masking_prob'],
            seq_len=config_dict['max_seq_len'],
            use_pos_emb=config_dict['use_pos_emb'],
            use_l2_norm=config_dict['use_l2_norm'],
            output_dim=gp_latent_size,
            use_flash=config_dict['use_flash'],
            no_mask_tokens=no_mask_tokens,
            condition_on_length=condition_on_length,
        )

        self.max_seq_len = config_dict['max_seq_len']

    def forward(
        self,
        input_dataset,
        masking,
        return_mean_non_padding=False,
    ):
        # print('idx', input_dataset['idx'])
        # Clean up - remove Geneformer cls since we add our own
        input_dataset['input_ids'] = input_dataset['input_ids'][:, 1:]

        # input is tokenized dataset
        # get gene embeddings based on input_ids
        if hasattr(self, 'max_seq_len') and self.max_seq_len:
            if input_dataset['input_ids'].shape[1] > self.max_seq_len:
                input_dataset['input_ids'] = input_dataset['input_ids'][
                    :, : self.max_seq_len
                ]

        genes = self.gene_embeddings(input_dataset['input_ids'])

        # encode gene labels
        labels_unencoded = input_dataset['input_ids']
        labels = self.gene_tokens_lookup[labels_unencoded].long()

        # build attention mask
        attn_mask = torch.zeros(labels.shape[0], labels.shape[1]).to(labels.device)
        attn_mask[labels != -100] = 1
        # add row for cls
        attn_mask = torch.cat(
            [torch.ones(attn_mask.shape[0], 1).to(attn_mask.device), attn_mask], dim=1
        )

        # forward pass through encoder
        emb_out = self.model(
            genes,
            attn_mask=attn_mask,
            gene_labels=labels,
            masking=masking,
            return_attention=False,
            return_gene_embeddings=True,
            lengths=input_dataset['scaled_length'],
            return_mean_non_padding=return_mean_non_padding,
        )

        gene_output = {}
        gene_output['gene_emb'] = emb_out['gene_embeddings'].clone().detach()

        # print('Gene emb cell 1', gene_output['gene_emb'][0, :5, :5])
        # print(('Emb first 5 genes', emb_out['gene_embeddings'][:5, 0, :5]))
        # raise ValueError('stop')

        gene_output['gene_mlm_labels'] = emb_out['gene_labels']
        gene_output['gene_mlm_logits'] = emb_out['logits_lm']
        gene_output['gene_encoder_cls'] = emb_out['cls']

        return gene_output


####################################
# GP wrapper
####################################


class gpWrapper(nn.Module):
    def __init__(
        self,
        gp_inputs,
        database,
        do_ensembl_conversion,
        gene_token_path,
        gene_name_path,
        gp_latent_size,
        n_blocks,
        num_heads,
        mgm_mask_ratio,
        use_flash,
        model_type,
        learn_new_gp,
        use_pos_emb,
        fm_model_input_size,
        use_l2_norm,
        condition_on_length,
    ):
        super().__init__()

        self.gp_latent_size = gp_latent_size
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.mgm_mask_ratio = mgm_mask_ratio
        self.gp_inputs = gp_inputs
        self.model_type = model_type
        self.learning_new_gp = learn_new_gp
        self.gene_name_path = gene_name_path
        self.gene_token_path = gene_token_path

        # Get vocab size
        with open(gene_token_path, 'rb') as f:
            token_dict = pickle.load(f)
        self.vocab_size = max(token_dict.values())

        # Store all genes included in at least one GP
        self.all_gp_tokens = set()

        for i, gpi in enumerate(self.gp_inputs):
            gp_tokens = get_gp_tokens(
                database[gpi],
                do_ensembl_conversion,
                gpi,
                # gene_counts_df, # could edit to rm rare tokens?
                gene_token_path,
                gene_name_path,
            )

            gp_tokens_tensor = torch.tensor(list(gp_tokens), dtype=torch.int32)

            self.register_buffer(f'gp{i}_tokens', gp_tokens_tensor)

            print('Number of genes in GP', gpi, len(getattr(self, f'gp{i}_tokens')))
            self.all_gp_tokens.update(gp_tokens)

            # Set up look up tensor
            # for converting gene tokens to encoded values inside transformer block
            # +4 because in geneformer 0 --> padding
            # 1 --> mask
            # 2 --> cls
            # 3 --> eos
            if fm_model_input_size == 4096:
                lookup_tensor = torch.full(
                    (self.vocab_size + 4,), -100, dtype=torch.int32
                )
            else:
                lookup_tensor = torch.full(
                    (self.vocab_size + 2,), -100, dtype=torch.int32
                )
            # Create a tensor of indices corresponding to positions in gp_tokens
            indices = torch.arange(gp_tokens_tensor.shape[0], dtype=torch.int32)

            # Use tensor indexing to assign values
            lookup_tensor[gp_tokens_tensor.long()] = indices
            self.register_buffer(f'gp{i}_tokens_lookup', lookup_tensor)

        self.encoder: nn.ModuleList[gpTransformerEncoder] = nn.ModuleList(
            [
                gpTransformerEncoder(
                    n_gp_tokens=len(getattr(self, f'gp{i}_tokens')),
                    embed_dim=self.gp_latent_size,
                    depth=self.n_blocks,
                    num_heads=self.num_heads,
                    mlm_masking_prob=self.mgm_mask_ratio,
                    use_flash=use_flash,
                    seq_len=fm_model_input_size,
                    use_pos_emb=use_pos_emb,
                    use_l2_norm=use_l2_norm,
                    no_mask_tokens=[0, 1, 2, 3],
                    condition_on_length=condition_on_length,
                )
                for i in range(len(gp_inputs))
            ]
        )

    def forward(
        self,
        gf_emb_dict,
        input_dataset,
        masking=False,
        return_attention=False,
        return_gene_embeddings=False,
        tokens_to_keep=None,
        gp_of_interest=None,
        return_mean_non_padding=False,
    ):
        # Subset GP embeddings
        gp_token_list = []
        logits_lm_list = []
        gene_labels_list = []
        gene_original_labels_list = []
        num_genes_per_cell_list = []
        gene_emb_list = []

        # Extract embeddings for each gene program
        for i in range(len(self.gp_inputs)):
            if (gp_of_interest is None) or (self.gp_inputs[i] == gp_of_interest):
                # ensure max inputs ids matches gene encoder output
                if (
                    gf_emb_dict['gene_emb'].shape[1]
                    != input_dataset['input_ids'].shape[1]
                ):
                    input_ids = input_dataset['input_ids'][
                        :, : gf_emb_dict['gene_emb'].shape[1]
                    ]
                else:
                    input_ids = input_dataset['input_ids']

                (
                    emb_pad,
                    tokens_pad,
                    num_genes_per_cell,
                    attn_mask,
                ) = build_gp_input_matrix(
                    gf_emb_dict['gene_emb'],  # geneformer embeddings
                    input_ids,
                    getattr(self, f'gp{i}_tokens'),
                )

                # track number of genes per cell
                # divide by GP length
                num_genes_per_cell = num_genes_per_cell / (
                    getattr(self, f'gp{i}_tokens').shape[0]
                )
                num_genes_per_cell_list += [num_genes_per_cell]

                tokens_pad_unencoded = tokens_pad
                tokens_pad = getattr(self, f'gp{i}_tokens_lookup')[tokens_pad].long()

                # get token GP representation, logits for gene level prediction,
                # and gene_labels where masked genes = -100
                encoder_output = self.encoder[i](
                    emb_pad,
                    attn_mask=attn_mask,
                    gene_labels=tokens_pad,
                    masking=masking,
                    return_attention=return_attention,
                    return_gene_embeddings=return_gene_embeddings,
                    lengths=num_genes_per_cell,
                    return_mean_non_padding=return_mean_non_padding,
                )

                gp_token_list.append(encoder_output['cls'])
                logits_lm_list.append(encoder_output['logits_lm'])
                gene_labels_list.append(encoder_output['gene_labels'])

                if return_gene_embeddings:
                    gene_emb_list = encoder_output['gene_embeddings']
                    gene_original_labels_list = tokens_pad_unencoded
            else:
                continue

        # Concatenate tensors
        z = torch.stack(gp_token_list, dim=1)

        # store for output
        output = {
            'z': z,
            'logits_lm_list': logits_lm_list,
            'gene_labels_list': gene_labels_list,
            'gene_emb_list': gene_emb_list,
            'gene_original_labels_list': gene_original_labels_list,
            'num_genes_per_cell_list': num_genes_per_cell_list,
        }

        if return_gene_embeddings:
            output = self.wrangle_gene_embeddings(output, tokens_to_keep)

        return output

    def wrangle_gene_embeddings(self, emb_dict, tokens_to_keep):
        gene_emb = emb_dict['gene_emb_list']
        token_labels = emb_dict['gene_original_labels_list']

        output = {}

        for gene in tokens_to_keep:
            # zero out other genes
            mask = token_labels.unsqueeze(2) == gene
            mask = mask.to(torch.int)
            mask_expanded = mask.sum(dim=-1).unsqueeze(2)

            masked_emb = gene_emb * mask_expanded

            # Find the indices of the non-zero vectors
            non_zero_mask = torch.norm(masked_emb, dim=2) != 0
            indices = non_zero_mask.nonzero(as_tuple=True)

            # Initialize the result tensor with zeros
            result = torch.zeros(gene_emb.shape[0], gene_emb.shape[-1]).to(
                gene_emb.device
            )

            # Initialize the rank tensor with -1
            # (or any invalid index, indicating 'not found')
            rank = -torch.ones(gene_emb.shape[0], dtype=torch.int64).to(gene_emb.device)

            # Check if there are any non-zero rows, and update the result tensor
            if indices[0].nelement() != 0:
                result[indices[0]] = masked_emb[indices[0], indices[1]]
                rank[indices[0]] = indices[1]

            output[gene] = result
            output[f'{gene}_rank'] = rank

        return output

    def get_gene_gene_attention(self, gf_emb, input_dataset, gp_idx):
        '''
        If multilpe blocks, get attn matrix from last transformer block
        '''

        gp_tokens = getattr(self, f'gp{gp_idx}_tokens')

        # Extract embeddings for the gene program of interest
        emb_pad, tokens_pad, _, attn_mask = build_gp_input_matrix(
            gf_emb,  # geneformer embeddings
            input_dataset['input_ids'],
            gp_tokens,
        )

        # Encode tokens for MLM
        tokens_pad_unencoded = tokens_pad
        tokens_pad = getattr(self, f'gp{gp_idx}_tokens_lookup')[tokens_pad].long()

        # get token GP representation, logits for gene level prediction,
        # and gene_labels where masked genes = -100
        encoder_output = self.encoder[gp_idx](
            emb_pad,
            attn_mask=attn_mask,
            gene_labels=tokens_pad,
            masking=False,
            return_attention=True,
        )

        attn = encoder_output['attention']

        # Average attention across heads
        attn = attn.mean(dim=1)

        # Drop cls for gene-gene attention scores
        attn = attn[:, 1:, 1:]

        # Reorder attention matrix so GP genes are in the same order in each cell
        # Step 0 : Encode reference tokens for indexing
        genes = getattr(self, f'gp{gp_idx}_tokens_lookup')[gp_tokens].long()

        # Step 1: Replace padding tokens (-100) with a unique index value
        # that can be ignored during reordering
        n = genes.shape[0]
        unique_padding_index = n
        tokens_pad = tokens_pad.clone()
        tokens_pad[tokens_pad == -100] = unique_padding_index

        # Step 2: Create a tensor to hold the new order indices
        # Use advanced indexing to map the new order according to hvg
        gene_map = torch.full((n + 1,), unique_padding_index, dtype=torch.long).to(
            tokens_pad.device
        )
        gene_map[:n] = genes

        new_order_indices = gene_map[tokens_pad]

        # Step 3: Use the reordered indices to permute the attn tensor
        # Mask out padding tokens before reordering
        valid_mask = new_order_indices != unique_padding_index

        # Create the batch index tensor
        b = attn.shape[0]
        batch_indices = torch.arange(b).unsqueeze(1).expand(b, n)

        # Permute rows
        attn_reordered = attn[
            batch_indices,
            new_order_indices.where(valid_mask, torch.zeros_like(new_order_indices)),
        ]

        # Permute columns
        attn_reordered = attn_reordered.transpose(1, 2)[
            batch_indices,
            new_order_indices.where(valid_mask, torch.zeros_like(new_order_indices)),
        ].transpose(1, 2)

        # Set attention scores for padding tokens to 0
        padding_mask = tokens_pad_unencoded == unique_padding_index

        attn_reordered[padding_mask.unsqueeze(2).expand_as(attn_reordered)] = 0
        attn_reordered[padding_mask.unsqueeze(1).expand_as(attn_reordered)] = 0

        # Reorder attention matrix so genes are in the same order in each cell
        output = {
            'attn': attn_reordered,
        }

        return output

    def get_cls_attn(self, gf_emb, input_dataset, gp_idx):
        '''
        If multilpe blocks, get attn matrix from last transformer block
        '''

        gp_tokens = getattr(self, f'gp{gp_idx}_tokens')

        # Extract embeddings for the gene program of interest
        emb_pad, tokens_pad, _, attn_mask = build_gp_input_matrix(
            gf_emb,  # geneformer embeddings
            input_dataset['input_ids'],
            gp_tokens,
        )

        # Encode tokens for MLM
        tokens_pad_unencoded = tokens_pad
        tokens_pad = getattr(self, f'gp{gp_idx}_tokens_lookup')[tokens_pad].long()

        # get token GP representation
        encoder_output = self.encoder[gp_idx](
            emb_pad,
            attn_mask=attn_mask,
            gene_labels=tokens_pad,
            masking=False,
            return_attention=True,
        )

        attn = encoder_output['attention']

        # Average attention across heads
        attn = attn.mean(dim=1)

        output = {}

        # Select cls attention
        attn = attn[:, 0, :]

        output['cls'] = attn[:, 0].cpu().detach().numpy()

        # keep only gene scores
        attn = attn[:, 1:]

        for i, gene in enumerate(gp_tokens):
            # Ensure the data types match
            gene = gene.to(tokens_pad_unencoded.dtype)

            # zero out other genes
            mask = (tokens_pad_unencoded == gene).to(torch.int)

            masked_score = attn * mask

            # Find the indices of the non-zero scores
            non_zero_mask = masked_score != 0

            indices = non_zero_mask.nonzero(as_tuple=True)

            # Initialize the result tensor with zeros
            result = torch.zeros(attn.shape[0], attn.shape[-1]).to(attn.device)

            # Check if there are any non-zero rows, and update the result tensor
            if indices[0].nelement() != 0:
                result[indices] = masked_score[indices]

                # for debugging
                for row_idx in torch.unique(indices[0]):
                    if non_zero_mask[row_idx].sum() > 1:
                        raise ValueError('Multiple non-zero scores for the same gene')

            output[gene.item()] = result.cpu().detach().numpy().sum(axis=-1)

        return output


class cellWrapper(nn.Module):
    def __init__(
        self,
        gp_inputs,
        n_blocks,
        num_heads,
        gp_latent_size,
        global_masking_rate,
        use_flash,
        use_l2_norm,
    ):
        super().__init__()

        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.gp_inputs = gp_inputs
        self.gp_latent_size = gp_latent_size

        self.encoder = gpTransformerEncoder(
            n_gp_tokens=len(self.gp_inputs),
            embed_dim=self.gp_latent_size,
            depth=self.n_blocks,
            num_heads=self.num_heads,
            mlm_masking_prob=global_masking_rate,
            use_flash=use_flash,
            use_l2_norm=use_l2_norm,
            no_mask_tokens=[
                len(self.gp_inputs) + 1
            ],  # gp token labels are 0 to len(gp_inputs)-1
        )

    def build_input_matrix(self, z, num_genes_per_cell_list):
        # Prepare labels
        batch_size = z.shape[0]
        gp_labels = torch.tensor(
            [[i for i in range(len(self.gp_inputs))]] * batch_size
        ).to(z.device)

        # reorder gp based on number of genes per cell
        # n_genes_per_cell = torch.tensor(np.array(num_genes_per_cell_list).T).to(
        #     z.device
        # )
        n_genes_per_cell = torch.stack(num_genes_per_cell_list).T.to(z.device)

        # Find the indices that would sort each row in descending order
        sorted_indices = torch.argsort(-n_genes_per_cell, dim=1)

        # Create a mask where genes per cell are zero
        zero_mask = n_genes_per_cell == 0
        # reorder the mask based on sorted indices
        zero_mask = torch.gather(zero_mask, 1, sorted_indices)

        # Zero out positions where there are zero genes per cell
        z = torch.where(
            zero_mask.unsqueeze(-1),
            torch.zeros_like(z),
            torch.gather(
                z, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, self.gp_latent_size)
            ),
        )
        # label should be -100 where there are zero genes per cell
        gp_labels = torch.where(
            zero_mask,
            torch.ones_like(gp_labels) * -100,
            torch.gather(gp_labels, 1, sorted_indices),
        )

        # don't pay attention to those GP
        attn_mask = torch.zeros(gp_labels.shape[0], gp_labels.shape[1])
        attn_mask[gp_labels != -100] = 1

        # never mask cls
        attn_mask = torch.cat(
            [torch.ones(attn_mask.shape[0], 1).to(attn_mask.device), attn_mask], dim=1
        ).to(z.device)

        return z, gp_labels, attn_mask

    def forward(self, x, masking):
        """
        Input is the dictionary output of gpWrapper
        we need keys z and number of genes per cell
        """
        z, gp_labels, attn_mask = self.build_input_matrix(
            z=x['z'], num_genes_per_cell_list=x['num_genes_per_cell_list']
        )

        encoder_output = self.encoder(
            z,
            gene_labels=gp_labels,
            attn_mask=attn_mask,
            masking=masking,
            return_attention=False,
        )

        output = {
            'cell_token': encoder_output['cls'],
            'gp_logits_lm': encoder_output['logits_lm'],
            'gp_labels': encoder_output['gene_labels'],
        }

        return output

    def get_attn(self, x):
        """
        Input is the dictionary output of gpWrapper
        we need keys z and number of genes per cell
        """

        z, gp_labels, attn_mask = self.build_input_matrix(
            z=x['z'], num_genes_per_cell_list=x['num_genes_per_cell_list']
        )

        encoder_output = self.encoder(
            z,
            gene_labels=gp_labels,
            attn_mask=attn_mask,
            masking=False,
            return_attention=True,
        )

        # Reorder attention matrix so GP are in the same order in each cell
        attn = encoder_output['attention']

        # Average attention across heads
        attn = attn.mean(dim=1)

        # And focus on cls attention scores
        attn = attn[:, 0, :]

        output = {}

        output['cls'] = attn[:, 0].cpu().detach().numpy()

        # drop cls token
        attn = attn[:, 1:]

        for i, gp in enumerate(self.gp_inputs):
            # zero out other GP
            mask = (gp_labels == i).to(torch.int)

            masked_score = attn * mask

            # Find the indices of the non-zero scores
            non_zero_mask = masked_score != 0

            indices = non_zero_mask.nonzero(as_tuple=True)

            # Initialize the result tensor with zeros
            result = torch.zeros(attn.shape[0], attn.shape[-1]).to(attn.device)

            # Check if there are any non-zero rows, and update the result tensor
            if indices[0].nelement() != 0:
                result[indices] = masked_score[indices]

                # for debugging
                for row_idx in torch.unique(indices[0]):
                    if non_zero_mask[row_idx].sum() > 1:
                        raise ValueError('Multiple non-zero scores for the same gene')

            output[gp] = result.cpu().detach().numpy().sum(axis=-1)

        return output


####################################
# Count reconstruction
####################################


class CountHead(nn.Module):
    def __init__(
        self,
        loss_mode: str = 'mse',
        n_genes: int = 25426,
        d_model: int = 512,
    ):
        super().__init__()
        self.loss_mode = loss_mode

        self.mlp = Mlp(d_model, d_model)

        if self.loss_mode == 'mse':
            self.relu_output = nn.Sequential(nn.Linear(d_model, n_genes), nn.ReLU())

        elif self.loss_mode == 'zinb':
            self.linear_output = nn.Linear(d_model, n_genes)
            self.softmax_output = nn.Sequential(
                nn.Linear(d_model, n_genes), nn.Softmax(dim=-1)
            )

        elif self.loss_mode == 'nb':
            self.softmax_output = nn.Sequential(
                nn.Linear(d_model, n_genes), nn.Softmax(dim=-1)
            )

    def forward(self, x):
        # use cls token for count prediction
        count_outputs = {}
        mlp_output = self.mlp(x)
        # mlp_output = F.normalize(mlp_output, dim=-1, p=2)
        if self.loss_mode == 'mse':
            count_outputs['count_lognorm'] = self.relu_output(mlp_output)
        elif self.loss_mode == 'zinb':
            count_outputs['count_mean'] = self.softmax_output(mlp_output)
            count_outputs['count_dropout'] = self.linear_output(mlp_output)
        elif self.loss_mode == 'nb':
            count_outputs['count_mean'] = self.softmax_output(mlp_output)
        return count_outputs


class BinDecoder(nn.Module):
    '''
    Adapted from scGPT
    https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/model.py#L848
    accessed 03.04.24

    scGPT output has one dimension -> per gene
    here we need to reconstruct bins for n genes

    '''

    def __init__(
        self,
        n_genes: int = 25426,
        d_model: int = 512,
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, n_genes),
        )

    def forward(self, x):
        return self.fc(x)


####################################
# Define model
####################################


class gpTransformerBase(nn.Module):
    """
    Model to learn GP latent representation

    """

    def __init__(
        self,
        database,
        attn_dropout=0,
        gp_inputs=None,
        do_ensembl_conversion=True,
        num_heads=1,
        n_blocks=1,
        mgm_mask_ratio=0.5,
        use_flash=False,
        fm_encoder_pkg='geneformer',
        fm_encoder_name='gf-6L-30M-i2048',
        peft_config_path=None,
        fm_layer_to_quant=-1,
        model_type='Base',
        learn_new_gp=False,
        gp_of_interest=None,
        use_pos_emb='sin_cos',
        use_onehot_wrapper=False,
        vocab_gene_names=None,
        bert_config=None,
        gp_latent_size=None,
        use_gf_embeddings=False,
        use_l2_norm=False,
        all_genes=None,
        condition_on_length=False,
        warmup=0,
    ):
        """
        database :
            pandas dataframe with GP names as columns and genes as rows

        gp_inputs:
            list of gene programs to include in model.
            If None, defaults to all GPs in database

        gene_counts_df:
            dataframe where columns are
            ["gene", "ensembl", "token", "counts", "prop", "total"]
            where prop represents the proportion of cells in the main dataset
            that express a given gene
            filtered so that prop is above a user-specified threshold

        do_ensembl_conversion :
            whether gene names in database need to be converted to ensembl ID

        gp_latent_size :
            dimension for each GP latent representation

        num_heads :
            number of heads in self-attention blocks

        n_blocks :
            number of self-attention blocks

        mgm_mask_ratio :
            ratio of genes to mask for MGM task (nb this number
            represents the proportion of genes which will be kept)

        geneformer_model :
            path to pretrained geneformer model

        fm_layer_to_quant :
            layer of geneformer to use for extracting embeddings
            (added to numnber of total layers so that -1 corresponds
            to penumltimate layer)

        gene_token_path :
            path to gene token dictionary {ensembl_id : token}
            as provided by Geneformer

        gene_name_path :
            path to gene name dictionary {gene_name : ensembl_id}
            as provided by Geneformer

        """
        super().__init__()

        self.fm_encoder_pkg = fm_encoder_pkg
        self.fm_encoder_name = fm_encoder_name
        self.use_l2_norm = use_l2_norm
        self.condition_on_length = condition_on_length
        self.warmup = warmup

        if fm_encoder_pkg == 'geneformer':
            geneformer_repo_path = get_gf_repo()

            # Initialize geneformer model for getting geneformer embeddings
            if use_onehot_wrapper:
                self.gf_wrapper = OneHotWrapper(gene_names=vocab_gene_names)
            else:
                # Unpack arguments for geneformer model
                geneformer_model = os.path.join(
                    geneformer_repo_path,
                    fm_encoder_name,
                )

                # Load config json file
                gf_config = BertConfig.from_pretrained(geneformer_model)

                gp_latent_size = gf_config.hidden_size

                fm_model_input_size = gf_config.max_position_embeddings

                if fm_model_input_size == 4096:
                    self.gene_token_path = TOKEN_DICTIONARY_FILE
                    self.gene_name_path = ENSEMBL_DICTIONARY_FILE
                else:
                    self.gene_token_path = os.path.join(
                        geneformer_repo_path,
                        'geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl',
                    )

                    self.gene_name_path = os.path.join(
                        geneformer_repo_path,
                        'geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl',
                    )

                self.gf_wrapper = gfWrapper(
                    geneformer_model=geneformer_model,
                    fm_layer_to_quant=fm_layer_to_quant,
                    peft_config_path=peft_config_path,
                    token_dictionary_file=self.gene_token_path,
                    # max_len=fm_model_input_size,
                )
        elif fm_encoder_pkg == 'geneformer_2021':
            geneformer_repo_path = get_gf_repo()
            geneformer_model = fm_encoder_name

            gp_latent_size = 256
            fm_model_input_size = 2048
            self.gene_token_path = os.path.join(
                geneformer_repo_path,
                'geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl',
            )

            self.gene_name_path = os.path.join(
                geneformer_repo_path,
                'geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl',
            )

            self.gf_wrapper = gfWrapper(
                geneformer_model=geneformer_model,
                fm_layer_to_quant=fm_layer_to_quant,
                peft_config_path=peft_config_path,
                token_dictionary_file=self.gene_token_path,
            )

        elif fm_encoder_pkg == 'from_scratch':
            fm_model_input_size = bert_config['tokenization_input_size']

            if gp_latent_size is None:
                gp_latent_size = bert_config['hidden_size']

            geneformer_repo_path = get_gf_repo()

            if fm_model_input_size == 4096:
                self.gene_token_path = TOKEN_DICTIONARY_FILE
                self.gene_name_path = ENSEMBL_DICTIONARY_FILE
            else:
                self.gene_token_path = os.path.join(
                    geneformer_repo_path,
                    'geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl',
                )

                self.gene_name_path = os.path.join(
                    geneformer_repo_path,
                    'geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl',
                )

            self.gf_wrapper = GeneWrapper(
                config_dict=bert_config,
                gene_name_path=self.gene_name_path,
                gene_token_path=self.gene_token_path,
                all_genes=all_genes,
                do_ensembl_conversion=do_ensembl_conversion,
                use_gf_embeddings=use_gf_embeddings,
                gp_latent_size=gp_latent_size,
                condition_on_length=condition_on_length,
            )

        else:
            raise ValueError('Only geneformer is supported for now')

        # Track for downstream models
        self.fm_encoder_pkg = fm_encoder_pkg
        self.fm_encoder_name = fm_encoder_name
        self.gp_latent_size = gp_latent_size
        self.fm_model_input_size = fm_model_input_size

        # # Optionally: extract Geneformer cell embeddings
        # only need if MSE with gf cell embedding
        # self.gf_cell_encoder = AverageNonZero()
        # for backwards compatibility
        self.gf_cell_encoder = nn.Identity()

        # Set up token sets for each gene program
        if gp_inputs is None:
            gp_inputs = database.columns.tolist()
        elif isinstance(gp_inputs, str):
            gp_inputs = [gp_inputs]

        # / cause issues with saving
        gp_inputs = [x.replace('/', '_') for x in gp_inputs]
        database.columns = [x.replace('/', '_') for x in database.columns]

        self.gpdb = database[gp_inputs]
        self.gp_inputs = gp_inputs
        self.mgm_mask_ratio = mgm_mask_ratio
        self.do_ensembl_conversion = do_ensembl_conversion
        self.n_blocks = n_blocks
        self.attn_dropout = attn_dropout
        self.use_flash = use_flash

        if isinstance(gp_of_interest, str):
            gp_of_interest = [gp_of_interest]
        self.gp_of_interest = gp_of_interest

        self.multi_gp_encoder = gpWrapper(
            database=self.gpdb,
            do_ensembl_conversion=self.do_ensembl_conversion,
            gene_token_path=self.gene_token_path,
            gene_name_path=self.gene_name_path,
            gp_latent_size=self.gp_latent_size,
            n_blocks=self.n_blocks,
            num_heads=num_heads,
            mgm_mask_ratio=self.mgm_mask_ratio,
            gp_inputs=gp_inputs,
            use_flash=self.use_flash,
            model_type=model_type,
            learn_new_gp=learn_new_gp,
            use_pos_emb=use_pos_emb,
            fm_model_input_size=fm_model_input_size,
            use_l2_norm=use_l2_norm,
            condition_on_length=condition_on_length,
        )

    def forward(
        self,
        input_dataset,
        return_gene_embeddings=False,
        return_attention=False,
        tokens_to_keep=None,
        return_gf_cell_emb=False,
        gp_of_interest=None,
        masking=False,
        epoch=None,
    ):
        # input is tokenized dataset
        emb_out = self.gf_wrapper(
            input_dataset,
            masking=masking,
        )

        if hasattr(self, 'warmup') and isinstance(epoch, int) and epoch < self.warmup:
            return emb_out

        # Extract embeddings for each gene program
        # For backwards compataibilty
        # if not gp_of_interest attribute set to None
        if not hasattr(self, 'gp_of_interest'):
            self.gp_of_interest = None

        gp_to_pass = self.gp_of_interest if gp_of_interest is None else gp_of_interest

        output = self.multi_gp_encoder(
            emb_out,
            input_dataset,
            masking=masking,
            return_gene_embeddings=return_gene_embeddings,
            return_attention=return_attention,
            tokens_to_keep=tokens_to_keep,
            gp_of_interest=gp_to_pass,
        )

        # Optionally return logits for gene encoder
        if 'gene_mlm_logits' in emb_out:
            output['gene_mlm_labels'] = emb_out['gene_mlm_labels']
            output['gene_mlm_logits'] = emb_out['gene_mlm_logits']
            output['gene_encoder_cls'] = emb_out['gene_encoder_cls']

        # Optionally return geneformer cell embeddings
        if return_gf_cell_emb:
            raise NotImplementedError('Not implemented')
            gf_output_dict = self.gf_cell_encoder(emb_out)
            output['gf_emb'] = gf_output_dict['cls']

        return output

    def get_gene_gene_attn(self, input_dataset, gp_idx):
        gf_emb = self.gf_wrapper(input_dataset)

        output = self.multi_gp_encoder.get_gene_gene_attn(
            gf_emb, input_dataset, gp_idx=gp_idx
        )

        return output

    def get_cls_attn(self, input_dataset, gp):
        # Get gp index
        gp_idx = self.gp_inputs.index(gp)

        # Get Geneformer embeddings
        gf_emb = self.gf_wrapper(input_dataset)

        output = self.multi_gp_encoder.get_cls_attn(
            gf_emb, input_dataset, gp_idx=gp_idx
        )

        return output


class gpTransformerGlobal(gpTransformerBase):
    """
    Learn individual GP representations + global cell token
    """

    def __init__(
        self,
        global_attn_heads=8,
        global_loss='reconstruction',
        total_n_genes=25426,
        reconstruction_loss='nb',
        supervised_labels: Optional[Dict] = None,
        global_masking_rate=0,
        global_n_blocks=1,
        use_flash=False,
        use_l2_norm=False,
        n_bins=10,
        **kwargs,
    ):
        super().__init__(
            use_flash=use_flash,
            model_type='Global',
            use_l2_norm=use_l2_norm,
            **kwargs,
        )
        self.global_attn_heads = global_attn_heads

        self.global_loss = global_loss

        self.cell_token_learner = cellWrapper(
            gp_inputs=self.gp_inputs,
            gp_latent_size=self.gp_latent_size,
            n_blocks=global_n_blocks,
            num_heads=self.global_attn_heads,
            global_masking_rate=global_masking_rate,
            use_flash=use_flash,
            use_l2_norm=use_l2_norm,
        )

        if self.global_loss == 'supervised':
            if supervised_labels is None:
                raise ValueError(
                    'Please provide a dictionary of the form {task_name : n_classes}'
                )

            for k, n in supervised_labels.items():
                setattr(self, f'{k}_n_class', n)

            self.supervised_tasks = {
                t: i for i, t in enumerate(supervised_labels.keys())
            }

            self.clf_head = nn.ModuleList(
                [
                    nn.Linear(self.gp_latent_size, getattr(self, f'{k}_n_class'))
                    for k in supervised_labels.keys()
                ]
            )

        if self.global_loss == 'reconstruction':
            self.reconstruction_loss = reconstruction_loss

            if reconstruction_loss == 'binning':
                self.n_bins = n_bins
                self.count_head = BinDecoder(
                    n_genes=total_n_genes, d_model=self.gp_latent_size
                )

            else:
                self.count_head = CountHead(
                    loss_mode=reconstruction_loss,
                    n_genes=total_n_genes,
                    d_model=self.gp_latent_size,
                )

    def forward(
        self,
        input_dataset,
        return_gene_embeddings=False,
        return_attention=False,
        tokens_to_keep=None,
        gp_of_interest=None,
        masking=False,
        masking_global=False,
    ):
        return_gf_cell_emb = True if self.global_loss == 'mse' else False

        base_output = super().forward(
            input_dataset,
            return_gene_embeddings,
            return_attention,
            tokens_to_keep,
            return_gf_cell_emb,
            gp_of_interest=gp_of_interest,
            masking=masking,
        )

        if return_gene_embeddings:
            return base_output
        cell_output = self.cell_token_learner(base_output, masking=masking_global)

        base_output['cell_token'] = cell_output['cell_token']

        if self.global_loss == 'supervised':
            for t, i in self.supervised_tasks.items():
                base_output[f'logits_{t}'] = self.clf_head[i](cell_output['cell_token'])

        elif self.global_loss == 'masking':
            base_output['gp_logits_lm'] = cell_output['gp_logits_lm']
            base_output['gp_labels'] = cell_output['gp_labels']

        elif self.global_loss == 'reconstruction':
            count_output = self.count_head(cell_output['cell_token'])
            base_output['count_output'] = count_output

            if self.reconstruction_loss == 'binning':
                if self.training:
                    binned = bin_gene_expression(
                        input_dataset['counts'], n_bins=self.n_bins
                    )
                    binned = torch.tensor(binned).to(count_output.device)
                    base_output['true_bins'] = binned

        return base_output

    def get_cell_token_attention(self, input_dataset):
        base_output = super().forward(input_dataset)

        output = self.cell_token_learner.get_attn(base_output)

        return output


####################################
# Models with additional heads/losses
####################################


# ----------------------------------
# Condition on Geneformer mean embedding
# ----------------------------------

# class gpWrapperCondMean(gpWrapper):
#     def __init__(self, mean_emb_dict, **kwargs):
#         super().__init__(**kwargs)

#         self.mean_emb_dict = mean_emb_dict
#         if self.mean_emb_dict is not None:
#             # Load precomputed GP gene embeddings from the pickle file
#             with open(mean_emb_dict, 'rb') as f:
#                 z_mean = pickle.load(f)

#             # Convert strings to integers
#             z_mean = {int(k): v for k, v in z_mean.items()}

#             # Convert z_mean dictionary to a tensor for efficient indexing
#             max_token_id = max(z_mean.keys())
#             z_mean_tensor = torch.zeros(
#                 (max_token_id + 1, next(iter(z_mean.values())).shape[0])
#             )

#             for token_id, embedding in z_mean.items():
#                 z_mean_tensor[token_id] = torch.tensor(embedding)

#             self.register_buffer('z_mean', z_mean_tensor)

#     def build_input_matrix(self, gf, input_ids, gp_tokens, crop_to_gp_len=True):

#         # condition on mean gene representation from geneformer
#         # Use input_ids to index into z_mean_tensor and get embeddings
#         # in masked_labels, temporarily convert to 0 (padding token)
#         masked_labels_output[masked_labels_output == -100] = 0

#         z_mean_embeddings = self.z_mean[masked_labels_output]  # Shape: (b, e, e2)

#         result_matrix = torch.cat([result_matrix, z_mean_embeddings], dim=-1)

#         # Convert back
#         masked_labels_output[masked_labels_output == 0] = -100

#         return result_matrix, masked_labels_output, num_genes_per_cell, attn_mask


# class gpTransformerCondMean(gpTransformerGlobal):
#     # mean_emb_dict=None,
#             # If using PRBM, set up the buffers on GPU
#         if prbm is not None:
#             self.use_prbm = True
#             prbm = prbm[self.gp_inputs]
#             prbm_tensor = torch.tensor(prbm.values.T).to(torch.float32)
#             self.register_buffer('prbm', prbm_tensor)
#         else:
#             self.use_prbm = False

#         self.cond_to_shift = cond_to_shift


#     def forward(self):

#         # for each GP, optionally sum the reference mean embedding tensor
#         if self.use_prbm:
#             if self.cond_to_shift is not None:
#                 # cond_to_shift is of form {'name' : ['value']}
#                 condition_key = list(self.cond_to_shift.keys())[0]
#                 condition_values = self.cond_to_shift[condition_key]

#                 mask = torch.tensor(
#                     [name in condition_values
#                           for name in input_dataset[condition_key]],
#                     dtype=torch.bool,
#                 )
#                 mask = (
#                     mask.unsqueeze(-1).unsqueeze(-1).to(base_output['z'].device)
#                 )  # Ensure correct broadcasting
#                 prbm_expanded = self.prbm.unsqueeze(0).expand_as(base_output['z'])

#                 base_output['z'] = torch.where(
#                     mask,
#                     base_output['z'] + (base_output['z'] - prbm_expanded),
#                     base_output['z'],
#                 )
#             else:
#                 base_output['z'] = base_output['z']
#                       + (base_output['z'] - prbm_expanded)


# ----------------------------------
# Add virtual tokens
# ----------------------------------


class gpWrapperWithPrompt(gpWrapper):
    def __init__(
        self,
        use_flash,
        use_pos_emb,
        num_virtual_tokens,
        virtual_tokens_label,
        num_prompt_classes,
        **kwargs,
    ):
        super().__init__(use_flash=use_flash, use_pos_emb=use_pos_emb, **kwargs)

        self.num_virtual_tokens = num_virtual_tokens

        for i, gp in enumerate(self.gp_inputs):
            prompt_encoder = PromptEncoder(
                token_dim=self.gp_latent_size,
                encoder_hidden_size=int(self.gp_latent_size * 0.5),
                num_virtual_tokens=self.num_virtual_tokens,
            )

            setattr(self, f'prompt_encoder_gp{i}', prompt_encoder)

        self.encoder = nn.ModuleList(
            [
                gpTransformerEncoderWithPrompt(
                    n_gp_tokens=len(getattr(self, f'gp{i}_tokens')),
                    embed_dim=self.gp_latent_size,
                    depth=self.n_blocks,
                    num_heads=self.num_heads,
                    mlm_masking_prob=self.mgm_mask_ratio,
                    use_flash=use_flash,
                    use_pos_emb=use_pos_emb,
                )
                for i in range(len(self.gp_inputs))
            ]
        )

        # # freeze all other parameters
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        # add encoder for shared token
        self.prompt_encoder = PromptEncoder(
            token_dim=self.gp_latent_size,
            encoder_hidden_size=int(self.gp_latent_size * 0.5),
            num_virtual_tokens=self.num_virtual_tokens,
        )

        if virtual_tokens_label is not None:
            self.virtual_tokens_label = virtual_tokens_label
            self.prompt_clf = nn.Linear(self.gp_latent_size, num_prompt_classes)
        else:
            self.virtual_tokens_label = None

    def build_input_matrix(self, gf, input_ids, gp_tokens, crop_to_gp_len=True):
        (
            result_matrix,
            masked_labels_output,
            num_genes_per_cell,
            attn_mask,
        ) = super().build_input_matrix(gf, input_ids, gp_tokens, crop_to_gp_len)

        attn_mask = torch.cat(
            # times 2 because 1 GP-specific token + 1 non-specific token
            [
                attn_mask,
                torch.ones_like(attn_mask)[:, : 2 * self.num_virtual_tokens],
            ],
            dim=-1,
        )

        return result_matrix, masked_labels_output, num_genes_per_cell, attn_mask

    def forward(
        self,
        gf_emb,
        input_dataset,
        masking=False,
        return_attention=False,
        return_gene_embeddings=False,
        tokens_to_keep=None,
        gp_of_interest=None,
    ):
        # Subset GP embeddings
        gp_token_list = []
        logits_lm_list = []
        gene_labels_list = []
        gene_original_labels_list = []
        num_genes_per_cell_list = []
        gene_emb_list = []
        gp_virtual_token_logit_list = []
        shared_virtual_token_logit_list = []
        gp_virtual_token_list = []
        shared_virtual_token_list = []

        # Extract embeddings for each gene program
        for i in range(len(self.gp_inputs)):
            if (gp_of_interest is None) or (self.gp_inputs[i] in gp_of_interest):
                (
                    emb_pad,
                    tokens_pad,
                    num_genes_per_cell,
                    attn_mask,
                ) = build_gp_input_matrix(
                    gf_emb,  # geneformer embeddings
                    input_dataset['input_ids'],
                    getattr(self, f'gp{i}_tokens'),
                )

                # track number of genes per cell
                # divide by GP length
                num_genes_per_cell = num_genes_per_cell / (
                    getattr(self, f'gp{i}_tokens').shape[0]
                )
                num_genes_per_cell_list += [num_genes_per_cell]

                # Encode tokens for MLM
                tokens_pad_unencoded = tokens_pad
                tokens_pad = getattr(self, f'gp{i}_tokens_lookup')[tokens_pad].long()

                # append tokens for PEFT
                # get GP-specific token
                gp_virtual_token = getattr(self, f'prompt_encoder_gp{i}')(
                    torch.arange(self.num_virtual_tokens, device=emb_pad.device)
                )

                gp_virtual_token = gp_virtual_token.unsqueeze(0).expand(
                    emb_pad.shape[0], -1, -1
                )

                emb_pad = torch.cat([emb_pad, gp_virtual_token], dim=1)

                tokens_pad = torch.cat(
                    [
                        tokens_pad,
                        torch.tensor(
                            [-100] * self.num_virtual_tokens,
                            device=tokens_pad.device,
                        )
                        .unsqueeze(0)
                        .expand(tokens_pad.shape[0], -1),
                    ],
                    dim=1,
                )

                # Get shared token
                virtual_tokens = self.prompt_encoder(
                    torch.arange(self.num_virtual_tokens, device=emb_pad.device)
                )

                # Expand virtual tokens to match emb_pad
                virtual_tokens = virtual_tokens.unsqueeze(0).expand(
                    emb_pad.shape[0], -1, -1
                )

                emb_pad = torch.cat([virtual_tokens, emb_pad], dim=1)

                # Add to labels as well
                tokens_pad = torch.cat(
                    [
                        torch.tensor(
                            [-100] * self.num_virtual_tokens,
                            device=tokens_pad.device,
                        )
                        .unsqueeze(0)
                        .expand(tokens_pad.shape[0], -1),
                        tokens_pad,
                    ],
                    dim=1,
                )

                # get token GP representation, logits for gene level prediction,
                # and gene_labels where masked genes = -100
                encoder_output = self.encoder[i](
                    emb_pad,
                    attn_mask=attn_mask,
                    gene_labels=tokens_pad,
                    masking=masking,
                    return_attention=return_attention,
                    return_gene_embeddings=return_gene_embeddings,
                    num_virtual_tokens=self.num_virtual_tokens
                    * 2,  # *2 for shared and GP-specific tokens
                    using_gp_specific_token=self.num_virtual_tokens > 0,
                )
                gp_token_list.append(encoder_output['cls'])
                logits_lm_list.append(encoder_output['logits_lm'])
                gene_labels_list.append(encoder_output['gene_labels'])

                gp_virtual_token_list.append(encoder_output['gp_virtual_tokens'])
                shared_virtual_token_list.append(
                    encoder_output['shared_virtual_tokens']
                )

                if (
                    hasattr(self, 'virtual_tokens_label')
                    and self.virtual_tokens_label is not None
                ):
                    gpi_token_logits = self.prompt_clf(
                        encoder_output['gp_virtual_tokens']
                    )
                    gp_virtual_token_logit_list.append(gpi_token_logits)

                    shared_token_logits = self.prompt_clf(
                        encoder_output['shared_virtual_tokens']
                    )
                    shared_virtual_token_logit_list.append(shared_token_logits)

                if return_gene_embeddings:
                    gene_emb_list = encoder_output['gene_embeddings']
                    gene_original_labels_list = tokens_pad_unencoded
            else:
                continue

        # Concatenate tensors
        z = torch.stack(gp_token_list, dim=1)

        # store for output
        output = {
            'z': z,
            'logits_lm_list': logits_lm_list,
            'gene_labels_list': gene_labels_list,
            'gene_emb_list': gene_emb_list,
            'gene_original_labels_list': gene_original_labels_list,
            'num_genes_per_cell_list': num_genes_per_cell_list,
        }

        output['virtual_tokens'] = virtual_tokens  # for global model
        output['gp_virtual_token_logits'] = gp_virtual_token_logit_list
        output['shared_virtual_token_logits'] = shared_virtual_token_logit_list

        # for saving embeddings
        output['gp_virtual_tokens'] = gp_virtual_token_list
        output['shared_virtual_tokens'] = shared_virtual_token_list

        if return_gene_embeddings:
            output = self.wrangle_gene_embeddings(output, tokens_to_keep)

        return output

    def get_cls_attn(self, gf_emb, input_dataset, gp_idx, gene_names):
        '''
        If multilpe blocks, get attn matrix from last transformer block
        '''

        gp_tokens = getattr(self, f'gp{gp_idx}_tokens')

        # Extract embeddings for the gene program of interest
        emb_pad, tokens_pad, _, attn_mask = self.build_input_matrix(
            gf_emb,  # geneformer embeddings
            input_dataset['input_ids'],
            gp_tokens,
        )

        # Encode tokens for MLM
        tokens_pad_unencoded = tokens_pad
        tokens_pad = getattr(self, f'gp{gp_idx}_tokens_lookup')[tokens_pad].long()

        # get token GP representation
        # Optionally append tokens for PEFT
        if self.num_virtual_tokens > 0:
            # get GP-specific token
            gp_virtual_token = getattr(self, f'prompt_encoder_gp{gp_idx}')(
                torch.arange(self.num_virtual_tokens, device=emb_pad.device)
            )

            gp_virtual_token = gp_virtual_token.unsqueeze(0).expand(
                emb_pad.shape[0], -1, -1
            )

            emb_pad = torch.cat([emb_pad, gp_virtual_token], dim=1)

            tokens_pad = torch.cat(
                [
                    tokens_pad,
                    torch.tensor(
                        [-100] * self.num_virtual_tokens,
                        device=tokens_pad.device,
                    )
                    .unsqueeze(0)
                    .expand(tokens_pad.shape[0], -1),
                ],
                dim=1,
            )

            # Get shared token
            virtual_tokens = self.prompt_encoder(
                torch.arange(self.num_virtual_tokens, device=emb_pad.device)
            )

            # Expand virtual tokens to match emb_pad
            virtual_tokens = virtual_tokens.unsqueeze(0).expand(
                emb_pad.shape[0], -1, -1
            )

            emb_pad = torch.cat([virtual_tokens, emb_pad], dim=1)

            # Add to labels as well
            tokens_pad = torch.cat(
                [
                    torch.tensor(
                        [-100] * self.num_virtual_tokens,
                        device=tokens_pad.device,
                    )
                    .unsqueeze(0)
                    .expand(tokens_pad.shape[0], -1),
                    tokens_pad,
                ],
                dim=1,
            )

        else:
            virtual_tokens = None

        encoder_output = self.encoder[gp_idx](
            emb_pad,
            attn_mask=attn_mask,
            gene_labels=tokens_pad,
            masking=False,
            return_attention=True,
            num_virtual_tokens=self.num_virtual_tokens * 2,
            using_gp_specific_token=self.num_virtual_tokens > 0,
        )

        attn = encoder_output['attention']

        # Average attention across heads
        attn = attn.mean(dim=1)

        output = {}

        # Select cls attention
        attn = attn[:, 0, :]

        output['cls'] = attn[:, 0].cpu().detach().numpy()

        if self.num_virtual_tokens > 0:
            for i in range(self.num_virtual_tokens):
                output[f'gp_virtual_token_{i}'] = (
                    attn[:, -2 * (i + 1)].cpu().detach().numpy()
                )
                output[f'shared_virtual_token_{i}'] = (
                    attn[:, -1 * (i + 1)].cpu().detach().numpy()
                )

            # drop virtual tokens
            attn = attn[:, : -2 * self.num_virtual_tokens]

        # keep only gene scores
        attn = attn[:, 1:]

        for i, gene in enumerate(gp_tokens):
            # Ensure the data types match
            gene = gene.to(tokens_pad_unencoded.dtype)

            # zero out other genes
            mask = (tokens_pad_unencoded == gene).to(torch.int)

            masked_score = attn * mask

            # Find the indices of the non-zero scores
            non_zero_mask = masked_score != 0

            indices = non_zero_mask.nonzero(as_tuple=True)

            # Initialize the result tensor with zeros
            result = torch.zeros(attn.shape[0], attn.shape[-1]).to(attn.device)

            # Check if there are any non-zero rows, and update the result tensor
            if indices[0].nelement() != 0:
                result[indices] = masked_score[indices]

                # for debugging
                for row_idx in torch.unique(indices[0]):
                    if non_zero_mask[row_idx].sum() > 1:
                        raise ValueError('Multiple non-zero scores for the same gene')

            output[gene_names[i]] = result.cpu().detach().numpy().sum(axis=-1)

        return output


class cellWrapperWithPrompt(cellWrapper):
    def __init__(self, num_virtual_tokens, **kwargs):
        super().__init__(**kwargs)

        self.num_virtual_tokens = num_virtual_tokens

        self.prompt_encoder = PromptEncoder(
            token_dim=self.gp_latent_size,
            encoder_hidden_size=int(self.gp_latent_size * 0.5),
            num_virtual_tokens=self.num_virtual_tokens,
        )

    def forward(self, x, masking):
        """
        Input is the dictionary output of gpWrapper
        we need keys z and number of genes per cell
        """
        z, gp_labels, attn_mask = self.build_input_matrix(
            z=x['z'], num_genes_per_cell_list=x['num_genes_per_cell_list']
        )

        # Optionally append virtual tokens
        if self.num_virtual_tokens > 0:
            virtual_tokens = x['virtual_tokens']

            z = torch.cat([virtual_tokens, z], dim=1)

            # Add to labels as well
            gp_labels = torch.cat(
                [
                    torch.tensor(
                        [-100] * self.num_virtual_tokens, device=gp_labels.device
                    )
                    .unsqueeze(0)
                    .expand(gp_labels.shape[0], -1),
                    gp_labels,
                ],
                dim=1,
            )

            # And attention mask
            attn_mask = torch.cat(
                [
                    attn_mask,
                    torch.ones(attn_mask.shape[0], self.num_virtual_tokens).to(
                        attn_mask.device
                    ),
                ],
                dim=1,
            )

        encoder_output = self.encoder(
            z,
            gene_labels=gp_labels,
            attn_mask=attn_mask,
            masking=masking,
            return_attention=False,
            num_virtual_tokens=self.num_virtual_tokens,
        )

        output = {
            'cell_token': encoder_output['cls'],
            'gp_logits_lm': encoder_output['logits_lm'],
            'gp_labels': encoder_output['gene_labels'],
        }

        if self.num_virtual_tokens > 0:
            output['virtual_token'] = encoder_output['shared_virtual_tokens']

        return output

    def get_attn(self, x):
        """
        Input is the dictionary output of gpWrapper
        we need keys z and number of genes per cell
        """

        z, gp_labels, attn_mask = self.build_input_matrix(
            z=x['z'], num_genes_per_cell_list=x['num_genes_per_cell_list']
        )

        # Append virtual tokens
        virtual_tokens = x['virtual_tokens']

        z = torch.cat([virtual_tokens, z], dim=1)

        # Add to labels as well
        gp_labels = torch.cat(
            [
                torch.tensor([-100] * self.num_virtual_tokens, device=gp_labels.device)
                .unsqueeze(0)
                .expand(gp_labels.shape[0], -1),
                gp_labels,
            ],
            dim=1,
        )

        # And attention mask
        attn_mask = torch.cat(
            [
                attn_mask,
                torch.ones(attn_mask.shape[0], self.num_virtual_tokens).to(
                    attn_mask.device
                ),
            ],
            dim=1,
        )

        encoder_output = self.encoder(
            z,
            gene_labels=gp_labels,
            attn_mask=attn_mask,
            masking=False,
            return_attention=True,
            num_virtual_tokens=self.num_virtual_tokens,
        )

        # Reorder attention matrix so GP are in the same order in each cell
        attn = encoder_output['attention']

        # Average attention across heads
        attn = attn.mean(dim=1)

        # And focus on cls attention scores
        attn = attn[:, 0, :]

        output = {}

        output['cls'] = attn[:, 0].cpu().detach().numpy()

        # drop cls token
        attn = attn[:, 1:]

        if self.num_virtual_tokens > 0:
            for i in range(self.num_virtual_tokens):
                output[f'virtual_token_{i}'] = (
                    attn[:, -1 * (i + 1)].cpu().detach().numpy()
                )

            # drop virtual tokens
            attn = attn[:, : -self.num_virtual_tokens]
            gp_labels = gp_labels[:, : -self.num_virtual_tokens]

        for i, gp in enumerate(self.gp_inputs):
            # zero out other GP
            mask = (gp_labels == i).to(torch.int)

            masked_score = attn * mask

            # Find the indices of the non-zero scores
            non_zero_mask = masked_score != 0

            indices = non_zero_mask.nonzero(as_tuple=True)

            # Initialize the result tensor with zeros
            result = torch.zeros(attn.shape[0], attn.shape[-1]).to(attn.device)

            # Check if there are any non-zero rows, and update the result tensor
            if indices[0].nelement() != 0:
                result[indices] = masked_score[indices]

                # for debugging
                for row_idx in torch.unique(indices[0]):
                    if non_zero_mask[row_idx].sum() > 1:
                        raise ValueError('Multiple non-zero scores for the same gene')

            output[gp] = result.cpu().detach().numpy().sum(axis=-1)

        return output


class gpTransformerBaseWithPrompt(gpTransformerBase):
    def __init__(
        self,
        num_virtual_tokens,
        virtual_tokens_label,
        num_prompt_classes,
        # for gpWrapper
        gene_token_path=TOKEN_DICTIONARY_FILE,
        gene_name_path=ENSEMBL_DICTIONARY_FILE,
        num_heads=1,
        model_type='Base',
        learn_new_gp=False,
        use_pos_emb='sin_cos',
        **kwargs,
    ):
        super().__init__(
            gene_token_path=TOKEN_DICTIONARY_FILE,
            gene_name_path=ENSEMBL_DICTIONARY_FILE,
            num_heads=1,
            model_type='Base',
            learn_new_gp=False,
            use_pos_emb=use_pos_emb,
            **kwargs,
        )

        self.multi_gp_encoder = gpWrapperWithPrompt(
            database=self.gpdb,
            do_ensembl_conversion=self.do_ensembl_conversion,
            gene_token_path=gene_token_path,
            gene_name_path=gene_name_path,
            gp_latent_size=self.gp_latent_size,
            n_blocks=self.n_blocks,
            num_heads=num_heads,
            mgm_mask_ratio=self.mgm_mask_ratio,
            gp_inputs=self.gp_inputs,
            use_flash=self.use_flash,
            model_type=model_type,
            learn_new_gp=learn_new_gp,
            use_pos_emb=use_pos_emb,
            # Prompt-spwcific arguments
            num_virtual_tokens=num_virtual_tokens,
            virtual_tokens_label=virtual_tokens_label,
            num_prompt_classes=num_prompt_classes,
        )


class gpTransformerGlobalWithPrompt(gpTransformerBaseWithPrompt, gpTransformerGlobal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# ----------------------------------
# Prototypes
# ----------------------------------


class gpTransformerPrototypes(gpTransformerGlobal):
    def __init__(self, num_prototypes, **kwargs):
        super().__init__(**kwargs)

        self.num_prototypes = num_prototypes

        self.prototypes = nn.Parameter(
            torch.empty(
                self.num_prototypes,
                self.gp_latent_size,
            ),
            requires_grad=True,
        )

        nn.init.xavier_normal_(self.prototypes)


####################################
# Embedding evaluation
####################################


class EmbEvaluatorHead(nn.Module):
    '''
    Evaluate embeddings by training a classifier
    '''

    def __init__(
        self,
        emb_dim: int,
        n_classes: int,
        num_condition_cat: int = 0,
    ):
        super().__init__()

        if num_condition_cat > 0:
            self.clf_head = nn.Sequential(
                nn.Linear(emb_dim + num_condition_cat, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, n_classes),
            )

        else:
            self.clf_head = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        return self.clf_head(x)


if __name__ == '__main__':
    pass
