####################################
# Load packages
####################################

import os
import pickle
from pathlib import Path
from typing import Dict, Optional

# imports
import numpy as np
import torch
import torch.nn as nn
from peft import PeftConfig, get_peft_model
from transformers import BertConfig, BertForMaskedLM

from .. import ENSEMBL_DICTIONARY_FILE, TOKEN_DICTIONARY_FILE
from ..Modules.modules import Mlp, gpTransformerEncoder
from ..Utils.geneformer_utils import EmbExtractor, get_gf_repo
from ..Utils.utils import (
    bin_gene_expression,
    build_gp_input_matrix,
    get_gp_tokens,
)

####################################
# Geneformer
####################################


class gfWrapper(nn.Module):
    """Wrapper for Geneformer model to extract gene embeddings.

    Loads a pretrained Geneformer (BERT-based) model and optionally applies
    PEFT (Parameter-Efficient Fine-Tuning) configuration. Freezes model
    weights and extracts embeddings from specified layer.

    Parameters
    ----------
    geneformer_model : str
        Path to pretrained Geneformer model.
    fm_layer_to_quant : int
        Layer index to extract embeddings from (negative indexing supported).
    peft_config_path : str or None
        Path to PEFT configuration checkpoint, or None for base model.
    token_dictionary_file : str, optional
        Path to token dictionary file (default: TOKEN_DICTIONARY_FILE).

    Attributes
    ----------
    gf : BertForMaskedLM or PeftModel
        The Geneformer model with frozen weights.
    gf_emb_extractor : EmbExtractor
        Embedding extractor utility.
    """

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
    """Wrapper for gene-level encoder with transformer architecture.

    Builds gene embeddings lookup table and transformer encoder for learning
    gene-level representations. Supports initialization from Geneformer
    embeddings or training from scratch (random initialization of
    embeddings). In both cases, the transformer encoder is trained from scratch.

    Parameters
    ----------
    all_genes : list or dict
        Genes to include. If dict, expects 'hvg' and 'gp_genes' keys.
    do_ensembl_conversion : bool
        Whether to convert gene names to Ensembl IDs.
    gene_name_path : str
        Path to gene name to Ensembl ID dictionary.
    gene_token_path : str
        Path to gene token dictionary.
    config_dict : dict
        Model configuration dictionary with keys like 'hidden_size',
        'num_hidden_layers', 'num_attention_heads', etc.
    gp_latent_size : int
        Dimension of gene program latent representations.
    use_gene_embeddings : str or None, optional
        Model name (e.g., 'gf-12L-95M-i4096') or path to embeddings file
        This should be a file containing a tensor of shape (vocab_size, embedding_dim)
            with the same vocab size and embedding dimension as specified in config_dict.
        If a string is provided, it will be interpreted as a model name or path to load embeddings from.
        If False, initializes gene embeddings randomly and trains from scratch.
        (.pt or .npy) (default: None).
    attn_dropout : float, optional
        Attention dropout rate (default: 0.0).
    init_sparsity : float, optional
        fraction of model weights to initialize to 0.

    Attributes
    ----------
    gene_embeddings : nn.Embedding
        Gene embedding lookup table.
    gene_tokens : torch.Tensor
        Buffer containing gene tokens.
    gene_tokens_lookup : torch.Tensor
        Buffer for mapping tokens to indices.
    model : gpTransformerEncoder
        Transformer encoder for gene representations.
    max_seq_len : int
        Maximum sequence length.
    """

    def __init__(
        self,
        all_genes,
        do_ensembl_conversion,
        gene_name_path,
        gene_token_path,
        config_dict,
        gp_latent_size,
        use_gene_embeddings=None,
        attn_dropout=0.0,
        init_sparsity=0.0,
    ):
        super().__init__()

        # Intiialize lookup table for vocab
        if isinstance(use_gene_embeddings, str):
            if use_gene_embeddings == 'gf-12L-95M-i4096':
                emb_path = (
                    Path(__file__).parent.parent
                    / 'Utils/gf-12L-95M-i4096_word_embeddings_may2025.pt'
                )
                emb = torch.load(emb_path)
            elif use_gene_embeddings.endswith('.pt'):
                emb = torch.load(use_gene_embeddings)
            elif use_gene_embeddings.endswith('.npy'):
                emb = torch.from_numpy(np.load(use_gene_embeddings))
            else:
                raise ValueError(
                    'Unsupported file format for gene embeddings. '
                    'Please store as .pt or .npy'
                )

            self.gene_embeddings = nn.Embedding.from_pretrained(
                emb,
                padding_idx=0,
            )

            if '16' in str(config_dict['torch_dtype']):
                self.gene_embeddings.half()

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
            attn_drop_rate=attn_dropout,
            sparsity=init_sparsity,
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
    """Wrapper for gene program (GP) encoders.

    Creates separate transformer encoders for each gene program, handling
    gene-to-token mapping, masked gene modeling, and attention extraction.

    Parameters
    ----------
    gp_inputs : list of str
        Names of gene programs to encode.
    database : pd.DataFrame
        Gene program database where each column is a GP.
    do_ensembl_conversion : bool
        Whether to convert gene names to Ensembl IDs.
    gene_token_path : str
        Path to gene token dictionary.
    gene_name_path : str
        Path to gene name to Ensembl ID dictionary.
    gp_latent_size : int
        Dimension of GP latent representations.
    n_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    mgm_mask_ratio : float
        Masked gene modeling mask ratio.
    use_flash : bool
        Whether to use flash attention.
    model_type : str
        Type of model architecture.
    learn_new_gp : bool
        Whether learning new gene programs.
    use_pos_emb : str or bool
        Type of positional embedding to use.
    fm_model_input_size : int
        Foundation model input size.
    use_l2_norm : bool
        Whether to use L2 normalization.
    attn_dropout : float
        Attention dropout rate.
    init_sparsity : float
        Initial sparsity level.

    Attributes
    ----------
    encoder : nn.ModuleList
        List of gpTransformerEncoder modules, one per GP.
    all_gp_tokens : set
        Set of all gene tokens across all GPs.
    gp{i}_tokens : torch.Tensor
        Registered buffer of tokens for i-th GP.
    gp{i}_tokens_lookup : torch.Tensor
        Registered buffer for token-to-index mapping for i-th GP.
    """

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
        attn_dropout,
        init_sparsity,
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

            gp_tokens_tensor = torch.tensor(sorted(list(gp_tokens)), dtype=torch.int32)

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
                    attn_drop_rate=attn_dropout,
                    sparsity=init_sparsity,
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
        init_sparsity=0.0,
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
        if len(gp_token_list) > 1:
            z = torch.stack(gp_token_list, dim=1)
        else:
            z = gp_token_list[0].unsqueeze(dim=1)

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
            output['z'] = z

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
            gf_emb['gene_emb'],  # geneformer embeddings
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
        """
        If multiple blocks, get attn matrix from last transformer block.
        Vectorized: LUT + scatter_add_ (with offset) to build (n_cells x n_genes).
        """
        gp_tokens = getattr(
            self, f'gp{gp_idx}_tokens'
        )  # list/1D tensor of gene token ids

        # Build batch inputs
        emb_pad, tokens_pad, _, attn_mask = build_gp_input_matrix(
            gf_emb['gene_emb'],
            input_dataset['input_ids'],
            gp_tokens,
        )

        # Keep unencoded tokens for mapping to genes (shape [B, L], excluding CLS col)
        tokens_pad_unencoded = tokens_pad
        if not torch.is_tensor(tokens_pad_unencoded):
            tokens_pad_unencoded = torch.as_tensor(tokens_pad_unencoded)

        # Encode tokens for MLM (unchanged)
        tokens_pad = getattr(self, f'gp{gp_idx}_tokens_lookup')[tokens_pad].long()

        # Forward to get attention
        encoder_output = self.encoder[gp_idx](
            emb_pad,
            attn_mask=attn_mask,
            gene_labels=tokens_pad,
            masking=False,
            return_attention=True,
        )

        # attention: [B, H, S, S] -> mean heads -> take CLS row -> [B, S]
        attn = encoder_output['attention'].mean(dim=1)  # [B, S, S]
        attn_cls_all = attn[:, 0, :]  # [B, S]
        cls_scores = attn_cls_all[:, 0]  # [B]
        attn_to_tokens = attn_cls_all[:, 1:]  # [B, L], L = S-1

        # Token ids for those L positions
        toks = tokens_pad_unencoded  # [B, L]
        if toks.dtype != torch.long:
            toks = toks.long()

        device = toks.device
        dtype_scores = attn_to_tokens.dtype

        # gp_tokens -> tensor on device
        gp_tokens_t = torch.as_tensor(
            gp_tokens, device=device, dtype=toks.dtype
        ).reshape(
            -1
        )  # [G]
        G = gp_tokens_t.numel()
        B = attn_to_tokens.size(0)

        # --- LUT with offset over observed token ids in this batch ---
        vmin = int(toks.min().item())
        vmax = int(toks.max().item())
        R = vmax - vmin + 1
        offset = -vmin

        # Edge case: if R <= 0 (shouldn't happen), skip safely
        if R <= 0:
            result = torch.zeros(B, G, device=device, dtype=dtype_scores)
        else:
            lut = torch.full((R,), -1, device=device, dtype=torch.long)

            shifted_gp = gp_tokens_t + offset  # may fall outside [0, R)
            in_range = (shifted_gp >= 0) & (shifted_gp < R)
            if in_range.any():
                # map only in-range gp tokens to their column indices
                gp_cols = torch.arange(G, device=device, dtype=torch.long)
                lut[shifted_gp[in_range]] = gp_cols[in_range]

            # Map each token position to a gene column (or -1 if not in gp_tokens)
            gene_idx = lut[toks + offset]  # [B, L] in {-1, 0..G-1}
            valid = gene_idx >= 0  # [B, L] bool

            # Scatter-add attention per (cell, gene)
            result = torch.zeros(B, G, device=device, dtype=dtype_scores)
            src = attn_to_tokens * valid.to(dtype_scores)  # zero-out non-gp tokens
            idx = gene_idx.clamp(min=0)  # safe index; src==0 where invalid
            result.scatter_add_(dim=1, index=idx, src=src)  # (B, G)

        # --- Build output dict; move to CPU once ---
        out = {'cls': cls_scores.detach().cpu().numpy()}
        res_np = result.detach().cpu().numpy()  # (B, G)
        for j, tok in enumerate(gp_tokens_t.tolist()):
            out[tok] = res_np[:, j]

        return out


class cellWrapper(nn.Module):
    """Wrapper for cell-level encoder using GP representations.

    Takes gene program embeddings and learns a unified cell-level
    representation through a transformer encoder. Handles reordering of GPs
    by gene coverage and masking.

    Parameters
    ----------
    gp_inputs : list of str
        Names of gene programs.
    n_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    gp_latent_size : int
        Dimension of GP latent representations.
    global_masking_rate : float
        Masking rate for global (cell-level) masking.
    use_flash : bool
        Whether to use flash attention.
    use_l2_norm : bool
        Whether to use L2 normalization.
    global_pos_emb : str or bool
        Type of positional embedding to use.
    global_attn_dropout : float
        Attention dropout rate.

    Attributes
    ----------
    encoder : gpTransformerEncoder
        Transformer encoder for cell representations.
    gp_inputs : list of str
        Stored GP names.
    gp_latent_size : int
        Stored latent dimension.
    """

    def __init__(
        self,
        gp_inputs,
        n_blocks,
        num_heads,
        gp_latent_size,
        global_masking_rate,
        use_flash,
        use_l2_norm,
        global_pos_emb,
        global_attn_dropout,
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
            use_pos_emb=global_pos_emb,
            attn_drop_rate=global_attn_dropout,
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
    """Prediction head for gene expression count reconstruction.

    Supports multiple loss modes for count prediction: MSE with ReLU output,
    Negative Binomial (NB), or Zero-Inflated Negative Binomial (ZINB).

    Parameters
    ----------
    loss_mode : {'mse', 'nb', 'zinb'}, optional
        Loss function mode (default: 'mse').
    n_genes : int, optional
        Number of genes to predict (default: 25426).
    d_model : int, optional
        Input embedding dimension (default: 512).

    Attributes
    ----------
    loss_mode : str
        Stored loss mode.
    mlp : Mlp
        MLP layer for feature transformation.
    relu_output : nn.Sequential, optional
        ReLU output layer for MSE mode.
    linear_output : nn.Linear, optional
        Linear output for ZINB dropout logits.
    softmax_output : nn.Sequential, optional
        Softmax output for NB/ZINB mean parameters.
    """

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


####################################
# Define model
####################################


class gpTransformerBase(nn.Module):
    """Base model for learning gene program (GP) latent representations.

    Combines a foundation model encoder (e.g., Geneformer) with GP-specific
    transformer encoders to learn hierarchical gene program representations.
    Supports training from pretrained embeddings or from scratch.

    Parameters
    ----------
    database : pd.DataFrame
        Gene program database with GP names as columns and genes as rows.
    attn_dropout : float, optional
        Attention dropout rate (default: 0).
    gp_inputs : list of str or None, optional
        Gene programs to include. If None, uses all GPs in database
        (default: None).
    do_ensembl_conversion : bool, optional
        Whether to convert gene names to Ensembl IDs (default: True).
    num_heads : int, optional
        Number of attention heads (default: 1).
    n_blocks : int, optional
        Number of transformer blocks (default: 1).
    mgm_mask_ratio : float, optional
        Masked gene modeling mask ratio (default: 0.5).
    use_flash : bool, optional
        Whether to use flash attention (default: False).
    fm_encoder_pkg : str, optional
        Foundation model package ('geneformer', 'geneformer_2021',
        'from_scratch') (default: 'geneformer').
    fm_encoder_name : str, optional
        Foundation model name/path (default: 'gf-6L-30M-i2048').
    peft_config_path : str or None, optional
        Path to PEFT configuration (default: None).
    fm_layer_to_quant : int, optional
        Foundation model layer to extract embeddings from
        (default: -1, penultimate layer).
    model_type : str, optional
        Model type identifier (default: 'Base').
    learn_new_gp : bool, optional
        Whether learning new gene programs (default: False).
    gp_of_interest : str, list, or None, optional
        Specific GP(s) to focus on (default: None).
    use_pos_emb : str or bool, optional
        Type of positional embedding ('sin_cos', etc.) (default: 'sin_cos').
    vocab_gene_names : list or None, optional
        Vocabulary gene names for one-hot wrapper (default: None).
    bert_config : dict or None, optional
        BERT configuration dictionary for from-scratch training
        (default: None).
    gp_latent_size : int or None, optional
        Dimension of GP latent representations. If None, inferred from
        foundation model (default: None).
    use_gene_embeddings : str or bool, optional
        Model name (e.g., 'gf-12L-95M-i4096') or path to embeddings file,
        or False to train from scratch (default: False).
    use_l2_norm : bool, optional
        Whether to use L2 normalization (default: False).
    all_genes : list, dict, or None, optional
        All genes to include in model (default: None).
    warmup : int, optional
        Number of warmup epochs (default: 0).
    init_sparsity : float, optional
        Initial sparsity level (default: 0.0).

    Attributes
    ----------
    gf_wrapper : gfWrapper, or GeneWrapper
        Foundation model wrapper for extracting gene embeddings.
    multi_gp_encoder : gpWrapper
        Multi-GP encoder for learning GP representations.
    gf_cell_encoder : nn.Module
        Cell-level encoder (identity by default).
    gpdb : pd.DataFrame
        Stored gene program database subset.
    gp_inputs : list of str
        Stored list of GP names.
    gp_latent_size : int
        Dimension of GP representations.
    fm_model_input_size : int
        Foundation model input size.
    gene_token_path : str
        Path to gene token dictionary.
    gene_name_path : str
        Path to gene name dictionary.
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
        vocab_gene_names=None,
        bert_config=None,
        gp_latent_size=None,
        use_gene_embeddings=False,
        use_l2_norm=False,
        all_genes=None,
        warmup=0,
        init_sparsity=0.0,
    ):
        super().__init__()

        self.fm_encoder_pkg = fm_encoder_pkg
        self.fm_encoder_name = fm_encoder_name
        self.use_l2_norm = use_l2_norm
        self.warmup = warmup

        if fm_encoder_pkg == 'geneformer':
            geneformer_repo_path = get_gf_repo()

            # Initialize geneformer model for getting geneformer embeddings
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
                use_gene_embeddings=use_gene_embeddings,
                gp_latent_size=gp_latent_size,
                attn_dropout=attn_dropout,
                init_sparsity=init_sparsity,
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
            attn_dropout=attn_dropout,
            init_sparsity=init_sparsity,
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
        return_mean_non_padding=False,
    ):
        # input is tokenized dataset
        emb_out = self.gf_wrapper(
            input_dataset,
            masking=masking,
            return_mean_non_padding=return_mean_non_padding,
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
            return_mean_non_padding=return_mean_non_padding,
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

    def get_gene_gene_attn(self, input_dataset, gp_idx, masking):
        gf_emb = self.gf_wrapper(input_dataset, masking=masking)

        output = self.multi_gp_encoder.get_gene_gene_attn(
            gf_emb, input_dataset, gp_idx=gp_idx
        )

        return output

    def get_cls_attn(self, input_dataset, gp, masking):
        # Get gp index
        gp_idx = self.gp_inputs.index(gp)

        # Get Geneformer embeddings
        gf_emb = self.gf_wrapper(input_dataset, masking=masking)

        output = self.multi_gp_encoder.get_cls_attn(
            gf_emb, input_dataset, gp_idx=gp_idx
        )

        return output


class gpTransformerGlobal(gpTransformerBase):
    """Global model learning GP representations and unified cell token.

    Extends gpTransformerBase by adding a cell-level encoder that combines
    individual GP representations into a global cell token. Supports multiple
    training objectives: supervised classification, masked modeling, or
    count reconstruction.

    Parameters
    ----------
    global_attn_heads : int, optional
        Number of attention heads for cell-level encoder (default: 8).
    global_loss : {'reconstruction', 'supervised', 'masking'}, optional
        Global loss function type (default: 'reconstruction').
    total_n_genes : int, optional
        Total number of genes for reconstruction (default: 25426).
    reconstruction_loss : {'nb', 'zinb', 'mse', 'binning'}, optional
        Type of reconstruction loss (default: 'nb').
    supervised_labels : dict or None, optional
        Dictionary mapping task names to number of classes for supervised
        learning (default: None).
    global_masking_rate : float, optional
        Masking rate for global masking objective (default: 0).
    global_n_blocks : int, optional
        Number of transformer blocks in cell encoder (default: 1).
    use_flash : bool, optional
        Whether to use flash attention (default: False).
    use_l2_norm : bool, optional
        Whether to use L2 normalization (default: False).
    n_bins : int, optional
        Number of bins for binning reconstruction loss (default: 10).
    global_pos_emb : str or bool, optional
        Type of positional embedding for cell encoder
        (default: 'sin_cos').
    global_attn_dropout : float, optional
        Attention dropout for cell encoder (default: 0.0).
    **kwargs
        Additional arguments passed to gpTransformerBase.

    Attributes
    ----------
    cell_token_learner : cellWrapper or None
        Cell-level encoder for learning unified cell representations.
    global_loss : str
        Stored global loss type.
    global_attn_heads : int or None
        Number of attention heads.
    clf_head : nn.ModuleList, optional
        Classification heads for supervised tasks.
    supervised_tasks : dict, optional
        Mapping of task names to indices.
    count_head :
        Prediction head for count reconstruction.
    reconstruction_loss : str, optional
        Type of reconstruction loss (mse, nb, zinb)
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
        global_pos_emb='sin_cos',
        global_attn_dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            use_flash=use_flash,
            model_type='Global',
            use_l2_norm=use_l2_norm,
            **kwargs,
        )
        self.global_loss = global_loss

        self.global_attn_heads = global_attn_heads
        self.cell_token_learner = cellWrapper(
            gp_inputs=self.gp_inputs,
            gp_latent_size=self.gp_latent_size,
            n_blocks=global_n_blocks,
            num_heads=self.global_attn_heads,
            global_masking_rate=global_masking_rate,
            use_flash=use_flash,
            use_l2_norm=use_l2_norm,
            global_pos_emb=global_pos_emb,
            global_attn_dropout=global_attn_dropout,
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

            self.count_head = CountHead(
                loss_mode=reconstruction_loss,
                n_genes=total_n_genes,
                d_model=self.gp_latent_size,
            )

    def base_output_to_cell_output(
        self,
        base_output: Dict[str, torch.Tensor],
        masking_global: bool = False,
    ):
        cell_output = self.cell_token_learner(base_output, masking=masking_global)
        return cell_output

    def forward(
        self,
        input_dataset,
        return_gene_embeddings=False,
        return_attention=False,
        tokens_to_keep=None,
        gp_of_interest=None,
        masking=False,
        masking_global=False,
        epoch=None,
        return_mean_non_padding=False,
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
            epoch=epoch,  # hard set to 'Global' in trainer
        )

        if return_gene_embeddings:
            return base_output

        cell_output = self.base_output_to_cell_output(
            base_output,
            masking_global=masking_global,
        )

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


if __name__ == '__main__':
    pass
