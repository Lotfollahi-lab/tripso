####################################
# Load packages
####################################

import pickle
import warnings
from typing import Dict, Optional

# imports
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from peft import PeftConfig, get_peft_model
from transformers import BertForMaskedLM

from ..Modules.modules import (
    Mlp,
    PretrainedEmbeddings,
    PromptEncoder,
    gpTransformerEncoder,
)
from ..Utils.geneformer_utils import EmbExtractor
from ..Utils.utils import bin_gene_expression, get_gp_tokens

####################################
# Geneformer
####################################

GENE_NAME_FILE = '/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/Geneformer/geneformer/gene_name_id_dict.pkl'  # noqa
GENEFORMER_MODEL_PATH = (
    '/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/Geneformer/'
)


class BaselineWrapper(nn.Module):
    def __init__(
        self,
        vocab_size,
        gene2vec,
    ):
        super().__init__()

        if gene2vec is None:
            self.gene_embeddings = nn.Embedding(vocab_size, 256)
        else:
            # Load gene embeddings gene2vec
            # nn.Embedding from pretrained automatically turns off gradient
            gene2vec = pd.read_csv(gene2vec, index_col=0)
            gene2vec_tensor = torch.tensor(gene2vec.values, dtype=torch.float32)
            self.gene_embeddings = nn.Embedding.from_pretrained(gene2vec_tensor)
            print('Loaded gene2vec embeddings', gene2vec_tensor.shape)

    def forward(self, input_dataset, inference):
        # input is tokenized dataset
        # print(input_dataset['input_ids'][0])
        # print('maximum input id', input_dataset['input_ids'].max())
        # print('min input id', input_dataset['input_ids'].min())
        # print('Embedding shape', self.gene_embeddings.weight.shape)

        emb_out = self.gene_embeddings(input_dataset['input_ids'])

        return emb_out


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
        gf_layer_to_quant,
        peft_config_path,
    ):
        super().__init__()

        # Initialize geneformer model for getting geneformer embeddings
        model = BertForMaskedLM.from_pretrained(
            geneformer_model, output_attentions=False, output_hidden_states=True
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

        self.gf_emb_extractor = EmbExtractor(emb_layer=gf_layer_to_quant)

    def forward(self, input_dataset, inference):
        # input is tokenized dataset

        emb_out = self.gf_emb_extractor.extract_embs(
            model=self.gf, input_data=input_dataset, inference=inference
        )

        return emb_out


####################################
# GP wrapper
####################################


class gpWrapper(nn.Module):
    def __init__(
        self,
        gp_inputs,
        database,
        do_ensembl_conversion,
        gene_counts_df,
        gene_token_path,
        gene_name_path,
        gp_latent_size,
        n_blocks,
        num_heads,
        mgm_mask_ratio,
        add_remaining_var,
        use_flash,
        model_type,
        learn_new_gp,
        hvg_list,
        num_virtual_tokens,
        virtual_tokens_label,
        num_prompt_classes,
        mean_emb_dict,
        use_pos_emb,
    ):
        super().__init__()

        self.gp_latent_size = gp_latent_size
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.mgm_mask_ratio = mgm_mask_ratio
        self.gp_inputs = gp_inputs
        self.model_type = model_type
        self.learning_new_gp = learn_new_gp
        self.num_virtual_tokens = num_virtual_tokens

        # Get vocab size
        with open(gene_token_path, 'rb') as f:
            token_dict = pickle.load(f)
        self.vocab_size = max(token_dict.values())

        # Store all genes included in at least one GP
        self.all_gp_tokens = set()

        # Reset 'remaining var' --> will be added back in next step if needed
        if 'remaining_var' in self.gp_inputs:
            self.gp_inputs.remove('remaining_var')

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
            # +2 because in geneformer 0 --> padding and 1 --> mask
            lookup_tensor = torch.full((self.vocab_size + 2,), -100, dtype=torch.int32)
            # Create a tensor of indices corresponding to positions in gp_tokens
            indices = torch.arange(gp_tokens_tensor.shape[0], dtype=torch.int32)

            # Use tensor indexing to assign values
            lookup_tensor[gp_tokens_tensor.long()] = indices
            self.register_buffer(f'gp{i}_tokens_lookup', lookup_tensor)

            if self.num_virtual_tokens > 0:
                prompt_encoder = PromptEncoder(
                    token_dim=self.gp_latent_size,
                    encoder_hidden_size=int(self.gp_latent_size * 0.5),
                    num_virtual_tokens=self.num_virtual_tokens,
                )

                setattr(self, f'prompt_encoder_gp{i}', prompt_encoder)

        self.encoder = nn.ModuleList(
            [
                gpTransformerEncoder(
                    n_gp_tokens=len(getattr(self, f'gp{i}_tokens')),
                    embed_dim=self.gp_latent_size,
                    depth=self.n_blocks,
                    num_heads=self.num_heads,
                    mlm_masking_prob=self.mgm_mask_ratio,
                    use_flash=use_flash,
                    use_pos_emb=use_pos_emb,
                )
                for i in range(len(gp_inputs))
            ]
        )

        self.add_remaining_var = add_remaining_var

        if self.add_remaining_var is not None:
            n_gp = len(self.gp_inputs)
            self.gp_inputs = gp_inputs + ['remaining_var']

            # find non GP genes
            if gene_counts_df is None:
                raise ValueError(
                    'Please provide a dataframe '
                    'with counts of gene occurrences in dataset'
                )

            non_gp_tokens = set(gene_counts_df['token'].tolist()) - self.all_gp_tokens

            if hvg_list is not None:
                # convert to tokens
                with open(gene_token_path, 'rb') as f:
                    token_dict = pickle.load(f)

                with open(gene_name_path, 'rb') as f:
                    gene_name_dict = pickle.load(f)

                if do_ensembl_conversion:
                    hvg_list = [
                        gene_name_dict[x] for x in hvg_list if x in gene_name_dict
                    ]

                hvg_list = [token_dict[x] for x in hvg_list if x in token_dict]

                non_gp_tokens = non_gp_tokens.intersection(set(hvg_list))

            tokens_tensor = torch.tensor(list(non_gp_tokens), dtype=torch.int32)

            self.register_buffer(f'gp{n_gp}_tokens', tokens_tensor)

            # Set up look up tensor
            # for converting gene tokens to encoded values inside transformer block
            # +2 because in geneformer 0 --> padding and 1 --> mask
            lookup_tensor = torch.full((self.vocab_size + 2,), -100, dtype=torch.int32)
            # Create a tensor of indices corresponding to positions in gp_tokens
            indices = torch.arange(tokens_tensor.shape[0], dtype=torch.int32)

            # Use tensor indexing to assign values
            lookup_tensor[tokens_tensor.long()] = indices
            self.register_buffer(f'gp{n_gp}_tokens_lookup', lookup_tensor)

            self.encoder.append(
                gpTransformerEncoder(
                    n_gp_tokens=100
                    if self.add_remaining_var == 'top100'
                    else len(non_gp_tokens),  # we only keep top 100 genes
                    vocab_size=len(non_gp_tokens),
                    embed_dim=self.gp_latent_size,
                    depth=self.n_blocks,
                    num_heads=num_heads,
                    mlm_masking_prob=self.mgm_mask_ratio,
                    use_flash=(self.add_remaining_var == 'allgenes'),
                )
            )

        if self.num_virtual_tokens > 0:
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

        self.mean_emb_dict = mean_emb_dict
        if self.mean_emb_dict is not None:
            # Load precomputed GP gene embeddings
            # Load the embeddings from the pickle file
            with open(mean_emb_dict, 'rb') as f:
                z_mean = pickle.load(f)

            # Convert strings to integers
            z_mean = {int(k): v for k, v in z_mean.items()}

            # Convert z_mean dictionary to a tensor for efficient indexing
            max_token_id = max(z_mean.keys())
            z_mean_tensor = torch.zeros(
                (max_token_id + 1, next(iter(z_mean.values())).shape[0])
            )

            for token_id, embedding in z_mean.items():
                z_mean_tensor[token_id] = torch.tensor(embedding)

            self.register_buffer('z_mean', z_mean_tensor)

    def build_input_matrix(
        self, gf, input_ids, gp_tokens, crop_to_gp_len=True, is_gpfinder=False
    ):
        """
        Build a matrix of shape (n_cells, n_gp_tokens, 256)
        where (i, j, :) = 0 if gene j in cell i does not belong to the current GP
        maintains geneformer order

        Inputs:

        gf :
            geneformer embeddings (n_cells, 2048, 256)

        input_ids:
            list of lists with positional information for each token

        gp_tokens_list:
            list of tokens for each gene program

        model:
            "full_model" : set for input into geneformer
            "extract_genes" : when extracting gene embeddings
                            -> max size is total GP size
            # NEED TO REIMPLEMENT

        """
        # Get list of gp tokens
        # convert gp_tokens bf16 tensor to integers
        # gp_tokens = gp_tokens.to(torch.int)
        gp_tokens = gp_tokens.long()

        # FOR ONE HOT ENCODER VERSION ONLY
        b1, s1 = input_ids.shape
        b2, s2, e2 = gf.shape

        if s1 != s2:
            input_ids = input_ids[:, :s2]

        # Create a binary mask (h, i, k)
        # In cell h, is the gene at position i in our GP at position k?
        # Using broadcasting to compare tokens_arr with gp_tokens
        mask = input_ids.unsqueeze(2) == gp_tokens.unsqueeze(0)  # .unsqueeze(0)
        mask = mask.to(torch.int)

        # Now reshape so that we will zero out non GP genes in each cell
        # Sum along the last dimension to count how many GP tokens each gene matches
        mask_expanded = mask.sum(dim=-1).unsqueeze(2)

        if crop_to_gp_len or (
            is_gpfinder is not None
            and is_gpfinder
            and self.add_remaining_var == 'top100'
        ):
            # Apply the mask to the data using broadcasting
            masked_latent = gf * mask_expanded

            # Now wrangle so that the non zero genes are first
            # but we maintain the order
            # loop through the cells to deal with different shapes
            holder = []

            for i in range(masked_latent.shape[0]):
                x = masked_latent[i, :, :]
                c = masked_latent[i, :, 1]  # find which genes have been 0'd out
                idx = c != 0
                idx_zero = c == 0
                z = torch.concat((x[idx, :], x[idx_zero, :]), dim=0)
                holder += [z]

            result_matrix = torch.stack(holder)

            masked_labels = torch.where(
                mask.sum(axis=-1) == 0, torch.zeros_like(input_ids), input_ids
            )

            holder = []
            for i in range(masked_labels.shape[0]):
                x = masked_labels[i, :]
                nz = x != 0
                z = torch.concat((x[nz], x[~nz]), dim=0)
                holder += [z]

            masked_labels_output = torch.stack(holder)

            # crop
            if (
                is_gpfinder is not None
                and is_gpfinder
                and self.add_remaining_var == 'top100'
            ):
                n_genes_to_keep = 100
            else:
                n_genes_to_keep = gp_tokens.shape[0]
            result_matrix = result_matrix[:, :n_genes_to_keep, :]
            masked_labels_output = masked_labels_output[:, :n_genes_to_keep]

        else:
            # Apply the mask to the data using broadcasting
            result_matrix = gf * mask_expanded

            # Now do the same for labels
            # masked_labels_output = mask.sum(axis=-1) * input_ids
            masked_labels_output = torch.where(
                mask.sum(axis=-1) == 0, torch.zeros_like(input_ids), input_ids
            )

        # count number of genes per cell
        num_genes_per_cell = mask.sum(axis=-1).sum(axis=-1)

        # Make tensor for forward pass
        # comment the line below to leave 0s because they are actually informative
        # (this gene was not in the top 1000 of this cell)
        # masked_labels_output[masked_labels_output == 0] = -100
        # happens when do LOOKUP

        # Set up attention mask
        # to avoid attention to padding tokens
        attn_mask = torch.zeros_like(masked_labels_output)
        attn_mask[masked_labels_output != 0] = 1

        # never mask cls
        attn_mask = torch.cat(
            [torch.ones_like(attn_mask)[:, 0].unsqueeze(-1), attn_mask], dim=-1
        )

        # never mask prompt tokens
        if self.num_virtual_tokens > 0:
            attn_mask = torch.cat(
                # times 2 because 1 GP-specific token + 1 non-specific token
                [
                    attn_mask,
                    torch.ones_like(attn_mask)[:, : 2 * self.num_virtual_tokens],
                ],
                dim=-1,
            )

        # Optionally condition on mean gene representation from geneformer
        # Use input_ids to index into z_mean_tensor and get embeddings
        if self.mean_emb_dict:
            # in masked_labels, temporarily convert to 0 (padding token)
            masked_labels_output[masked_labels_output == -100] = 0

            z_mean_embeddings = self.z_mean[masked_labels_output]  # Shape: (b, e, e2)

            result_matrix = torch.cat([result_matrix, z_mean_embeddings], dim=-1)

            # Convert back
            masked_labels_output[masked_labels_output == 0] = -100

        # # Explore cells with no genes
        # idx = num_genes_per_cell == 0
        # if idx.sum() > 0:
        #     print('Number of cells with no genes', idx.sum())
        #     print('Result matrix')
        #     print(result_matrix[idx].sum()) # 0
        #     print('Masked labels')
        #     print(masked_labels_output[idx].sum()) # 0
        #     print('Num genes per cell')
        #     print(num_genes_per_cell[idx].sum()) # 0
        #     print('Attention mask')
        #     print(attn_mask[idx].sum()) # 44 (1 for each cls)

        return result_matrix, masked_labels_output, num_genes_per_cell, attn_mask

    def forward(
        self,
        gf_emb,
        input_dataset,
        return_attention,
        return_gene_embeddings=False,
        tokens_to_keep=None,
        gp_of_interest=None,
    ):
        # randomly mask genes only during training :
        if self.training:
            inference = False
        else:
            inference = True

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
                ) = self.build_input_matrix(
                    gf_emb,  # geneformer embeddings
                    input_dataset['input_ids'],
                    getattr(self, f'gp{i}_tokens'),
                    is_gpfinder=(self.gp_inputs[i] == 'remaining_var'),
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

                # Optionally append tokens for PEFT
                if self.num_virtual_tokens > 0:
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

                else:
                    virtual_tokens = None

                # get token GP representation, logits for gene level prediction,
                # and gene_labels where masked genes = -100
                encoder_output = self.encoder[i](
                    emb_pad,
                    attn_mask=attn_mask,
                    gene_labels=tokens_pad,
                    inference=inference,
                    return_attention=return_attention,
                    return_gene_embeddings=return_gene_embeddings,
                    num_virtual_tokens=self.num_virtual_tokens
                    * 2,  # *2 for shared and GP-specific tokens
                    using_gp_specific_token=self.num_virtual_tokens > 0,
                )

                # # Encoder output for cells with 0 GP genes
                # idx = num_genes_per_cell == 0
                # if idx.sum() > 0:
                #     cls = encoder_output['cls']
                #     logits_lm = encoder_output['logits_lm']
                #     gene_labels = encoder_output['gene_labels']

                #     print('Number of cells with no genes', idx.sum())
                #     print('CLS') # different for each cell
                #     if len(idx) > 1:
                #         print(cls[idx][:5])
                #     else:
                #         print(cls[idx])

                #     print('Logits')
                #     if len(idx) > 1:
                #         print(logits_lm[idx][:5])
                #     else:
                #         print(logits_lm[idx])

                #     print('Gene labels') # -100
                #     if len(idx) > 1:
                #         print(gene_labels[idx][:5])
                #     else:
                #         print(gene_labels[idx])

                #     raise ValueError('Cells with no genes')

                gp_token_list.append(encoder_output['cls'])
                logits_lm_list.append(encoder_output['logits_lm'])
                gene_labels_list.append(encoder_output['gene_labels'])

                if self.num_virtual_tokens > 0:
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

        if self.num_virtual_tokens > 0:
            output['virtual_tokens'] = virtual_tokens  # for global model
            output['gp_virtual_token_logits'] = gp_virtual_token_logit_list
            output['shared_virtual_token_logits'] = shared_virtual_token_logit_list

            # for saving embeddings
            output['gp_virtual_tokens'] = gp_virtual_token_list
            output['shared_virtual_tokens'] = shared_virtual_token_list

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

    def get_last_self_attn(self, gf_emb, input_dataset, gp_idx):
        '''
        If multilpe blocks, get attn matrix from last transformer block
        '''
        if self.training:
            inference = False
        else:
            inference = True

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

        # get token GP representation, logits for gene level prediction,
        # and gene_labels where masked genes = -100
        encoder_output = self.encoder[gp_idx](
            emb_pad,
            attn_mask=attn_mask,
            gene_labels=tokens_pad,
            inference=inference,
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

    def get_cls_attn(self, gf_emb, input_dataset, gp_idx, gene_names):
        '''
        If multilpe blocks, get attn matrix from last transformer block
        '''
        if self.training:
            inference = False
        else:
            inference = True

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
            inference=inference,
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


class cellWrapper(nn.Module):
    def __init__(
        self,
        gp_inputs,
        n_blocks,
        num_heads,
        gp_latent_size,
        global_masking_rate,
        use_flash,
        num_virtual_tokens,
    ):
        super().__init__()

        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.gp_inputs = gp_inputs
        self.gp_latent_size = gp_latent_size
        self.num_virtual_tokens = num_virtual_tokens

        self.encoder = gpTransformerEncoder(
            n_gp_tokens=len(self.gp_inputs),
            embed_dim=self.gp_latent_size,
            depth=self.n_blocks,
            num_heads=self.num_heads,
            mlm_masking_prob=global_masking_rate,
            use_flash=use_flash,
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

    def forward(self, x, inference):
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
            inference=inference,
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
            inference=True,
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


####################################
# Count reconstruction
####################################


class CountHead(nn.Module):
    def __init__(
        self,
        loss_mode: str = 'mse',
        n_genes: int = 25426,
        d_model: int = 256,
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
        d_model: int = 256,
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
        gene_counts_df=None,
        add_remaining_var=None,
        do_ensembl_conversion=True,
        gp_latent_size=256,
        num_heads=1,
        n_blocks=1,
        mgm_mask_ratio=0.5,
        use_flash=False,
        geneformer_model=GENEFORMER_MODEL_PATH,
        peft_config_path=None,
        gf_layer_to_quant=-1,
        gene_token_path=TOKEN_DICTIONARY_FILE,
        gene_name_path=GENE_NAME_FILE,
        model_type='Base',
        learn_new_gp=False,
        gp_of_interest=None,
        hvg_list=None,
        num_virtual_tokens=0,
        virtual_tokens_label=None,
        num_prompt_classes=0,
        mean_emb_dict=None,
        use_baseline_tk=False,
        tk_vocab_size=0,
        gene2vec=None,
        use_pos_emb=True,
        use_onehot_wrapper=False,
        vocab_gene_names=None,
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

        gf_layer_to_quant :
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

        # Initialize geneformer model for getting geneformer embeddings
        if use_baseline_tk:
            self.gf_wrapper = BaselineWrapper(tk_vocab_size, gene2vec)
        elif use_onehot_wrapper:
            self.gf_wrapper = OneHotWrapper(gene_names=vocab_gene_names)
        else:
            self.gf_wrapper = gfWrapper(
                geneformer_model=geneformer_model,
                gf_layer_to_quant=gf_layer_to_quant,
                peft_config_path=peft_config_path,
            )

        # Optionally: extract Geneformer cell embeddings
        self.gf_cell_encoder = AverageNonZero()

        # Set up token sets for each gene program
        if gp_inputs is None:
            gp_inputs = database.columns.tolist()
        elif isinstance(gp_inputs, str):
            gp_inputs = [gp_inputs]

        # / cause issues with saving
        gp_inputs = [x.replace('/', '_') for x in gp_inputs]
        database.columns = [x.replace('/', '_') for x in database.columns]

        gp_in_db = gp_inputs.copy()
        if 'remaining_var' in gp_in_db:
            gp_in_db.remove('remaining_var')

        self.gpdb = database[gp_in_db]
        self.gp_inputs = gp_inputs
        self.gp_latent_size = gp_latent_size
        self.mgm_mask_ratio = mgm_mask_ratio
        self.do_ensembl_conversion = do_ensembl_conversion
        self.n_blocks = n_blocks
        self.attn_dropout = attn_dropout
        self.use_flash = use_flash

        if isinstance(gp_of_interest, str):
            gp_of_interest = [gp_of_interest]
        self.gp_of_interest = gp_of_interest

        # ADD UNITTEST
        self.multi_gp_encoder = gpWrapper(
            database=self.gpdb,
            do_ensembl_conversion=self.do_ensembl_conversion,
            gene_counts_df=gene_counts_df,
            gene_token_path=gene_token_path,
            gene_name_path=gene_name_path,
            gp_latent_size=self.gp_latent_size,
            n_blocks=self.n_blocks,
            num_heads=num_heads,
            mgm_mask_ratio=self.mgm_mask_ratio,
            gp_inputs=gp_inputs,
            add_remaining_var=add_remaining_var,
            use_flash=self.use_flash,
            model_type=model_type,
            learn_new_gp=learn_new_gp,
            hvg_list=hvg_list,
            num_virtual_tokens=num_virtual_tokens,
            virtual_tokens_label=virtual_tokens_label,
            num_prompt_classes=num_prompt_classes,
            mean_emb_dict=mean_emb_dict,
            use_pos_emb=use_pos_emb,
        )

    def forward(
        self,
        input_dataset,
        return_gene_embeddings=False,
        return_attention=False,
        tokens_to_keep=None,
        return_gf_cell_emb=False,
        gp_of_interest=None,
        masking=True,
    ):
        if self.training:
            inference = False
        else:
            inference = True

        # input is tokenized dataset
        emb_out = self.gf_wrapper(input_dataset, inference)

        # Extract embeddings for each gene program
        # For backwards compataibilty
        # if not gp_of_interest attribute set to None
        if not hasattr(self, 'gp_of_interest'):
            self.gp_of_interest = None

        gp_to_pass = self.gp_of_interest if gp_of_interest is None else gp_of_interest

        output = self.multi_gp_encoder(
            emb_out,
            input_dataset,
            return_gene_embeddings=return_gene_embeddings,
            return_attention=return_attention,
            tokens_to_keep=tokens_to_keep,
            gp_of_interest=gp_to_pass,
        )

        # Optionally return geneformer cell embeddings
        if return_gf_cell_emb:
            gf_output_dict = self.gf_cell_encoder(emb_out)
            output['gf_emb'] = gf_output_dict['cls']

        return output

    def get_last_self_attn(self, input_dataset, gp_idx):
        if self.training:
            inference = False
        else:
            inference = True

        # Get Geneformer embeddings
        gf_emb = self.gf_wrapper(input_dataset, inference)

        output = self.multi_gp_encoder.get_last_self_attn(
            gf_emb, input_dataset, gp_idx=gp_idx
        )

        return output

    def get_cls_attn(self, input_dataset, gp):
        if self.training:
            inference = False
        else:
            inference = True

        # Get gp index
        gp_idx = self.gp_inputs.index(gp)
        gene_names = self.gpdb.iloc[:, gp_idx].tolist()

        # Get Geneformer embeddings
        gf_emb = self.gf_wrapper(input_dataset, inference)

        output = self.multi_gp_encoder.get_cls_attn(
            gf_emb, input_dataset, gp_idx=gp_idx, gene_names=gene_names
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
        n_bins=10,
        num_virtual_tokens=0,
        num_prototypes=0,
        prbm=None,
        cond_to_shift=None,
        **kwargs,
    ):
        super().__init__(
            use_flash=use_flash,
            model_type='Global',
            num_virtual_tokens=num_virtual_tokens,
            **kwargs,
        )
        self.global_attn_heads = global_attn_heads

        self.global_loss = global_loss

        # If using PRBM, set up the buffers on GPU
        if prbm is not None:
            self.use_prbm = True
            prbm = prbm[self.gp_inputs]
            prbm_tensor = torch.tensor(prbm.values.T).to(torch.float32)
            self.register_buffer('prbm', prbm_tensor)
        else:
            self.use_prbm = False

        self.cond_to_shift = cond_to_shift

        # for backwards compatibility
        if not hasattr(self, 'use_prbm'):
            self.use_prbm = False

        self.cell_token_learner = cellWrapper(
            gp_inputs=self.gp_inputs,
            gp_latent_size=self.gp_latent_size,
            n_blocks=global_n_blocks,
            num_heads=self.global_attn_heads,
            global_masking_rate=global_masking_rate,
            use_flash=use_flash,
            num_virtual_tokens=num_virtual_tokens,
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

        self.num_prototypes = num_prototypes
        if num_prototypes > 0:
            self.prototypes = nn.Parameter(
                torch.empty(
                    num_prototypes,  # use 1 prototype per class = set as same number
                    # num_prototypes,
                    self.gp_latent_size,
                ),
                requires_grad=True,
            )

            nn.init.xavier_normal_(self.prototypes)

    def forward(
        self,
        input_dataset,
        return_gene_embeddings=False,
        return_attention=False,
        tokens_to_keep=None,
        gp_of_interest=None,
    ):
        return_gf_cell_emb = True if self.global_loss == 'mse' else False

        if self.global_loss != 'masking':
            # no masking
            inference = True
        else:
            if self.training:
                inference = False
            else:
                inference = True

        base_output = super().forward(
            input_dataset,
            return_gene_embeddings,
            return_attention,
            tokens_to_keep,
            return_gf_cell_emb,
            gp_of_interest=gp_of_interest,
        )

        if return_gene_embeddings:
            return base_output

        # for each GP, optionally sum the reference mean embedding tensor
        if self.use_prbm:
            if self.cond_to_shift is not None:
                # cond_to_shift is of form {'name' : ['value']}
                condition_key = list(self.cond_to_shift.keys())[0]
                condition_values = self.cond_to_shift[condition_key]

                mask = torch.tensor(
                    [name in condition_values for name in input_dataset[condition_key]],
                    dtype=torch.bool,
                )
                mask = (
                    mask.unsqueeze(-1).unsqueeze(-1).to(base_output['z'].device)
                )  # Ensure correct broadcasting
                prbm_expanded = self.prbm.unsqueeze(0).expand_as(base_output['z'])

                base_output['z'] = torch.where(
                    mask,
                    base_output['z'] + (base_output['z'] - prbm_expanded),
                    base_output['z'],
                )
            else:
                base_output['z'] = base_output['z'] + (base_output['z'] - prbm_expanded)

        cell_output = self.cell_token_learner(base_output, inference=inference)

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

        if self.cell_token_learner.num_virtual_tokens > 0:
            logits = self.multi_gp_encoder.prompt_clf(cell_output['virtual_token'])
            base_output['global_virtual_token_logit'] = logits

        return base_output

    def get_cell_token_attention(self, input_dataset):
        base_output = super().forward(input_dataset)

        output = self.cell_token_learner.get_attn(base_output)

        return output


class iGpWrapper(nn.Module):
    def __init__(
        self,
        gp_transformer,
        clf_layer,
        gp_of_interest,
        gene_token_path=TOKEN_DICTIONARY_FILE,
        gene_name_path=GENE_NAME_FILE,
    ):
        super().__init__()
        # get index of gp of interest
        self.gp_of_interest = gp_of_interest
        self.gp_idx = gp_transformer.model.gp_inputs.index(gp_of_interest)

        # select relevant gp block
        self.gp_block = gp_transformer.model.multi_gp_encoder.encoder[self.gp_idx]
        self.clf_layer = clf_layer

        # store relevant gp tokens as nn.Embedding
        gp_tokens = getattr(
            gp_transformer.model.multi_gp_encoder, f'gp{self.gp_idx}_tokens'
        )

        # table for converting between different gene labels
        with open(gene_name_path, 'rb') as f:
            name_dictionary = pickle.load(f)
        with open(gene_token_path, 'rb') as f:
            token_dictionary = pickle.load(f)

        ensembl_to_name = {v: k for k, v in name_dictionary.items()}
        token_to_gene = {v: k for k, v in token_dictionary.items()}

        gene_conversion = {
            'token': gp_tokens.cpu().numpy().tolist(),
            'ensembl': [token_to_gene[t.item()] for t in gp_tokens],
        }

        gene_conversion['symbol'] = [
            ensembl_to_name[e] for e in gene_conversion['ensembl']
        ]

        self.gene_conversion = gene_conversion

    def forward(self, emb, additional_input_dict):
        output = self.gp_block(
            emb,
            attn_mask=additional_input_dict['attn_mask'],
            gene_labels=additional_input_dict['token_labels'],
            inference=True,
            return_attention=False,
            return_gene_embeddings=False,
        )

        logits = self.clf_layer(output['cls'])
        return logits

        # out = output['logits_lm'][:, 0, :]
        # return out.max(1).values


class iGlobalWrapper(nn.Module):
    def __init__(
        self,
        gp_transformer,
        clf_layer=None,
        global_loss='reconstruction',
        task_index=None,
        use_embedding=False,
        pretrained_emb=None,
        vocab_size=None,
        embedding_dim=None,
    ):
        super().__init__()

        self.global_block = gp_transformer.model.cell_token_learner
        # set decoder to identiy
        self.global_block.encoder.decoder = nn.Identity()

        if global_loss != 'supervised':
            if clf_layer is None:
                raise ValueError('Please provide a classifier layer')
            self.clf_layer = clf_layer
        else:
            if task_index is None:
                raise ValueError('Please provide a task index')
            self.clf_layer = gp_transformer.model.clf_head[task_index]

        self.use_embedding = use_embedding
        if self.use_embedding:
            self.gp_embedding = PretrainedEmbeddings(
                pretrained_emb=pretrained_emb,
                pretrained_pos_emb=self.global_block.encoder.pos_embed,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
            )

            # turn off positional embeddings
            self.global_block.encoder.pos_embed = nn.Identity()

    def forward(self, emb, additional_input_dict):
        if self.use_embedding:
            emb = self.gp_embedding(emb)

        input_dataset = {
            'z': emb,
            'num_genes_per_cell_list': additional_input_dict['num_genes_per_cell_list'],
        }

        # Global cell token learner
        out = self.global_block(input_dataset, inference=False)

        # Pass through linear layer
        logits = self.clf_layer(out['cell_token'])

        # return logits.max(1).values
        return logits


####################################
# Baseline : averaging GP embeddings
####################################


class AverageNonZero(nn.Module):
    def __init__(self, cls_tag='cls'):
        super().__init__()
        self.cls_tag = cls_tag
        self.num_virtual_tokens = 0

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
        # also for compatability: extract tensor if necessary
        if isinstance(x, dict):
            x = x['z']

        # Replace zero values with NaN to facilitate ignoring them during averaging
        x[x == 0] = float('nan')

        # Calculate the mean along the last dimension (embedding_dim)
        # Specify 'nanmean' to ignore NaN values during the mean calculation
        x = torch.nanmean(x, dim=1)

        # Replace NaN values with 0
        x[torch.isnan(x)] = 0

        if torch.isnan(x).any():
            print('Found nan in line 1847 of AverageNZ')

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
        emb_pad, tokens_pad, _, attn_mask = self.build_input_matrix(
            gf_emb,  # geneformer embeddings
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
        gene_counts_df,
        num_heads,
        gene_token_path=TOKEN_DICTIONARY_FILE,
        gene_name_path=GENE_NAME_FILE,
        add_remaining_var=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.multi_gp_encoder = gpAverager(
            database=self.gpdb,
            do_ensembl_conversion=self.do_ensembl_conversion,
            gene_counts_df=gene_counts_df,
            gp_latent_size=self.gp_latent_size,
            n_blocks=self.n_blocks,
            num_heads=num_heads,
            mgm_mask_ratio=self.mgm_mask_ratio,
            gene_token_path=gene_token_path,
            gene_name_path=gene_name_path,
            gp_inputs=self.gp_inputs,
            add_remaining_var=add_remaining_var,
            use_flash=False,
            model_type='Mean',
            learn_new_gp=False,
            # MAY NEED TO UPDATE THIS
            hvg_list=None,
            num_virtual_tokens=0,
            virtual_tokens_label=None,
            num_prompt_classes=0,
            mean_emb_dict=None,
            use_pos_emb=False,
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
        gene_counts_df,
        num_heads,
        gene_token_path=TOKEN_DICTIONARY_FILE,
        gene_name_path=GENE_NAME_FILE,
        add_remaining_var=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.multi_gp_encoder = gpAverager(
            database=self.gpdb,
            do_ensembl_conversion=self.do_ensembl_conversion,
            gene_counts_df=gene_counts_df,
            gp_latent_size=self.gp_latent_size,
            n_blocks=self.n_blocks,
            num_heads=num_heads,
            mgm_mask_ratio=self.mgm_mask_ratio,
            gene_token_path=gene_token_path,
            gene_name_path=gene_name_path,
            gp_inputs=self.gp_inputs,
            add_remaining_var=add_remaining_var,
            use_flash=False,
            model_type='Mean',
            learn_new_gp=False,
            # MAY NEED TO UPDATE THIS
            hvg_list=None,
            num_virtual_tokens=0,
            virtual_tokens_label=None,
            num_prompt_classes=0,
            mean_emb_dict=None,
            use_pos_emb=False,
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
    ):
        super().__init__()

        self.clf_head = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        return self.clf_head(x)


if __name__ == '__main__':
    pass
