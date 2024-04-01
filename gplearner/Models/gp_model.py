####################################
# Load packages
####################################

import warnings
from typing import Dict, Optional

# imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from scipy.sparse import csr_matrix
from transformers import BertForMaskedLM

from ..Modules.modules import Mlp, gpTransformerEncoder
from ..Utils.geneformer_utils import EmbExtractor
from ..Utils.utils import get_gp_tokens, pad_array

####################################
# Geneformer
####################################

GENE_NAME_FILE = '/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/Geneformer/geneformer/gene_name_id_dict.pkl'  # noqa
GENEFORMER_MODEL_PATH = (
    '/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/Geneformer/'
)


class gfWrapper(nn.Module):
    def __init__(
        self,
        geneformer_model,
        gf_layer_to_quant,
    ):
        super().__init__()

        # Initialize geneformer model for getting geneformer embeddings
        self.gf = BertForMaskedLM.from_pretrained(
            geneformer_model, output_attentions=False, output_hidden_states=True
        )  # .to("cuda")

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
    ):
        super().__init__()

        self.gp_latent_size = gp_latent_size
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.mgm_mask_ratio = mgm_mask_ratio
        self.gp_inputs = gp_inputs
        self.model_type = model_type
        self.learning_new_gp = learn_new_gp

        # Store all genes included in at least one GP
        self.all_gp_tokens = set()

        # TO DO : ADD THIS TO A FUNCTION THAT RETURNS TWO DICTIONARIES
        for i, gpi in enumerate(self.gp_inputs):
            setattr(
                self,
                f'gp{i}_tokens',
                get_gp_tokens(
                    gpi,
                    database,
                    do_ensembl_conversion,
                    gene_counts_df,
                    gene_token_path,
                    gene_name_path,
                ),
            )

            print('Number of genes in GP', gpi, len(getattr(self, f'gp{i}_tokens')))
            self.all_gp_tokens.update(getattr(self, f'gp{i}_tokens'))

            # within each gene program,
            # gene tokens need to be re-encoded to avoid having 25_000 classes in each
            setattr(
                self,
                f'gp{i}_tokens_encoded',
                self.encode_gp_tokens(getattr(self, f'gp{i}_tokens')),
            )

        self.encoder = nn.ModuleList(
            [
                gpTransformerEncoder(
                    n_gp_tokens=len(getattr(self, f'gp{i}_tokens')),
                    embed_dim=self.gp_latent_size,
                    depth=self.n_blocks,
                    num_heads=self.num_heads,
                    mlm_masking_prob=self.mgm_mask_ratio,
                    use_flash=use_flash,
                )
                for i in range(len(gp_inputs))
            ]
        )

        self.add_remaining_var = add_remaining_var
        if self.add_remaining_var:
            n_gp = len(self.gp_inputs)
            gp_inputs.append('remaining_var')
            self.gp_inputs = gp_inputs

            # find non GP genes
            if gene_counts_df is None:
                raise ValueError(
                    'Please provide a dataframe'
                    'with counts of gene occurrences in dataset'
                )

            non_gp_tokens = set(gene_counts_df['token'].tolist()) - self.all_gp_tokens

            setattr(self, f'gp{n_gp}_tokens', non_gp_tokens)
            setattr(
                self, f'gp{n_gp}_tokens_encoded', self.encode_gp_tokens(non_gp_tokens)
            )

            self.encoder.append(
                gpTransformerEncoder(
                    n_gp_tokens=100,  # we only keep top 100 genes
                    vocab_size=len(non_gp_tokens),
                    embed_dim=self.gp_latent_size,
                    depth=self.n_blocks,
                    num_heads=num_heads,
                    mlm_masking_prob=self.mgm_mask_ratio,
                )
            )

    def build_input_matrix(
        self, gf, input_ids, gpi_tokens_list, gp_idx, mode='full_model'
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

        """
        # Get list of gp tokens
        gp_tokens = np.array(list(gpi_tokens_list)).astype(np.int16)

        # Convert input IDs (list of lists) to array:
        holder = []

        # Find max value for padding
        if mode == 'full_model':
            max_value = 2048

            for i in range(len(input_ids)):
                if len(input_ids[i]) == max_value:
                    holder.append(input_ids[i].cpu().numpy())
                else:
                    padded = pad_array(
                        input_ids[i].cpu().numpy(), desired_length=max_value
                    )
                    holder.append(padded)

        else:
            # when we are filtering gene embeddings,
            # outputs are already padded to same length
            holder = input_ids.cpu().numpy()

        # Build an array (n_cells, n_genes) with token IDs at each position
        tokens_arr = np.array(holder)

        # binary mask (h, i, k)
        # in cell h, is the gene as position i in our GP at position k?
        mask = (tokens_arr[:, :, np.newaxis] == gp_tokens[np.newaxis, :]).astype(int)

        # Now reshape so that we will zero out non GP genes in each cell
        mask_expanded = torch.tensor(mask.sum(axis=-1)[:, :, np.newaxis]).to(gf.device)

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

        result_matrix = torch.stack(holder).to(gf.device)

        # Now do the same for labels
        masked_labels = mask.sum(axis=-1) * tokens_arr

        holder = []
        for i in range(masked_labels.shape[0]):
            x = masked_labels[i, :]
            nz = x != 0
            z = np.concatenate((x[nz], x[~nz]))
            holder += [z]

        masked_labels_output = np.array(holder)

        # count number of genes per cell
        num_genes_per_cell = mask.sum(axis=-1).sum(axis=-1)

        # Make tensor for forward pass
        masked_labels_output = torch.tensor(masked_labels_output).to(gf.device)
        # comment the line below to leave 0s because they are actually informative
        # (this gene was not in the top 1000 of this cell)
        masked_labels_output[masked_labels_output == 0] = -100

        # We know that at most, the non zero genes is the number of genes in the GP
        # for known GP we keep all genes
        # for remaining var we only keep top 100
        if self.add_remaining_var and gp_idx == len(self.gp_inputs) - 1:
            n_genes_to_keep = 100
        else:
            n_genes_to_keep = len(gpi_tokens_list)
        result_matrix = result_matrix[:, :n_genes_to_keep, :]
        masked_labels_output = masked_labels_output[:, :n_genes_to_keep]

        # Set up attention mask
        # to avoid attention to padding tokens
        attn_mask = torch.zeros_like(masked_labels_output)
        attn_mask[masked_labels_output != -100] = 1

        # never mask cls
        attn_mask = torch.cat(
            [torch.ones(attn_mask.shape[0], 1).to(attn_mask.device), attn_mask], dim=1
        )

        return result_matrix, masked_labels_output, num_genes_per_cell, attn_mask

    def encode_gp_tokens(self, gp_tokens):
        """
        Convert tokens to encoded values inside transformer block
        goes from 1 to n_gp_genes not starting from 0 so 0 corresponds to missing gene
        """
        return {gene: idx for idx, gene in enumerate(gp_tokens)}

    def forward(
        self,
        gf_emb,
        input_dataset,
        return_attention,
        return_gene_embeddings=False,
        tokens_to_keep=None,
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
        gp_labels_list = []

        # Extract embeddings for each gene program
        for i in range(len(self.gp_inputs)):
            # when learning new GP, onyl do FW pass on new GP
            if (
                (self.training)
                & (self.encoder[i].blocks[0].attn.qkv.weight.requires_grad is False)
                & (self.model_type == 'Base')
                # & (self.learning_new_gp) # commented out for backwards compatibility
                # but would be good to have
            ):
                continue
            else:
                (
                    emb_pad,
                    tokens_pad,
                    num_genes_per_cell,
                    attn_mask,
                ) = self.build_input_matrix(
                    gf_emb,  # geneformer embeddings
                    input_dataset['input_ids'],
                    getattr(self, f'gp{i}_tokens'),
                    gp_idx=i,
                )

                # track number of genes per cell
                # divide by GP length
                num_genes_per_cell = num_genes_per_cell / len(
                    getattr(self, f'gp{i}_tokens')
                )
                num_genes_per_cell_list += [num_genes_per_cell]

                # Encode tokens for encoding
                tokens_pad_unencoded = tokens_pad

                tokens_pad = (
                    tokens_pad.cpu()
                    .apply_(
                        lambda x: getattr(self, f'gp{i}_tokens_encoded')[x]
                        if x in getattr(self, f'gp{i}_tokens_encoded').keys()
                        else -100
                    )
                    .to(emb_pad.device)
                )

                # get token GP representation, logits for gene level prediction,
                # and gene_labels where masked genes = -100

                encoder_output = self.encoder[i](
                    emb_pad,
                    attn_mask=attn_mask,
                    gene_labels=tokens_pad,
                    inference=inference,
                    return_attention=return_attention,
                    return_gene_embeddings=return_gene_embeddings,
                )

                gp_token_list.append(encoder_output['cls'])
                logits_lm_list.append(encoder_output['logits_lm'])
                gene_labels_list.append(encoder_output['gene_labels'])

                if return_gene_embeddings:
                    gene_emb_list.append(encoder_output['gene_embeddings'])
                    gene_original_labels_list.append(tokens_pad_unencoded)
                    gp_labels_list.append(
                        [self.gp_inputs[i] for _ in range(gf_emb[0].shape[0])]
                    )

        # Concatenate tensors
        z = torch.stack(gp_token_list, dim=1)

        # store for output
        output = {
            'z': z,
            'logits_lm_list': logits_lm_list,
            'gene_labels_list': gene_labels_list,
            'gene_emb_list': gene_emb_list,
            'gp_labels_list': gp_labels_list,
            'gene_original_labels_list': gene_original_labels_list,
            'num_genes_per_cell_list': num_genes_per_cell_list,
        }

        if return_gene_embeddings:
            output = self.filter_gene_embeddings(output, tokens_to_keep)

        return output

    def filter_gene_embeddings(self, emb_dict, tokens_to_keep):
        gene_emb_list = emb_dict['gene_emb_list']
        gp_labels_list = emb_dict['gp_labels_list']
        tokens_list = emb_dict['gene_original_labels_list']

        x_scgpl = []
        tokens_scgpl = []
        gp_labels = []

        # Filter to only keep genes in multiple GP
        # loop through emb list = embeddings are grouped by GP
        for i in range(len(gene_emb_list)):
            x_out, tokens, _, attn_mask = self.build_input_matrix(
                gene_emb_list[i],
                tokens_list[i],
                tokens_to_keep,
                mode='extract_genes',
                gp_idx=i,
            )

            gp_label_i = gp_labels_list[i]

            # remove missing values
            x_out = x_out.reshape(x_out.shape[0] * x_out.shape[1], -1)
            non_missing = (x_out != 0).all(dim=1)
            x_out = x_out[non_missing]

            tokens = tokens.reshape(tokens.shape[0] * tokens.shape[1])
            tokens = tokens[tokens != -100]

            gp_label_out = [gp_label_i[0] for _ in range(tokens.shape[0])]

            # Add to list
            x_scgpl.append(x_out)
            tokens_scgpl.append(tokens)
            gp_labels.append(gp_label_out)

        output = {
            'x_scgpl': x_scgpl,
            'tokens_scgpl': tokens_scgpl,
            'gp_labels': gp_labels,
        }

        return output

    def get_last_self_attn(self, gf_emb, input_dataset, gp_idx):
        # randomly mask genes only during training :
        inference = False

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

        # get token GP representation, logits for gene level prediction,
        # and gene_labels where masked genes = -100
        encoder_output = self.encoder[gp_idx](
            emb_pad,
            attn_mask=attn_mask,
            gene_labels=tokens_pad,
            inference=inference,
            return_attention=True,
        )

        # Reorder attention matrix so genes are in the same order in each cell
        attn = encoder_output['attention']

        # Average attention across heads
        attn = attn.mean(dim=1)

        # For the padding tokens, attention will be 0
        # so we can randomly reassign gene tokens to help with ranking
        all_gp_tokens = set(getattr(self, f'gp{gp_idx}_tokens_encoded').values())
        # add one for cls
        all_gp_tokens.add(max(all_gp_tokens) + 1)

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

            # cls is at first position in embedding
            # so we move the label the first position
            new_padded = torch.cat([new_padded[-1].unsqueeze(0), new_padded[:-1]])

            holder.append(new_padded)

        tokens_pad = torch.stack(holder).long().to(attn.device)

        # Reorder attention matrix so genes are in the same order in each cell
        # Create an index tensor to sort tokens_pad
        _, indices = torch.sort(tokens_pad, dim=1)

        # Apply sorting to the corresponding rows in x
        attn = torch.gather(attn, 1, indices)
        tokens_pad = torch.gather(tokens_pad, 1, indices)

        # the cls label is 1 + number of gp
        # so sorting will move it to last position
        # bring back to the start
        attn = torch.cat([attn[:, -1].unsqueeze(1), attn[:, :-1]], dim=1)
        tokens_pad = torch.cat([tokens_pad[-1].unsqueeze(0), tokens_pad[:-1]])

        output = {
            'attn': csr_matrix(attn.detach().cpu().numpy()),
        }

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
        )

    def build_input_matrix(self, z, num_genes_per_cell_list):
        # Prepare labels
        batch_size = z.shape[0]
        gp_labels = torch.tensor(
            [[i for i in range(len(self.gp_inputs))]] * batch_size
        ).to(z.device)

        # reorder gp based on number of genes per cell
        n_genes_per_cell = torch.tensor(np.array(num_genes_per_cell_list).T).to(
            z.device
        )

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

        encoder_output = self.encoder(
            z,
            gene_labels=gp_labels,
            attn_mask=attn_mask,
            inference=inference,
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
            inference=True,
            return_attention=True,
        )

        # Reorder attention matrix so GP are in the same order in each cell
        attn = encoder_output['attention']

        # Average attention across heads
        attn = attn.mean(dim=1)

        # For the padding tokens, attention will be 0
        # + 1 for cls
        all_gp = set([i for i in range(len(self.gp_inputs) + 1)])

        holder = []

        for i in range(gp_labels.shape[0]):
            # because GP are ranked by number of genes per cell
            # all the -100 tokens will be at the end
            x = gp_labels[i, :]
            labeled_gp_idx = x != -100

            values_to_fill_in = all_gp - set(x[labeled_gp_idx].cpu().numpy().tolist())

            new_labels = torch.tensor(list(values_to_fill_in)).to(x.device)

            new_padded = torch.concat([x[labeled_gp_idx], new_labels], dim=0)

            # cls is at first position in embedding
            # so we move the label the first position
            new_padded = torch.cat([new_padded[-1].unsqueeze(0), new_padded[:-1]])

            holder.append(new_padded)

        gp_labels = torch.stack(holder).long().to(attn.device)

        # Reorder attention matrix so GP are in the same order in each cell
        # Create an index tensor to sort tokens_pad
        _, indices = torch.sort(gp_labels, dim=1)

        # Apply sorting to the corresponding rows in x
        attn = torch.gather(attn, 1, indices)
        gp_labels = torch.gather(gp_labels, 1, indices)

        # the cls label is 1 + number of gp
        # so sorting will move it to last position
        # bring back to the start
        attn = torch.cat([attn[:, -1].unsqueeze(1), attn[:, :-1]], dim=1)
        gp_labels = torch.cat([gp_labels[-1].unsqueeze(0), gp_labels[:-1]])

        output = {
            'attn': csr_matrix(attn.detach().cpu().numpy()),
        }

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
        mlp_output = F.normalize(mlp_output, dim=-1, p=2)
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
    """
    Model to learn GP latent representation

    """

    def __init__(
        self,
        database,
        attn_dropout=0,
        gp_inputs=None,
        gene_counts_df=None,
        add_remaining_var=False,
        do_ensembl_conversion=True,
        gp_latent_size=256,
        num_heads=1,
        n_blocks=1,
        mgm_mask_ratio=0.5,
        use_flash=False,
        geneformer_model=GENEFORMER_MODEL_PATH,
        gf_layer_to_quant=-1,
        gene_token_path=TOKEN_DICTIONARY_FILE,
        gene_name_path=GENE_NAME_FILE,
        model_type='Base',
        learn_new_gp=False,
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
        self.gf_wrapper = gfWrapper(
            geneformer_model=geneformer_model, gf_layer_to_quant=gf_layer_to_quant
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

        self.gpdb = database[gp_inputs]
        self.gp_inputs = gp_inputs
        self.gp_latent_size = gp_latent_size
        self.mgm_mask_ratio = mgm_mask_ratio
        self.do_ensembl_conversion = do_ensembl_conversion
        self.n_blocks = n_blocks
        self.attn_dropout = attn_dropout
        self.use_flash = use_flash

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
        )

    def forward(
        self,
        input_dataset,
        return_gene_embeddings=False,
        return_attention=False,
        tokens_to_keep=None,
        return_gf_cell_emb=False,
    ):
        if self.training:
            inference = False
        else:
            inference = True

        # input is tokenized dataset
        emb_out = self.gf_wrapper(input_dataset, inference)

        # Extract embeddings for each gene program
        output = self.multi_gp_encoder(
            emb_out,
            input_dataset,
            return_gene_embeddings=return_gene_embeddings,
            return_attention=return_attention,
            tokens_to_keep=tokens_to_keep,
        )

        # Optionally return geneformer cell embeddings
        if return_gf_cell_emb:
            gf_output_dict = self.gf_cell_encoder(emb_out)
            output['gf_emb'] = gf_output_dict['cls']

        return output

    def get_last_self_attn(self, input_dataset, gp):
        gp_idx = self.gp_inputs.index(gp)
        # input is tokenized dataset
        emb_out = self.gf_wrapper(input_dataset)

        # Extract attention matrix for our GP of interest
        output = self.multi_gp_encoder.get_last_self_attn(
            emb_out, input_dataset, gp_idx=gp_idx
        )

        return output


class gpTransformerGlobal(gpTransformerBase):
    """
    Learn individual GP representations + global cell token
    """

    def __init__(
        self,
        global_attn_heads=8,
        global_loss='supervised',
        total_n_genes=25426,
        reconstruction_loss='mse',
        supervised_labels: Optional[Dict] = None,
        global_masking_rate=0,
        global_n_blocks=1,
        use_flash=False,
        **kwargs,
    ):
        super().__init__(use_flash=use_flash, model_type='Global', **kwargs)
        self.global_attn_heads = global_attn_heads

        self.global_loss = global_loss

        self.cell_token_learner = cellWrapper(
            gp_inputs=self.gp_inputs,
            gp_latent_size=self.gp_latent_size,
            n_blocks=global_n_blocks,
            num_heads=self.global_attn_heads,
            global_masking_rate=global_masking_rate,
            use_flash=use_flash,
        )

        if self.global_loss == 'supervised':
            if supervised_labels is None:
                raise ValueError(
                    'Please provide a dictionary' 'of the form {task_name : n_classes}'
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
                loss_mode=reconstruction_loss, n_genes=total_n_genes
            )

    def forward(
        self,
        input_dataset,
        return_gene_embeddings=False,
        return_attention=False,
        tokens_to_keep=None,
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
        )

        cell_output = self.cell_token_learner(base_output, inference)

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

        return base_output

    def get_cell_token_attention(self, input_dataset):
        base_output = super().forward(input_dataset)

        output = self.cell_token_learner.get_attn(base_output)

        return output


####################################
# Baseline : averaging GP embeddings
####################################


class AverageNonZero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
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

        # print count of nan values to check
        output = {'cls': x, 'logits_lm': [], 'gene_labels': []}

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cell_token_learner = AverageNonZero()

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
    from gplearner.Datamodules.datamodule import txDataModule

    txdata = txDataModule(
        folder='/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium'
        '/scgpl_reproducibility/examples/pbmc_ifn/data/input_dataset',
        batch_size=32,
    )
    txdata.setup()

    dataloader = txdata.val_dataloader()

    iterator = iter(dataloader)

    batch = next(iterator)

    gpdb = pd.read_csv(
        '/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/'
        'scgpl_reproducibility/examples/pbmc_ifn/ifn_db_3gp.csv'
    )

    model = gpTransformerGlobal(
        database=gpdb,
        attn_dropout=0,
        gene_counts_df=None,
        do_ensembl_conversion=True,
        gp_latent_size=256,
        num_heads=8,
        n_blocks=1,
        mgm_mask_ratio=0.5,
        geneformer_model='/lustre/scratch126/cellgen/team292/mm58/'
        'geneformer_endometrium/Geneformer/',
        gf_layer_to_quant=-1,
        global_loss='masking',
        global_attn_heads=1,
        global_masking_rate=0.3,
    )

    out = model.forward(batch)

    for k, v in out.items():
        print(k, v.shape)

    print('done')
