####################################
# Load packages
####################################

import warnings
from typing import Dict, Optional
from types import SimpleNamespace
import json 

# imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from transformers import BertForMaskedLM

from ..Modules.modules import gpTransformerEncoder
from ..Utils.geneformer_utils import EmbExtractor
from ..Utils.utils import get_gp_tokens, pad_array

from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
####################################
# Geneformer
####################################


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
# scGPT ####
####################################

class scgptWrapper(nn.Module):
    def __init__(
        self,
        scgpt_mod='scGPT_human',
    ):
        super().__init__()

        hyperparameter_defaults = dict(
            seed=0,
            dataset_name="ms",
            do_train=False,
            load_model=f"/lustre/scratch126/cellgen/team205/ha11/scGPT/{scgpt_mod}/",
            mask_ratio=0.0,
            epochs=10,
            n_bins=51,
            MVC=False, # Masked value prediction for cell embedding
            ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
            dab_weight=0.0,
            lr=1e-4,
            batch_size=32,
            layer_size=128,
            nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            nhead=4,  # number of heads in nn.MultiheadAttention
            dropout=0.2,  # dropout probability
            schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
            save_eval_interval=5,
            fast_transformer=True,
            pre_norm=False,
            amp=True,  # Automatic Mixed Precision
            include_zero_gene = False,
            freeze = False, #freeze
            DSBN = False,  # Domain-spec batchnorm
        )
        config = SimpleNamespace(**hyperparameter_defaults)
        # settings for input and preprocessing
        pad_token = "<pad>"
        special_tokens = [pad_token, "<cls>", "<eoc>"]
        mask_ratio = config.mask_ratio
        mask_value = "auto"  # for masked values, now it should always be auto
        
        include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
        max_seq_len = 3001
        n_bins = config.n_bins
        
        # input/output representation
        input_style = "binned"  # "normed_raw", "log1p", or "binned"
        output_style = "binned"  # "normed_raw", "log1p", or "binned"
        
        # settings for training
        MLM = False  # whether to use masked language modeling, currently it is always on.
        CLS = True  # celltype classification objective
        ADV = False  # Adversarial training for batch correction
        CCE = False  # Contrastive cell embedding objective
        MVC = config.MVC  # Masked value prediction for cell embedding
        ECS = config.ecs_thres > 0  # Elastic cell similarity objective
        DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
        INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
        input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
        cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
        adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
        adv_D_delay_epochs = 0
        mvc_decoder_style = "inner product"
        ecs_threshold = config.ecs_thres
        dab_weight = config.dab_weight
        
        explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
        do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training
        
        per_seq_batch_sample = False
        
        if input_emb_style == "category":
            mask_value = n_bins + 1
            pad_value = n_bins  # for padding gene expr values
            n_input_bins = n_bins + 2
        else:
            mask_value = -1
            pad_value = -2
            n_input_bins = n_bins
        
        # settings for optimizer
        lr = config.lr  # TODO: test learning rate ratio between two tasks
        lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
        batch_size = config.batch_size
        eval_batch_size = config.batch_size
        epochs = config.epochs
        schedule_interval = 1
        
        # settings for the model
        fast_transformer = config.fast_transformer
        fast_transformer_backend = "flash"  # "linear" or "flash"
        embsize = config.layer_size  # embedding dimension
        d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
        nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
        nhead = config.nhead  # number of heads in nn.MultiheadAttention
        dropout = config.dropout  # dropout probability
        
        # logging
        log_interval = 100  # iterations
        save_eval_interval = config.save_eval_interval  # epochs
        do_eval_scib_metrics = True
        num_types = 5 # TODO: hard coded for synthetic data
        cell_embedding_mode = 'cls'
        max_length = 9585
        batch_size = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_file = f"/lustre/scratch126/cellgen/team205/ha11/scGPT/{scgpt_mod}/vocab.json"
        self.vocab = GeneVocab.from_file(self.vocab_file)
        with open(f'/lustre/scratch126/cellgen/team205/ha11/scGPT/{scgpt_mod}/args.json', "r") as f:
            self.model_configs = json.load(f)
        ntokens = len(self.vocab)  # size of vocabulary
        self.model = TransformerModel(
            ntokens,
            embsize,
            nhead,
            d_hid,
            nlayers,
            nlayers_cls=3,
            n_cls=num_types if CLS else 1,
            vocab=self.vocab,
            dropout=dropout,
            pad_token=pad_token,
            pad_value=pad_value,
            do_mvc=MVC,
            do_dab=DAB,
            use_batch_labels=INPUT_BATCH_LABELS,
            num_batch_labels=0,
            domain_spec_batchnorm=config.DSBN,
            input_emb_style=input_emb_style,
            n_input_bins=n_input_bins,
            cell_emb_style=cell_emb_style,
            mvc_decoder_style=mvc_decoder_style,
            ecs_threshold=ecs_threshold,
            explicit_zero_prob=explicit_zero_prob,
            use_fast_transformer=True, # TODO: create new env on farm22 to add flash_transformer. change to True on farm5
            fast_transformer_backend=fast_transformer_backend,
            pre_norm=config.pre_norm,
        )
        self.use_batch_labels = INPUT_BATCH_LABELS

    def forward(self, data_dict, *args, **kwargs):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            count = 0
            input_gene_ids = data_dict["gene"].to(self.device)
            src_key_padding_mask = input_gene_ids.eq(
                self.vocab[self.model_configs["pad_token"]]
            )
            embeddings = self.model._encode(
                input_gene_ids,
                data_dict["expr"].to(self.device),
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=data_dict["batch_labels"].to(self.device)
                if self.use_batch_labels
                else None,
            )
            embeddings = embeddings[:, 1:, :]  # get all the token embeddings except the <cls> (just the corresponding gene positions)
            return embeddings
    

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
    ):
        super().__init__()

        self.gp_latent_size = gp_latent_size
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.mgm_mask_ratio = mgm_mask_ratio
        self.gp_inputs = gp_inputs

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

        for i in range(len(self.gp_inputs)):
            print(
                'Number of genes in GP',
                self.gp_inputs[i],
                len(getattr(self, f'gp{i}_tokens')),
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
            max_value = gf.shape[1]

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
        self, gp_inputs, n_blocks, num_heads, gp_latent_size, global_masking_rate
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
# Define model
####################################


class gpTransformerBase(nn.Module):
    """
    Model to learn GP latent representation

    """

    def __init__(
        self,
        database,
        mode='geneformer',
        attn_dropout=0,
        gp_inputs=None,
        gene_counts_df=None,
        add_remaining_var=False,
        do_ensembl_conversion=True,
        gp_latent_size=256,
        num_heads=1,
        n_blocks=1,
        mgm_mask_ratio=0.5,
        geneformer_model='/lustre/scratch126/cellgen/team292/mm58/'
        'geneformer_endometrium/Geneformer/',
        gf_layer_to_quant=-1,
        gene_token_path='/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/Geneformer/geneformer/token_dictionary.pkl',
        gene_name_path='/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/Geneformer/geneformer/gene_name_id_dict.pkl',
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
        if mode == 'geneformer':
            self.gf_wrapper = gfWrapper(
                geneformer_model=geneformer_model, gf_layer_to_quant=gf_layer_to_quant
            )
            gene_token_path = '/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/Geneformer/geneformer/token_dictionary.pkl'
            gene_name_path = '/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium/Geneformer/geneformer/gene_name_id_dict.pkl'
        elif mode == 'scgpt':
            self.gf_wrapper = scgptWrapper()
            gene_token_path = '/lustre/scratch126/cellgen/team205/ha11/scGPT/synthetic_token_dict.pkl'
            gene_name_path = '/lustre/scratch126/cellgen/team205/ha11/scGPT/gene_name_id_dict.pkl'
        else:
            raise NotImplementedError()

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
        supervised_labels: Optional[Dict] = None,
        global_masking_rate=0,
        global_n_blocks=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_attn_heads = global_attn_heads

        self.global_loss = global_loss

        self.cell_token_learner = cellWrapper(
            gp_inputs=self.gp_inputs,
            gp_latent_size=self.gp_latent_size,
            n_blocks=global_n_blocks,
            num_heads=self.global_attn_heads,
            global_masking_rate=global_masking_rate,
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

    def forward(
        self,
        input_dataset,
        return_gene_embeddings=False,
        return_attention=False,
        tokens_to_keep=None,
    ):
        return_gf_cell_emb = True if self.global_loss == 'mse' else False

        if self.global_loss != 'masking':
            inference = False
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
        gene_token_path='/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium'
        '/Geneformer/geneformer/token_dictionary.pkl',
        gene_name_path='/lustre/scratch126/cellgen/team292/mm58/geneformer_endometrium'
        '/Geneformer/geneformer/gene_name_id_dict.pkl',
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
