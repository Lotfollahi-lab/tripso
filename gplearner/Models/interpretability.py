import pickle

import torch.nn as nn

from ..Modules.modules import PretrainedEmbeddings

####################################
# For GradCAM
####################################


class iGpWrapper(nn.Module):
    def __init__(self, gp_transformer, clf_layer, gp_of_interest):
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
        with open(gp_transformer.model.multi_gp_encoder.gene_name_path, 'rb') as f:
            name_dictionary = pickle.load(f)
        with open(gp_transformer.model.multi_gp_encoder.gene_token_path, 'rb') as f:
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
            masking=False,
            return_attention=False,
            return_gene_embeddings=False,
        )

        logits = self.clf_layer(output['cls'])

        return logits


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
        out = self.global_block(input_dataset, masking=False)

        # Pass through linear layer
        logits = self.clf_layer(out['cell_token'])

        # return logits.max(1).values
        return logits
