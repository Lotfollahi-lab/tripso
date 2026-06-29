import math
import pickle
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
# Vocab remap helper
####################################


def _build_gf_to_local_remap(gf_token_dict_path, local_vocab, pad_id):
    """Build a LongTensor remapping Geneformer token ids to a local vocab.

    Parameters
    ----------
    gf_token_dict_path : str or Path
        Path to geneformer_token_dictionary_may2025.pkl (gene_name -> gf_token_id).
    local_vocab : dict
        Maps gene_name -> local_token_id.
    pad_id : int
        Local vocab id to use for genes not found in local_vocab.

    Returns
    -------
    torch.LongTensor of shape (gf_vocab_size,)
        remap[gf_token_id] = local_token_id, or pad_id if not found.
    """
    with open(gf_token_dict_path, 'rb') as f:
        gf_dict = pickle.load(f)
    vocab_size = max(gf_dict.values()) + 4  # +4 for Geneformer special tokens
    remap = torch.full((vocab_size,), pad_id, dtype=torch.long)
    for gene_name, gf_tok_id in gf_dict.items():
        if gene_name in local_vocab:
            remap[gf_tok_id] = local_vocab[gene_name]
    return remap


####################################
# Alternative FM wrappers
####################################


class scGPTWrapper(nn.Module):
    """Wraps scGPT to produce per-gene hidden states compatible with tripso.

    Lazily imports scGPT so the class definition is safe in any environment;
    ImportError only fires when you actually instantiate this class.

    Parameters
    ----------
    model_dir : str
        Path to scGPT checkpoint directory (vocab.json, args.json, best_model.pt).
    gf_token_dict_path : str
        Path to tripso's geneformer_token_dictionary_may2025.pkl.
    fm_layer_to_quant : int
        Unused; kept for API parity with gfWrapper.
    """

    def __init__(self, model_dir, gf_token_dict_path, fm_layer_to_quant=-1):
        super().__init__()
        try:
            import json
            from pathlib import Path as _Path

            from scgpt.model import TransformerModel
            from scgpt.tokenizer import GeneVocab
            from scgpt.utils import load_pretrained
        except ImportError:
            raise ImportError(
                'scGPT is not installed in this environment. '
                'Install it with: pip install git+https://github.com/bowang-lab/scGPT\n'
                'Only instantiate scGPTWrapper inside the scGPT environment.'
            )
        model_dir = _Path(model_dir)
        pad_token = '<pad>'
        special_tokens = [pad_token, '<cls>', '<eoc>']
        vocab = GeneVocab.from_file(model_dir / 'vocab.json')
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        vocab.set_default_index(vocab[pad_token])
        with open(model_dir / 'args.json') as f:
            cfg = json.load(f)
        self.pad_token_id = vocab[pad_token]
        self.pad_value = cfg['pad_value']
        self.hidden_size = cfg['embsize']
        model = TransformerModel(
            ntoken=len(vocab),
            d_model=cfg['embsize'],
            nhead=cfg['nheads'],
            d_hid=cfg['d_hid'],
            nlayers=cfg['nlayers'],
            nlayers_cls=cfg.get('n_layers_cls', 3),
            n_cls=1,
            vocab=vocab,
            dropout=0.0,
            pad_token=pad_token,
            pad_value=self.pad_value,
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=False,
            pre_norm=False,
        )
        load_pretrained(
            model,
            torch.load(model_dir / 'best_model.pt', map_location='cpu'),
            verbose=False,
        )
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        remap = _build_gf_to_local_remap(
            gf_token_dict_path,
            local_vocab=vocab,
            pad_id=self.pad_token_id,
        )
        self.register_buffer('_gf_to_scgpt', remap)

    def forward(self, input_dataset, masking=False, **kwargs):
        device = next(self.model.parameters()).device
        gf_ids = input_dataset['input_ids'].to(device)
        B, L = gf_ids.shape
        clamped = gf_ids.clamp(min=0, max=self._gf_to_scgpt.shape[0] - 1)
        scgpt_ids = self._gf_to_scgpt[clamped]
        if 'counts' in input_dataset and input_dataset['counts'] is not None:
            values = input_dataset['counts'].to(device).float()
            if values.shape[1] > L:
                values = values[:, :L]
            elif values.shape[1] < L:
                values = F.pad(values, (0, L - values.shape[1]), value=self.pad_value)
        else:
            values = torch.full(
                (B, L), self.pad_value, dtype=torch.float32, device=device
            )
        src_key_padding_mask = scgpt_ids.eq(self.pad_token_id)
        with torch.no_grad():
            gene_emb = self.model._encode(
                scgpt_ids, values, src_key_padding_mask=src_key_padding_mask
            )
        return {'gene_emb': gene_emb}


class TahoeWrapper(nn.Module):
    """Wraps Tahoe-x1 to produce per-gene hidden states compatible with tripso.

    Loads directly from a safetensors checkpoint — no llm-foundry or tahoe_x1
    package required.  Only safetensors (pip install safetensors) is needed.

    Parameters
    ----------
    safetensors_path : str
        Path to model.safetensors.
    vocab_path : str
        Path to vocab.json from the same model directory.
    gf_token_dict_path : str
        Path to tripso's geneformer_token_dictionary_may2025.pkl.
    n_heads : int
        Number of attention heads. 8 for 70m (d_model=512, head_dim=64).
    fm_layer_to_quant : int
        Unused; kept for API parity with gfWrapper.
    """

    def __init__(
        self,
        safetensors_path,
        vocab_path,
        gf_token_dict_path,
        n_heads=8,
        fm_layer_to_quant=-1,
    ):
        super().__init__()
        from ..Utils.tahoe_utils import load_tahoe_from_safetensors

        model, gene2id = load_tahoe_from_safetensors(
            safetensors_path=safetensors_path,
            vocab_path=vocab_path,
            n_heads=n_heads,
        )
        self._model = model
        self.hidden_size = model.d_model
        self._pad_id_tahoe = gene2id.get('<pad>', 0)
        remap = _build_gf_to_local_remap(
            gf_token_dict_path,
            local_vocab=gene2id,
            pad_id=self._pad_id_tahoe,
        )
        self.register_buffer('_gf_to_tahoe', remap)

    def forward(self, input_dataset, masking=False, **kwargs):
        device = next(self._model.parameters()).device
        gf_ids = input_dataset['input_ids'].to(device)
        clamped = gf_ids.clamp(min=0, max=self._gf_to_tahoe.shape[0] - 1)
        tahoe_ids = self._gf_to_tahoe[clamped]
        with torch.no_grad():
            hidden = self._model(tahoe_ids)
        return {'gene_emb': hidden}


class StateWrapper(nn.Module):
    """Wraps STATE SE to produce per-gene hidden states compatible with tripso.

    Lazily imports arc-state so the class definition is safe in any environment;
    ImportError only fires when you actually instantiate this class.

    Parameters
    ----------
    checkpoint_path : str
        Path to STATE SE .ckpt file (e.g. SE-600M/se600m_epoch15.ckpt).
    gf_token_dict_path : str
        Path to tripso's geneformer_token_dictionary_may2025.pkl.
    protein_embeddings_path : str or None
        Path to protein_embeddings.pt. If None, STATE loads from the checkpoint.
    fm_layer_to_quant : int
        Unused; kept for API parity with gfWrapper.
    """

    def __init__(
        self,
        checkpoint_path,
        gf_token_dict_path,
        protein_embeddings_path=None,
        fm_layer_to_quant=-1,
    ):
        super().__init__()
        try:
            from state.emb.inference import Inference
        except ImportError:
            raise ImportError(
                'arc-state is not installed in this environment. '
                'Install it with: pip install arc-state\n'
                'Only instantiate StateWrapper inside the STATE environment.'
            )
        protein_embeds = None
        if protein_embeddings_path is not None:
            protein_embeds = torch.load(
                protein_embeddings_path, map_location='cpu', weights_only=False
            )
        inferer = Inference(cfg=None, protein_embeds=protein_embeds)
        inferer.load_model(checkpoint_path)
        self._model = inferer.model
        self._protein_embeds = inferer.protein_embeds
        # STATE SE checkpoints load in bfloat16; keep the model in bf16 for the
        # forward pass (lower memory / faster matmuls). forward() casts inputs
        # to the model dtype and casts the output back to float32, since the
        # rest of the tripso pipeline (and numpy, which has no bf16) is float32.
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False
        self.hidden_size = self._model.d_model
        self._esm_dim = 5120
        with open(gf_token_dict_path, 'rb') as f:
            gf_dict = pickle.load(f)
        vocab_size = max(gf_dict.values()) + 4
        cache = torch.zeros(vocab_size, self._esm_dim)
        for gene_name, gf_tok_id in gf_dict.items():
            if gf_tok_id < vocab_size and gene_name in self._protein_embeds:
                emb = self._protein_embeds[gene_name]
                cache[gf_tok_id] = (
                    emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
                )
        self.register_buffer('_protein_cache', cache)

    def forward(self, input_dataset, masking=False, **kwargs):
        device = self._model.device
        gf_ids = input_dataset['input_ids'].to(device)
        B, L = gf_ids.shape
        clamped = gf_ids.clamp(min=0, max=self._protein_cache.shape[0] - 1)
        model_dtype = next(self._model.parameters()).dtype
        protein_emb = self._protein_cache[clamped].to(device=device, dtype=model_dtype)
        with torch.no_grad():
            # Match StateEmbeddingModel: L2-normalise protein embeddings, prepend
            # the cls token (which lives in the 5120-dim token space), THEN project
            # to d_model via the encoder with the sqrt(d_model) scaling, and only
            # then run the transformer. See state/emb/nn/model.py:228-231 and :295.
            protein_emb = F.normalize(protein_emb, dim=2)
            cls = self._model.cls_token.to(protein_emb.dtype).expand(B, -1).unsqueeze(1)
            with_cls = torch.cat([cls, protein_emb], dim=1)
            projected = self._model.encoder(with_cls) * math.sqrt(self._model.d_model)
            hidden = self._model.transformer_encoder(projected)
        # Drop the prepended cls token so the per-gene embeddings line up 1:1 with
        # input_ids (build_gp_input_matrix multiplies gene_emb by an input_ids-shaped
        # mask). The cls still served its purpose: genes attended to it in the encoder.
        hidden = hidden[:, 1:, :]
        # Model runs in bf16; cast the output back to float32 so downstream
        # averaging and the HuggingFace Dataset.from_dict write (Arrow/numpy,
        # which have no bf16 dtype) work.
        return {'gene_emb': hidden.float()}


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


class fmBaseline(gpTransformerBase):
    """Foundation-model baseline using gpAverager (AverageNonZero) GP encoding.

    Supports fm_encoder_pkg in {'geneformer', 'geneformer_2021', 'from_scratch',
    'scgpt', 'tahoe', 'state'}.

    For 'state': also pass state_hidden_size=<int> and optionally
    protein_embeddings_path=<str>.
    For 'tahoe': fm_encoder_name='repo_id:model_size', e.g. 'tahoebio/Tahoe-x1:70m'.
    For 'scgpt': fm_encoder_name=path to checkpoint dir.
    """

    def __init__(
        self, fm_encoder_pkg='geneformer', fm_encoder_name='gf-6L-30M-i2048', **kwargs
    ):
        super().__init__(
            fm_encoder_pkg=fm_encoder_pkg,
            fm_encoder_name=fm_encoder_name,
            **kwargs,
        )

        self.multi_gp_encoder = gpAverager(
            database=self.gpdb,
            do_ensembl_conversion=self.do_ensembl_conversion,
            gene_token_path=self.gene_token_path,
            gene_name_path=self.gene_name_path,
            gp_latent_size=self.gp_latent_size,
            n_blocks=self.n_blocks,
            num_heads=self.num_heads,
            mgm_mask_ratio=self.mgm_mask_ratio,
            gp_inputs=self.gp_inputs,
            use_flash=False,
            model_type='Mean',
            learn_new_gp=False,
            use_pos_emb=False,
            use_l2_norm=False,
            attn_dropout=0.0,
            init_sparsity=0.0,
            fm_model_input_size=self.fm_model_input_size,
        )

    def get_last_self_attn(self, input_dataset, gp):
        warnings.warn(
            f'fmBaseline (fm_encoder_pkg={self.fm_encoder_pkg}): '
            'attention matrices unavailable; returning cosine similarity instead.'
        )
        gp_idx = self.gp_inputs.index(gp)
        emb_out = self.gf_wrapper(input_dataset)
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
