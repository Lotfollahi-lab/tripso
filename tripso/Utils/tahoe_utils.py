"""
Minimal pure-PyTorch reimplementation of the Tahoe-x1 transformer encoder.

Loads directly from safetensors — no llm-foundry or tahoe_x1 package required.

Architecture (inferred from checkpoint keys/shapes):
  gene_encoder.embedding   nn.Embedding(vocab_size, d_model)
  gene_encoder.enc_norm    nn.LayerNorm(d_model)
  transformer_encoder:
    layers.i:
      norm1                nn.LayerNorm(d_model)   pre-norm before attention
      self_attn.Wqkv       nn.Linear(d_model, 3*d_model)  fused QKV
      self_attn.out_proj   nn.Linear(d_model, d_model)
      norm2                nn.LayerNorm(d_model)   pre-norm before FFN
      up_proj              nn.Linear(d_model, ffn_dim)
      down_proj            nn.Linear(ffn_dim, d_model)
    norm                   nn.LayerNorm(d_model)   final norm

Expression / flag inputs from the original model are unused here (zero-shot,
gene-ID-only mode). Position 0 of the output corresponds to the first token
in the input sequence (no CLS prepended by this model).

Usage
-----
    from tripso.Utils.tahoe_utils import load_tahoe_from_safetensors
    model, gene2id = load_tahoe_from_safetensors(
        safetensors_path='/path/to/model.safetensors',
        vocab_path='/path/to/vocab.json',
        n_heads=8,   # 8 for 70m, check for larger sizes
    )
    gene_ids = torch.tensor([[1, 2, 3, 0]])    # (B, L), 0 = pad
    hidden   = model(gene_ids)                 # (B, L, d_model)
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Architecture ──────────────────────────────────────────────────────────────


class _TahoeEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.Wqkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.up_proj = nn.Linear(d_model, ffn_dim)
        self.down_proj = nn.Linear(ffn_dim, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        qkv = self.Wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        x = self.out_proj(attn) + residual
        # Pre-norm FFN (GELU)
        residual = x
        x = self.norm2(x)
        x = F.gelu(self.up_proj(x))
        x = self.down_proj(x) + residual
        return x


class TahoeLite(nn.Module):
    """Minimal Tahoe-x1 encoder: gene IDs → per-token hidden states.

    No expression values, no flag tokens, no CLS token.
    Output shape: (B, L, d_model).
    """

    def __init__(
        self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, ffn_dim: int
    ):
        super().__init__()
        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.enc_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [_TahoeEncoderLayer(d_model, n_heads, ffn_dim) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, gene_ids: torch.Tensor) -> torch.Tensor:
        x = self.gene_embedding(gene_ids)  # (B, L, d_model)
        x = self.enc_norm(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


# ── Weight loading ────────────────────────────────────────────────────────────


def _remap_key(st_key: str) -> str | None:
    """Map a safetensors key (with 'model.' prefix) to a TahoeLite state-dict key.

    Returns None for keys that are not part of TahoeLite (expression encoder,
    decoders, flag encoder, etc.).
    """
    # strip top-level 'model.' prefix
    if st_key.startswith('model.'):
        key = st_key[len('model.') :]
    else:
        return None

    # gene embedding + enc_norm
    if key.startswith('gene_encoder.embedding.'):
        return 'gene_embedding.' + key[len('gene_encoder.embedding.') :]
    if key.startswith('gene_encoder.enc_norm.'):
        return 'enc_norm.' + key[len('gene_encoder.enc_norm.') :]

    # transformer layers
    if key.startswith('transformer_encoder.layers.'):
        rest = key[len('transformer_encoder.layers.') :]
        layer_idx, subkey = rest.split('.', 1)

        if subkey.startswith('self_attn.Wqkv.'):
            subkey = 'Wqkv.' + subkey[len('self_attn.Wqkv.') :]
        elif subkey.startswith('self_attn.out_proj.'):
            subkey = 'out_proj.' + subkey[len('self_attn.out_proj.') :]
        # norm1, norm2, up_proj, down_proj keys are unchanged

        return f'layers.{layer_idx}.{subkey}'

    # final norm
    if key.startswith('transformer_encoder.norm.'):
        return 'norm.' + key[len('transformer_encoder.norm.') :]

    return None  # expression_encoder, decoders, flag_encoder — skip


def _infer_arch(safetensors_path: str) -> dict:
    """Read architecture hyperparameters from checkpoint shapes."""
    from safetensors import safe_open

    with safe_open(str(safetensors_path), framework='pt', device='cpu') as f:
        keys = list(f.keys())
        vocab_size, d_model = f.get_tensor('model.gene_encoder.embedding.weight').shape
        ffn_dim = f.get_tensor(
            'model.transformer_encoder.layers.0.up_proj.weight'
        ).shape[0]

    layer_ids = {
        int(k.split('transformer_encoder.layers.')[1].split('.')[0])
        for k in keys
        if 'transformer_encoder.layers.' in k
    }
    n_layers = max(layer_ids) + 1

    # head_dim=64 is standard for BERT-style models; parameterised via n_heads
    # caller may override via the n_heads argument to load_tahoe_from_safetensors
    return dict(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, ffn_dim=ffn_dim
    )


def load_tahoe_from_safetensors(
    safetensors_path: str,
    vocab_path: str,
    n_heads: int = 8,
) -> tuple:
    """Load Tahoe-x1 from a safetensors checkpoint with no llm-foundry dependency.

    Parameters
    ----------
    safetensors_path : str
        Path to model.safetensors (downloaded from tahoebio/Tahoe-x1 on HF).
    vocab_path : str
        Path to vocab.json (gene_name/ensembl_id → int token id).
    n_heads : int
        Number of attention heads. Default 8 works for the 70m model
        (d_model=512, head_dim=64). Adjust for other sizes if needed.

    Returns
    -------
    model : TahoeLite  (eval mode, all parameters frozen)
    gene2id : dict  {gene_name_or_ensembl_id: int}
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError('safetensors is not installed. pip install safetensors')

    arch = _infer_arch(safetensors_path)
    model = TahoeLite(
        vocab_size=arch['vocab_size'],
        d_model=arch['d_model'],
        n_layers=arch['n_layers'],
        n_heads=n_heads,
        ffn_dim=arch['ffn_dim'],
    )

    # Load weights from safetensors, remapping keys
    state_dict = {}
    with safe_open(str(safetensors_path), framework='pt', device='cpu') as f:
        for st_key in f.keys():
            mapped = _remap_key(st_key)
            if mapped is not None:
                state_dict[mapped] = f.get_tensor(st_key)

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        raise RuntimeError(f'Missing keys after loading Tahoe weights: {missing}')

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    with open(vocab_path) as fh:
        gene2id = json.load(fh)

    return model, gene2id
