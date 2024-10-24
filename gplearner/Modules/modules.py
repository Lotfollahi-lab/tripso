# Code from DINO
# https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
# Accessed 18.10.2023
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn import flash_attn_func
from torch import Tensor

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

except ModuleNotFoundError:
    print('flex_attention not available')


from gplearner.Utils.utils import (
    RMSNorm,
    apply_rotary_emb,
    drop_path,
    mlm_mask_generator,
    trunc_normal_,
)

######################################################################
# Flex attention
######################################################################

# Compile flex_attention function if available
try:
    flex_attention = torch.compile(
        flex_attention, dynamic=False, mode='max-autotune-no-cudagraphs'
    )
except NameError:
    pass

######################################################################
# Differential transformer
# from https://github.com/microsoft/unilm/blob/master/
# Diff-Transformer/multihead_flashdiff_2.py
# Accessed 10/10/2024
######################################################################


def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadFlashDiff2(nn.Module):
    """
    DiffAttn implemented with FlashAttention,
    for packages that does not support different qk/v dimensions
    e.g., flash-attention (https://github.com/Dao-AILab/flash-attention)
    """

    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        # args:
        model_parallel_size,
        decoder_kv_attention_heads,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads // model_parallel_size
        self.num_kv_heads = (
            decoder_kv_attention_heads // model_parallel_size
            if decoder_kv_attention_heads is not None
            else num_heads // model_parallel_size
        )
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads  # // 2 (otherwise cant reshape)
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)

    def forward(
        self,
        x,
        rel_pos,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2, self.head_dim)

        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        attn11 = flash_attn_func(q1, k1, v1, causal=True)
        attn12 = flash_attn_func(q1, k1, v2, causal=True)
        attn1 = torch.cat([attn11, attn12], dim=-1)

        attn21 = flash_attn_func(q2, k2, v1, causal=True)
        attn22 = flash_attn_func(q2, k2, v2, causal=True)
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn


######################################################################
# Dino
######################################################################


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        # act_layer = nn.ReLU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        input_dim,
        num_heads=1,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_flash=False,
        use_flex=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_flash = use_flash
        self.use_flex = use_flex

    def forward(self, x, attn_mask, return_attention, block_mask):
        if self.use_flash:
            return_attention = False
            # do masking here
            x = x * attn_mask.unsqueeze(-1)

        # Attention mask is 0 for padding tokens (no attention)
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_flash:
            attn_out = flash_attn_func(q, k, v)  # (batch_size, seqlen, nheads, headdim)

            # with torch.backends.cuda.sdp_kernel(enable_flash=True):
            #     # from https://discuss.pytorch.org/t/flash-attention/174955/14
            #     attn_out = F.scaled_dot_product_attention(
            #         q,
            #         k,
            #         v,
            #         # pytorch flash attention does not support mask
            #         scale=self.scale,
            #         dropout_p=0.0,
            #     )
            #     # if scale is None, default is 1/sqrt(dim)

            x = attn_out.reshape(B, N, C)

        elif self.use_flex:
            attn_out = flex_attention(
                q,
                k,
                v,
                # score_mod = score_mod,
                block_mask=block_mask,
            )

        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

            mask = rearrange(attn_mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(attn.dtype).max

            # Repeat the mask for each head
            mask = repeat(mask, 'b j -> b h () j', h=self.num_heads)

            # Apply the mask to the attention scores
            attn.masked_fill_(mask == 0, max_neg_value)

            # Apply softmax to get attention weights
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # Calculate the weighted sum of values
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention is False:
            attn = None

        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=0.5,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        # act_layer=nn.ReLU,
        norm_layer=nn.LayerNorm,
        use_flash=False,
        use_flex=False,
        seq_len=2048,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim,
            input_dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_flash=use_flash,
            use_flex=use_flex,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, attn_mask, return_attention, block_mask):
        y, attn = self.attn(
            self.norm1(x),
            attn_mask=attn_mask,
            return_attention=return_attention,
            block_mask=block_mask,
        )  # attn is None when using flash attention
        # y = self.attn(self.norm1(x), attn_mask=attn_mask)

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn


# --------------------------------------------------------------------
# Positional encoding
# --------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """

        pe = self.pe[:, : x.size(1)]  # (1, seq_len, 512)
        x = x + pe  # (batch, seq_len, 512)

        return self.dropout(x)


class LearntPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)
        # Register a buffer for position IDs,
        # precomputed for the maximum sequence length
        position_ids = torch.arange(max_seq_length).expand((1, -1))
        self.register_buffer('position_ids', position_ids)

    def forward(self, x, position_ids=None):
        # TODO: register buffer
        if position_ids is None:
            position_ids = self.position_ids[:, : x.size(1)]
        position_ids = position_ids.expand(x.size(0), -1)

        return x + self.position_embeddings(position_ids)


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py#L450

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles

    From https://pytorch.org/torchtune/0.1/_modules/torchtune/modules/
    position_embeddings.html#RotaryPositionalEmbeddings

    accessed 11/10/2024

    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/
    # 797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer('theta', theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum('i, j -> ij', seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        # modified to return cos and sin separately
        # cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        # self.register_buffer("cache", cache, persistent=False)

        cache_cos = torch.cos(idx_theta)
        cache_sin = torch.sin(idx_theta)

        # convert to bf16
        cache_cos = cache_cos.to(torch.bfloat16)
        cache_sin = cache_sin.to(torch.bfloat16)

        self.register_buffer('cache_cos', cache_cos, persistent=False)
        self.register_buffer('cache_sin', cache_sin, persistent=False)

    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        pass

    # pytorch implementation returns tensor with RoPE already applied


# --------------------------------------------------------------------
# Main encoder class
# --------------------------------------------------------------------


class gpTransformerEncoder(nn.Module):
    """GP Transformer main block"""

    def __init__(
        self,
        n_gp_tokens,
        depth,  # number of blocks
        mlm_masking_prob,
        embed_dim=512,
        num_heads=1,
        mlp_ratio=0.5,  # factor of how much MLP reduces layer size
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,  # dropout on position embedding
        attn_drop_rate=0.0,  # passed to attention module (attn_drop)
        drop_path_rate=0.0,  # no effect if only 1 block
        norm_layer=nn.LayerNorm,
        use_pos_emb='sin_cos',
        vocab_size=None,
        use_flash=False,
        use_diffl=False,  # use differential transformer
        seq_len=2048,
        use_flex=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_pos_emb = use_pos_emb
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.use_flex = use_flex
        self.num_heads = num_heads

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.mask_generator = mlm_mask_generator(
            padding_token=1,
            mask_token=0,
            n_tokens=n_gp_tokens,
            masking_prob=mlm_masking_prob,
            randomize_prob=0.1,
            no_change_prob=0.1,
            no_mask_tokens=[None],
        )

        self.use_diffl = use_diffl
        if use_diffl:
            if num_heads == 1:
                raise ValueError('Differential transformer requires num_heads > 1')
            self.blocks = nn.ModuleList(
                [
                    MultiheadFlashDiff2(
                        embed_dim,
                        depth,
                        num_heads,
                        model_parallel_size=2,
                        decoder_kv_attention_heads=num_heads,
                    )
                ]
            )

        else:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        use_flash=use_flash,
                        use_flex=use_flex,
                        seq_len=seq_len + 1,  # +1 for cls
                    )
                    for i in range(depth)
                ]
            )

        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.cls_token, std=0.02)

        # Decoder for masked language modelling
        # self.decoder = nn.Linear(embed_dim, n_gp_tokens, bias=False)
        self.vocab_size = vocab_size
        if self.vocab_size is None:
            self.decoder = nn.Linear(embed_dim, n_gp_tokens, bias=False)
        else:
            self.decoder = nn.Linear(embed_dim, self.vocab_size, bias=False)

        self.decoder_bias = nn.Parameter(torch.zeros(n_gp_tokens))

        self.apply(self._init_weights)

        if self.use_pos_emb == 'sin_cos':
            self.pos_embed = PositionalEncoding(
                d_model=embed_dim,
                dropout=drop_rate,
                max_len=seq_len + 1,  # 2048
            )
        elif self.use_pos_emb == 'learned':
            self.pos_embed = LearntPositionalEncoding(
                d_model=embed_dim, max_seq_length=seq_len + 1
            )

        if self.use_diffl:
            # Differential transformer uses rotatory embeddings
            self.pos_embed = nn.Identity()
            self.rope = RotaryPositionalEmbeddings(
                dim=embed_dim // num_heads // 2,
                max_seq_len=seq_len + 1,
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_gene_masking(self, x, gene_labels):
        x = x.clone()
        gene_labels = gene_labels.clone()

        full_mask, mask, random_mask = self.mask_generator(gene_labels)

        # Apply the mask to the target tensor
        # if mask = 1, we want to 0 out the token embedding
        # but keep the label for loss calculation
        x = torch.where(mask.unsqueeze(-1), self.mask_emb.expand_as(x), x)
        # x = x.masked_fill(mask.unsqueeze(-1), 0)

        # Add random tokens to the masked positions
        random_tokens = torch.randn(x.shape, device=x.device, dtype=x.dtype)

        x[random_mask] = random_tokens[random_mask]

        # Replace unmasked indices with -100 in the labels
        # since we only compute loss on masked tokens
        gene_labels[~full_mask] = -100

        return x, gene_labels

    def prepare_tokens(self, x, gene_labels):
        B = x.shape[0]  # batch size

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (512, 1, 512)

        x = torch.cat((cls_tokens, x), dim=1)

        # add dummy label for cls
        cls_label = torch.full(
            (gene_labels.shape[0], 1),
            -100,
            dtype=gene_labels.dtype,
            device=gene_labels.device,
        )  # Create a column of -100 values

        gene_labels = torch.cat((cls_label, gene_labels), dim=1)

        # add positional encoding to each token
        if self.use_pos_emb is not None:
            x = self.pos_embed(x)

        return self.pos_drop(x), gene_labels

    def forward(
        self,
        x,
        gene_labels,
        masking,
        attn_mask,
        return_attention,
        return_gene_embeddings=False,
    ):
        # Random masking:
        if masking:
            x, gene_labels = self.random_gene_masking(x, gene_labels)

        # Prepare tokens for transformer
        x, gene_labels = self.prepare_tokens(x, gene_labels)

        # Optionally prepare mask to be shared across heads and layers
        if self.use_flex:
            # B, H, Q_LEN, KV_LEN
            # convert attention mask to bool
            attn_mask = attn_mask.bool()
            attn_mask = (
                attn_mask.unsqueeze(1)
                .unsqueeze(2)
                .expand(x.shape[0], self.num_heads, x.shape[1], x.shape[1])
            )

            torch._dynamo.config.optimize_ddp = False
            torch._dynamo.config.suppress_errors = True

            def padding(b, h, q_idx, kv_idx):
                # print('q_idx', q_idx)
                # return attn_mask #[b, q_idx]
                return attn_mask[b, h, q_idx, kv_idx]

            block_mask = create_block_mask(
                padding,
                B=x.shape[0],
                H=self.num_heads,
                Q_LEN=x.shape[1],
                KV_LEN=x.shape[1],
                BLOCK_SIZE=x.shape[1],
                # _compile=True
            )

        else:
            block_mask = None

        for blk in self.blocks:
            if self.use_diffl:
                x = blk(
                    x,
                    rel_pos=(self.rope.cache_cos, self.rope.cache_sin),
                    attn_mask=attn_mask,
                )
                attn = None
            else:
                x, attn = blk(
                    x,
                    attn_mask=attn_mask,
                    return_attention=return_attention,
                    block_mask=block_mask,
                )

        x = self.norm(x)

        token = x[:, 0]  # equivalent to x[:, 0, :] = return <GP> token

        logits_lm = self.decoder(x)

        output = {'cls': token, 'logits_lm': logits_lm, 'gene_labels': gene_labels}

        if attn is not None:
            #  returns full attention matrix not just CLS
            output['attention'] = attn

        if return_gene_embeddings:
            output['gene_embeddings'] = x[:, 1:, :]

        return output

    def get_intermediate_layers(self, x, gene_labels, n=1):
        x, gene_labels = self.prepare_tokens(x, gene_labels)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class gpTransformerEncoderWithPrompt(gpTransformerEncoder):
    def __init__(self, **kwargs):
        super()._init_(**kwargs)

    def random_gene_masking(self, x, gene_labels, unmask_last_n=0):
        x = x.clone()
        gene_labels = gene_labels.clone()

        # Ensure the last n tokens are never masked
        if unmask_last_n > 0:
            # Create a mask to prevent masking of the last n tokens
            protect_mask = torch.zeros_like(gene_labels, dtype=torch.bool)
            protect_mask[:, -unmask_last_n:] = True

        full_mask, mask, random_mask = self.mask_generator(gene_labels)

        # Apply the protect_mask to ensure last n tokens are not masked
        if unmask_last_n > 0:
            full_mask &= ~protect_mask
            mask &= ~protect_mask
            random_mask &= ~protect_mask

        # Apply the mask to the target tensor
        # if mask = 1, we want to 0 out the token embedding
        # but keep the label for loss calculation
        x = x.masked_fill(mask.unsqueeze(-1), 0)

        # Add random tokens to the masked positions
        random_tokens = torch.randn(x.shape, device=x.device)

        # x = x.masked_fill(random_mask.unsqueeze(-1), random_tokens[random_mask])
        x[random_mask] = random_tokens[random_mask]

        # Replace unmasked indices with -100 in the labels
        # since we only compute loss on masked tokens
        gene_labels[~full_mask] = -100

        return x, gene_labels

    def forward(
        self,
        x,
        gene_labels,
        masking,
        attn_mask,
        return_attention,
        return_gene_embeddings=False,
        num_virtual_tokens=0,
        using_gp_specific_token=False,
    ):
        # Random masking:
        if masking:
            x, gene_labels = self.random_gene_masking(
                x, gene_labels, unmask_last_n=num_virtual_tokens
            )

        # Prepare tokens for transformer
        x, gene_labels = self.prepare_tokens(x, gene_labels)

        #     # Extract the <cls> token
        #     # shapes indicate shape of line below
        #     # Shape: [batch_size, 1, feature_dim]
        #     cls_token = x[:, :1, :]
        #     # Extract the virtual tokens from the end
        #     # Shape: [batch_size, num_virtual_tokens, feature_dim]
        #     virtual_tokens = x[:, -num_virtual_tokens:, :]
        #     # Extract the gene tokens from the remaining part
        #     # [batch_size, sequence_length - num_virtual_tokens - 1, feature_dim]
        #     gene_tokens = x[:, 1:-num_virtual_tokens, :]
        #     # Concatenate the parts in the required order:
        #     # [<cls>, <virtual tokens>, <gene_tokens>]
        #     # Shape: [batch_size, sequence_length, feature_dim]
        #     x = torch.cat([cls_token, virtual_tokens, gene_tokens], dim=1)

        #     # And the same for gene labels
        #     cls_label = gene_labels[:, :1]  # Shape: [batch_size, 1]
        #     # Shape: [batch_size, num_virtual_tokens]
        #     virtual_labels = gene_labels[:, -num_virtual_tokens:]
        #     # Shape: [batch_size, sequence_length - num_virtual_tokens - 1]
        #     g_labels = gene_labels[:, 1:-num_virtual_tokens]
        #     # Shape: [batch_size, sequence_length]
        #     gene_labels = torch.cat([cls_label, virtual_labels, g_labels], dim=1)

        #     # And attention mask
        #     cls_mask = attn_mask[:, :1]  # Shape: [batch_size, 1]
        #     # Shape: [batch_size, num_virtual_tokens]
        #     virtual_mask = attn_mask[:, -num_virtual_tokens:]
        #     # Shape: [batch_size, sequence_length - num_virtual_tokens - 1]
        #     g_mask = attn_mask[:, 1:-num_virtual_tokens]
        #     # Shape: [batch_size, sequence_length]
        #     attn_mask = torch.cat([cls_mask, virtual_mask, g_mask], dim=1)

        for blk in self.blocks:
            x, attn = blk(x, attn_mask=attn_mask, return_attention=return_attention)

        x = self.norm(x)

        token = x[:, 0]  # equivalent to x[:, 0, :] = return <GP> token

        logits_lm = self.decoder(x)

        output = {'cls': token, 'logits_lm': logits_lm, 'gene_labels': gene_labels}

        if attn is not None:
            #  returns full attention matrix not just CLS
            output['attention'] = attn

        if return_gene_embeddings:
            output['gene_embeddings'] = x[:, 1:-num_virtual_tokens, :]

        if using_gp_specific_token:
            output['gp_virtual_tokens'] = x[
                :, -num_virtual_tokens : -int(num_virtual_tokens / 2), :
            ]
            output['shared_virtual_tokens'] = x[:, -int(num_virtual_tokens / 2) :, :]
        else:
            output['shared_virtual_tokens'] = x[:, -num_virtual_tokens:, :]

        return output


class PretrainedEmbeddings(nn.Module):
    '''
    BertEmbedding style class for exploring LIG
    Initialize nn.Embedding directly from embeddings
    which are output from another part of the model
    # NOT USED
    '''

    def __init__(
        self,
        pretrained_emb,
        pretrained_pos_emb,
        vocab_size,
        embedding_dim,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim).from_pretrained(
            pretrained_emb
        )
        self.position_embeddings = pretrained_pos_emb

    def forward(self):
        embeddings = self.word_embeddings + self.position_embeddings
        return embeddings


class PromptEncoder(torch.nn.Module):
    """
    The prompt encoder network that is used to generate the
    virtual token embeddings for p-tuning.

    Adapted from
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/p_tuning/model.py

    Accessed 26.06.2024

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) --
            The embedding layer of the prompt encoder.
        - **mlp_head** (`torch.nn.Sequential`) --
            The MLP head of the prompt encoder if `inference_mode=False`.
        - **lstm_head** (`torch.nn.LSTM`) --
            The LSTM head of the prompt encoder if `inference_mode=False` and
        `encoder_reparameterization_type="LSTM"`.
        - **token_dim** (`int`) --
            The hidden embedding dimension of the base transformer model.
        - **input_size** (`int`) -- The input size of the prompt encoder.
        - **output_size** (`int`) -- The output size of the prompt encoder.
        - **hidden_size** (`int`) -- The hidden size of the prompt encoder.
        - **total_virtual_tokens** (`int`): The total number of virtual tokens of the
        prompt encoder.
        - **encoder_type** --> here MLP only (recommended)

    Input shape: (`batch_size`, `total_virtual_tokens`)

    Output shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(
        self,
        token_dim: int,
        encoder_hidden_size: int,
        num_virtual_tokens: int,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.input_size = token_dim
        self.output_size = token_dim
        self.hidden_size = encoder_hidden_size
        self.total_virtual_tokens = num_virtual_tokens

        # embedding
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.token_dim)

        layers = [
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.output_size),
        ]
        self.mlp_head = torch.nn.Sequential(*layers)

    def forward(self, indices):
        input_embeds = self.embedding(indices)

        output_embeds = self.mlp_head(input_embeds)

        return output_embeds


if __name__ == '__main__':
    print('Testing the model')
    model = gpTransformerEncoder(
        n_gp_tokens=5,
        depth=1,
        mlm_masking_prob=0.4,
        embed_dim=32,
        seq_len=16,
        num_heads=8,
        use_flex=True,
    )

    # for n, p in model.named_parameters():
    #     print(n, p.shape)

    x = torch.randn(1, 15, 32, dtype=torch.float16)
    gene_labels = torch.randint(0, 10, (1, 15), dtype=torch.long)
    attn_mask = torch.ones(1, 16, dtype=torch.float16)

    # move everything to cuda
    x = x.to('cuda')
    gene_labels = gene_labels.to('cuda')
    attn_mask = attn_mask.to('cuda')
    model = model.to('cuda')

    # make model fp16 for flash attention
    model = model.half()

    out = model(
        x,
        gene_labels,
        masking=True,
        attn_mask=attn_mask,
        return_attention=False,
    )

    for k, v in out.items():
        print(k, v.shape)

    print(out['gene_labels'])
