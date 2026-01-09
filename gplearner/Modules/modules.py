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

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..Utils.utils import (
    drop_path,
    mlm_mask_generator,
    trunc_normal_,
)

######################################################################
# Extra
######################################################################


class ReduceDim(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        if dim_in == dim_out:
            self.dim_red_layer = nn.Identity()
        else:
            self.dim_red_layer = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.GELU(),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        return self.dim_red_layer(x)


######################################################################
# Dino
######################################################################


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Applies stochastic depth when used in main path of residual blocks.

    Parameters
    ----------
    drop_prob : float or None, optional
        Probability of dropping a path (default: None).
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
        self.attn_drop_rate = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_flash = use_flash
        self.use_flex = use_flex

    def forward(self, x, attn_mask, return_attention):
        # Attention mask is 0 for padding tokens (no attention)
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_flash:
            # attn_out = flash_attn_func(q, k, v)
            # # (batch_size, seqlen, nheads, headdim)
            # x = attn_out.reshape(B, N, C)

            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                # from https://discuss.pytorch.org/t/flash-attention/174955/14
                mask = attn_mask.unsqueeze(1).unsqueeze(2)
                mask = mask.expand(-1, self.num_heads, x.shape[1], x.shape[1])

                # match x dtype
                mask = mask.bool()

                attn_out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    # pytorch flash attention does not support mask
                    scale=self.scale,
                    dropout_p=self.attn_drop_rate,
                    attn_mask=mask,
                )  # if scale is None, default is 1/sqrt(dim)

                # attn_out shape: (B, num_heads, seq_len, emb_size//num_heads)
                # transpose so that it is (B, seq_len, num_heads, emb_size//num_heads)
                x = attn_out.transpose(1, 2).reshape(B, N, C)

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

    def forward(self, x, attn_mask, return_attention):
        y, attn = self.attn(
            self.norm1(x),
            attn_mask=attn_mask,
            return_attention=return_attention,
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
        """Forward pass with positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns
        -------
        torch.Tensor
            Input with added positional encoding and dropout applied.
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


# --------------------------------------------------------------------
# Main encoder class
# --------------------------------------------------------------------


class gpTransformerEncoder(nn.Module):
    """Gene Program Transformer Encoder.

    Building block transformer encoder for the GPformer model, supporting
    masked language modeling, multiple attention mechanisms, and flexible
    positional encodings.

    Parameters
    ----------
    n_gp_tokens : int
        Number of gene program tokens in the vocabulary.
    depth : int
        Number of transformer blocks.
    mlm_masking_prob : float
        Probability of masking tokens for masked language modeling.
    embed_dim : int, optional
        Embedding dimension (default: 512).
    num_heads : int, optional
        Number of attention heads (default: 1).
    mlp_ratio : float, optional
        Ratio of MLP hidden dimension to embedding dimension (default: 0.5).
    qkv_bias : bool, optional
        Whether to add bias to QKV projection (default: False).
    qk_scale : float or None, optional
        Scale factor for attention scores. If None, uses 1/sqrt(head_dim)
        (default: None).
    drop_rate : float, optional
        Dropout rate for position embeddings (default: 0.0).
    attn_drop_rate : float, optional
        Dropout rate for attention weights (default: 0.0).
    drop_path_rate : float, optional
        Stochastic depth rate (default: 0.0).
    norm_layer : nn.Module, optional
        Normalization layer class (default: nn.LayerNorm).
    use_pos_emb : {'sin_cos', 'learned', 'absolute', None}, optional
        Type of positional embedding. 'sin_cos' uses sinusoidal embeddings,
        'learned' uses learned embeddings, 'absolute' uses BERT-style
        embeddings, None uses no positional encoding (default: 'sin_cos').
    vocab_size : int or None, optional
        Size of vocabulary for decoder. If None, uses n_gp_tokens
        (default: None).
    use_flash : bool, optional
        Whether to use flash attention (default: False).
    seq_len : int, optional
        Maximum sequence length (default: 2048).
    use_l2_norm : bool, optional
        Whether to use L2 normalization instead of LayerNorm (default: False).
    output_dim : int or None, optional
        Output dimension. If None, uses embed_dim. If output_dim != embed_dim,
        an MLP layer is added to project to output_dim (default: None).
    no_mask_tokens : list, optional
        List of token IDs that should not be masked during MLM
        (default: [None]).
    sparsity : float, optional
        Sparsity level for weight initialization, where 0.0 is fully dense
        (default: 0.0).

    Attributes
    ----------
    embed_dim : int
        Stored embedding dimension.
    output_dim : int
        Stored output dimension.
    cls_token : nn.Parameter
        Learnable CLS token.
    mask_emb : nn.Parameter
        Learnable mask embedding for MLM.
    pos_embed : nn.Module
        Positional embedding module.
    blocks : nn.ModuleList
        List of transformer blocks.
    decoder : nn.Linear
        Linear decoder for masked language modeling.
    mask_generator : callable
        Function for generating MLM masks.
    """

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
        seq_len=2048,
        use_l2_norm=False,
        output_dim=None,
        no_mask_tokens=[None],
        sparsity=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_pos_emb = use_pos_emb
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.num_heads = num_heads
        self.use_l2_norm = use_l2_norm

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
            no_mask_tokens=no_mask_tokens,
        )

        if output_dim is None:
            output_dim = embed_dim

        if output_dim is not None and (output_dim != embed_dim):
            self.dim_red = ReduceDim(embed_dim, output_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, output_dim))

        if self.use_pos_emb == 'sin_cos':
            self.pos_embed = PositionalEncoding(
                d_model=output_dim,
                dropout=drop_rate,
                max_len=seq_len + 1,  # 2048
            )
        elif self.use_pos_emb == 'learned':
            self.pos_embed = LearntPositionalEncoding(
                d_model=output_dim, max_seq_length=seq_len + 1
            )
        elif self.use_pos_emb == 'absolute':
            # following Bert
            # https://github.com/huggingface/transformers/blob/main/src/
            # transformers/models/bert/modeling_bert.py#L159
            self.pos_embed = nn.Embedding(seq_len + 1, output_dim)
            trunc_normal_(self.pos_embed.weight, std=0.02)

        self.output_dim = output_dim

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.output_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_flash=use_flash,
                    seq_len=seq_len + 1,  # +1 for cls
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(self.output_dim)
        trunc_normal_(self.cls_token, std=0.02)

        # Decoder for masked language modelling
        self.vocab_size = vocab_size
        decoder_in = self.output_dim
        decoder_out = n_gp_tokens if self.vocab_size is None else self.vocab_size

        self.decoder = nn.Linear(decoder_in, decoder_out, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(n_gp_tokens))

        self.sparsity = sparsity
        if self.sparsity > 0:
            self.apply(self.sparse_init)

        else:
            self.apply(self._init_weights)

    def sparse_init(self, m, std=0.01):
        """Initialize layer weights with sparsity.

        Parameters
        ----------
        m : nn.Module
            Layer to initialize (Linear or LayerNorm).
        std : float, optional
            Standard deviation for non-zero weight values (default: 0.01).

        Notes
        -----
        Uses self.sparsity to determine fraction of elements set to zero.
        """
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                mask = (
                    torch.rand(m.weight.shape) > self.sparsity
                )  # True for non-zero elements
                values = torch.randn(m.weight.shape) * std  # Random normal values
                m.weight.zero_()  # Set all elements to zero
                m.weight[mask] = values[mask]  # Assign only to non-zero positions

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_gene_masking(self, x, gene_labels):
        """Apply random masking to gene tokens for masked language modeling.

        Masks a subset of tokens according to the masking probability, replacing them
        with mask embeddings or random tokens. Updates labels to only compute loss on
        masked tokens.

        Args:
            x (torch.Tensor): Input embeddings of shape
                (batch_size, seq_len, embed_dim).
            gene_labels (torch.Tensor): Gene labels of shape (batch_size,
                seq_len).

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): Masked input embeddings.
                - gene_labels (torch.Tensor): Updated labels with -100
                  for unmasked positions.
        """
        x = x.clone()
        gene_labels = gene_labels.clone()

        full_mask, mask, random_mask = self.mask_generator(gene_labels)

        # Apply the mask to the target tensor
        # if mask = 1, we want to 0 out the token embedding
        # but keep the label for loss calculation
        """Prepare input tokens by adding CLS token and positional encodings.

        Prepends a learnable CLS token to the sequence and adds positional information
        according to the specified encoding type. Also adds a dummy label (-100) for the
        CLS token position.

        Args:
            x (torch.Tensor): Input embeddings of shape
                (batch_size, seq_len, embed_dim).
            gene_labels (torch.Tensor): Gene labels of shape
                (batch_size, seq_len).

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): Token embeddings with CLS and positional
                  encoding, shape (batch_size, seq_len+1, embed_dim).
                - gene_labels (torch.Tensor): Labels with CLS dummy label
                  prepended, shape (batch_size, seq_len+1).
        """
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
        """Prepare input tokens by adding CLS token and positional encodings.

        Args:
            x (torch.Tensor): Input embeddings of shape
                (batch_size, seq_len, embed_dim).
            gene_labels (torch.Tensor): Gene labels of shape (batch_size,
                seq_len).

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): Token embeddings with CLS and positional encoding,
                  shape (batch_size, seq_len+1, embed_dim).
                - gene_labels (torch.Tensor): Labels with CLS dummy label prepended,
                  shape (batch_size, seq_len+1).
        """
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
        if self.use_pos_emb == 'absolute':
            pos = self.pos_embed(torch.arange(x.shape[1], device=x.device))
            x = x + pos
        elif self.use_pos_emb is not None:
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
        return_mean_non_padding=False,
    ):
        """
        Forward pass of the GP Transformer Encoder.
        Args:
            x (torch.Tensor): Input embeddings of shape
                (batch_size, seq_len, embed_dim).
            gene_labels (torch.Tensor): Gene labels of shape (batch_size,
                seq_len).
            masking (bool): Whether to apply random masking for MLM.
            attn_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len+1).
            return_attention (bool): Whether to return attention weights.
            return_gene_embeddings (bool): Whether to return gene embeddings.
            return_mean_non_padding (bool):
                Whether to return mean of non-padding gene embeddings
                (not used in main model, only for baselines)
        Returns:
            dict: A dictionary containing:
                - 'cls': CLS token embeddings of shape (batch_size, embed_dim).
                - 'logits_lm': Logits for masked language modeling of
                  shape (batch_size, seq_len+1, vocab_size).
                - 'gene_labels': Updated gene labels with -100 for
                  unmasked positions.
                - 'attention' (optional): Attention weights if
                  return_attention is True.
                - 'gene_embeddings' (optional): Gene embeddings if
                  return_gene_embeddings is True.
                - 'mean_non_padding' (optional): Mean of non-padding gene
                  embeddings if return_mean_non_padding is True.
                  (not used in main model, only for baselines)
        """
        # Optionally reduce dimensions
        if hasattr(self, 'dim_red') and (self.output_dim != self.embed_dim):
            x = self.dim_red(x)

        # Random masking:
        if masking:
            x, gene_labels = self.random_gene_masking(x, gene_labels)

        # Prepare tokens for transformer
        x, gene_labels = self.prepare_tokens(x, gene_labels)

        for blk in self.blocks:
            x, attn = blk(
                x,
                attn_mask=attn_mask,
                return_attention=return_attention,
            )

        if self.use_l2_norm:
            x = F.normalize(x, p=2, dim=-1)  # following scimilarity
        else:
            x = self.norm(x)

        token = x[:, 0]  # equivalent to x[:, 0, :] = return <GP> token

        # instead of token, take mean of all gene embeddings
        # token = x[:, 1:, :].mean(dim=1)
        if return_mean_non_padding:
            mask_non_padding = attn_mask[:, 1:]
            x = x[:, 1:, :]
            x = x * mask_non_padding.unsqueeze(-1)
            token = x.sum(dim=1) / mask_non_padding.sum(dim=1).unsqueeze(-1)

            # set to all 0 if no GP genes --> avoids nan values
            token[mask_non_padding.sum(dim=1) == 0] = 0

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


if __name__ == '__main__':
    print('Testing the model')
    model = gpTransformerEncoder(
        n_gp_tokens=5,
        depth=1,
        mlm_masking_prob=0.4,
        embed_dim=32,
        seq_len=16,
        num_heads=8,
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
