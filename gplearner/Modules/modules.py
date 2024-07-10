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

from ..Utils import (
    drop_path,
    mlm_mask_generator,
    trunc_normal_,
)


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

    def forward(self, x, attn_mask, return_attention):
        # for compatibility with previous versions
        # if no use_flash attribute, set to false
        if not hasattr(self, 'use_flash'):
            self.use_flash = False

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
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                # from https://discuss.pytorch.org/t/flash-attention/174955/14
                attn_out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    # pytorch flash attention does not support mask
                    scale=self.scale,
                    dropout_p=0.0,
                )
                # if scale is None, default is 1/sqrt(dim)

            x = attn_out.transpose(1, 2).reshape(B, N, C)

        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

            # apply attention mask for padding tokens
            # Mask rows:
            attn = attn * attn_mask.unsqueeze(1).unsqueeze(
                -1
            )  # unsqueeze to add head dimension
            # Mask columns:
            attn = attn * attn_mask.unsqueeze(1).unsqueeze(1)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

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
            self.norm1(x), attn_mask=attn_mask, return_attention=return_attention
        )  # attn is None when using flash attention
        # y = self.attn(self.norm1(x), attn_mask=attn_mask)

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class gpTransformerEncoder(nn.Module):
    """GP Transformer main block"""

    def __init__(
        self,
        n_gp_tokens,
        depth,  # number of blocks
        mlm_masking_prob,
        embed_dim=256,
        num_heads=1,
        mlp_ratio=0.5,  # factor of how much MLP reduces layer size
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,  # dropout on position embedding
        attn_drop_rate=0.0,  # passed to attention module (attn_drop)
        drop_path_rate=0.0,  # no effect if only 1 block
        norm_layer=nn.LayerNorm,
        use_pos_emb=True,
        vocab_size=None,
        use_flash=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_pos_emb = use_pos_emb

        self.pos_drop = nn.Dropout(p=drop_rate)

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

        self.pos_embed = PositionalEncoding(
            d_model=embed_dim, dropout=drop_rate, max_len=2048
        )
        # self.pos_embed = nn.Embedding(2048, embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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

    def prepare_tokens(self, x, gene_labels):
        B = x.shape[0]  # batch size

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
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
        if self.use_pos_emb:
            x = self.pos_embed(x)
        #     position_ids = torch.arange(x.shape[1], device=x.device)
        #     pos_emb = self.pos_embed(position_ids)
        #     print('pos_emb shape', pos_emb.shape)
        #     print('x shape', x.shape)
        #     x = x + pos_emb

        #     print('x shape after pos emb', x.shape)

        return self.pos_drop(x), gene_labels

    def forward(
        self,
        x,
        gene_labels,
        inference,
        attn_mask,
        return_attention,
        return_gene_embeddings=False,
        num_virtual_tokens=0,
    ):
        # Random masking:
        if inference is False:
            x, gene_labels = self.random_gene_masking(
                x, gene_labels, unmask_last_n=num_virtual_tokens
            )

        # Prepare tokens for transformer
        x, gene_labels = self.prepare_tokens(x, gene_labels)

        # # Optionally move virtual tokens to start of the sequence
        # if num_virtual_tokens > 0:
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
            # Now returns full attention matrix not just CLS
            # for attributions, use gradcam
            # could implement method for cls attention scores as well
            # print('Attention shape', attn.shape)
            # (batch, heads, 1 + tokens, 1 + tokens)
            # print('<cls>', attn[:, :, 0, :].shape)
            output['attention'] = attn

        if return_gene_embeddings:
            output['gene_embeddings'] = (
                x[:, 1:-num_virtual_tokens, :]
                if num_virtual_tokens > 0
                else x[:, 1:, :]
            )

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
        n_gp_tokens=5, depth=1, mlm_masking_prob=0.4, embed_dim=32
    )
    x = torch.randn(1, 5, 32)
    gene_labels = torch.randint(0, 10, (1, 5))
    out = model(x, gene_labels, inference=False, attn_mask=None, return_attention=False)
    print(out['gene_labels'])
