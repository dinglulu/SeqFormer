import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtune
import torchtune.models.llama3_1
from typing import Optional, List, Tuple, Dict


# class TransformerLayer(nn.Module):
#     def __init__(self, dim, num_heads=8, max_seq_len=1024, ):
#         super().__init__()
#         head_dim = dim // num_heads
#
#         # B x N x T x C => (B * N) x T x C (attn for different tokens in the same seq)
#         self.attn1 = torchtune.modules.MultiHeadAttention(
#             embed_dim=dim,
#             num_heads=num_heads,
#             num_kv_heads=num_heads,
#             head_dim=head_dim,
#             q_proj=nn.Linear(dim, dim, bias=False),
#             k_proj=nn.Linear(dim, dim, bias=False),
#             v_proj=nn.Linear(dim, dim, bias=False),
#             output_proj=nn.Linear(dim, dim, bias=False),
#             pos_embeddings=torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE(
#                 head_dim, max_seq_len=max_seq_len, base=10000, scale_factor=8,
#             ),
#             q_norm=None,
#             k_norm=None,
#             max_seq_len=max_seq_len,
#             is_causal=False,
#             attn_dropout=0.0,
#         )
#
#         # B x N x T x C => B x T x N x C => (B * T) x N x C (attn for aligned tokens in the different seqs)
#         self.attn2 = torchtune.modules.MultiHeadAttention(
#             embed_dim=dim,
#             num_heads=num_heads,
#             num_kv_heads=num_heads,
#             head_dim=head_dim,
#             q_proj=nn.Linear(dim, dim, bias=False),
#             k_proj=nn.Linear(dim, dim, bias=False),
#             v_proj=nn.Linear(dim, dim, bias=False),
#             output_proj=nn.Linear(dim, dim, bias=False),
#             pos_embeddings=None,
#             q_norm=None,
#             k_norm=None,
#             max_seq_len=max_seq_len,
#             is_causal=False,
#             attn_dropout=0.0,
#         )
#
#         self.mlp = torchtune.modules.FeedForward(
#             gate_proj=nn.Linear(dim, 2048),
#             down_proj=nn.Linear(2048, dim),
#             up_proj=nn.Linear(dim, 2048),
#         )
#
#         self.attn_norm1 = torchtune.modules.RMSNorm(dim, eps=norm_eps)
#         self.attn_norm2 = torchtune.modules.RMSNorm(dim, eps=norm_eps)
#         self.mlp_norm = torchtune.modules.RMSNorm(dim, eps=norm_eps)
#
#     def forward(self, x, mask):
#         """
#         :param x: B x N x T x C (B: batch_size, N: num_aligned_sequences, T: num_tokens, C: num_channels)
#         :param mask: (B * N) x T x T
#         :return:
#         """
#         B = x.shape[0]
#         N = x.shape[1]
#         T = x.shape[2]
#         C = x.shape[3]
#
#         x = x.view(B * N, T, C)
#         h = self.attn_norm1(x)
#         attn_out = self.attn1(h, h, mask=mask, input_pos=None)
#         x = attn_out + x
#
#         x = x.view(B, N, T, C).transpose(1, 2).reshape(B * T, N, C)
#         h = self.attn_norm2(x)
#         attn_out = self.attn2(h, h, mask=None, input_pos=None)
#         x = attn_out + x
#
#         x = x.view(B, T, N, C).transpose()


def build_prefix_lm_attention_mask(x_lengths: torch.Tensor, T: int, L: int):
    """
    Construct a prefix-LM attention mask for Transformer.

    Args:
        x_lengths (torch.Tensor): Tensor of shape [B] indicating the valid lengths of prompt `x`.
        T (int): Length of the prefix/prompt `x`.
        L (int): Length of the target/completion `y`.

    Returns:
        torch.BoolTensor: Boolean attention mask of shape [B, T+L, T+L], where True means attention is allowed.
    """
    B = x_lengths.size(0)
    device = x_lengths.device
    total_len = T + L

    # Step 1: Base mask [T+L, T+L] for prefix-LM
    base_mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)

    # Step 2: Prefix tokens attend to all prefix tokens (T x T block)
    base_mask[:T, :T] = True

    # Step 3: Target tokens attend to all prefix tokens (L x T block)
    base_mask[T:, :T] = True

    # Step 4: Target tokens attend to themselves causally (L x L lower triangle block)
    base_mask[T:, T:] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))

    # Step 5: Broadcast to batch [B, T+L, T+L]
    base_mask = base_mask.unsqueeze(0).expand(B, -1, -1)  # (B, T+L, T+L)

    # Step 6: Construct padding mask for x (prompt)
    # mask_x: shape [B, T], True for valid tokens
    mask_x = torch.arange(T, device=device).unsqueeze(0) < x_lengths.unsqueeze(1)  # (B, T)

    # pad_mask: shape [B, T+L], True for valid tokens (x + y), assume y has no padding
    pad_mask = torch.cat([mask_x, torch.ones(B, L, dtype=torch.bool, device=device)], dim=1)  # (B, T+L)

    # Step 7: Apply pad_mask to base_mask
    # Prevent attention to padded positions in x: if j is padded, mask[:, :, j] = False
    final_mask = base_mask & pad_mask.unsqueeze(1)  # (B, T+L, T+L)

    return final_mask  # shape (B, T+L, T+L)


class Model(nn.Module):
    def __init__(
        self,
        base_vocab_size=7,  # padding(0), A(1),C(2),G(3),T(4),-(5), N(6)
        ref_vocab_size=5,   # SOS(0), A(1), C(2), G(3), T(4)
        qual_vocab_size=41,  # "-" 质量值为 1, padding 位置质量值为 0, Illumina 碱基质量值最大为 40, ONT 碱基质量值最大为 94 左右(需要从数据集中获取最大质量值).
        dim=512,
        output_dim=4,
        num_heads=8,
        max_seq_len=1024,
        decode_len=300,
        num_layers=8,
        norm_eps=1e-05,
    ):
        super().__init__()
        self.base_vocab_size = base_vocab_size
        self.ref_vocab_size = ref_vocab_size
        self.qual_vocab_size = qual_vocab_size
        self.dim = dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.decode_len = decode_len
        self.num_layers = num_layers
        self.cache_enabled = False

        self.base_embedder = torch.nn.Sequential(
            nn.Embedding(num_embeddings=base_vocab_size, embedding_dim=dim, padding_idx=0),
            torchtune.modules.RMSNorm(dim, eps=norm_eps),
        )
        self.qual_embedder = torch.nn.Sequential(
            nn.Embedding(num_embeddings=qual_vocab_size, embedding_dim=dim, padding_idx=0),
            torchtune.modules.RMSNorm(dim, eps=norm_eps),
        )
        self.ref_embedder = torch.nn.Sequential(
            nn.Embedding(num_embeddings=ref_vocab_size, embedding_dim=dim)
        )
        self.fusion_layer = nn.Linear(2 * dim, dim)

        head_dim = dim // num_heads
        self.layers = nn.ModuleList([
            torchtune.modules.TransformerSelfAttentionLayer(
                attn=torchtune.modules.MultiHeadAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                    head_dim=head_dim,
                    q_proj=nn.Linear(dim, dim, bias=False),
                    k_proj=nn.Linear(dim, dim, bias=False),
                    v_proj=nn.Linear(dim, dim, bias=False),
                    output_proj=nn.Linear(dim, dim, bias=False),
                    pos_embeddings=torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE(
                        head_dim, max_seq_len=max_seq_len, base=10000, scale_factor=8,
                    ),
                    q_norm=None,
                    k_norm=None,
                    max_seq_len=max_seq_len,
                    is_causal=False,
                    attn_dropout=0.0,
                ),
                mlp=torchtune.modules.FeedForward(
                    gate_proj=nn.Linear(dim, 2048),
                    down_proj=nn.Linear(2048, dim),
                    up_proj=nn.Linear(dim, 2048),
                ),
                sa_norm=torchtune.modules.RMSNorm(dim, eps=norm_eps),
                mlp_norm=torchtune.modules.RMSNorm(dim, eps=norm_eps),
                sa_scale=None,
                mlp_scale=None,
            ) for _ in range(num_layers)
        ])
        self.final_norm = torchtune.modules.RMSNorm(dim, eps=norm_eps)
        self.output_layer = nn.Linear(dim, output_dim)

    def setup_cache(self, batch_size, dtype=torch.float16):
        if not self.cache_enabled:
            for layer in self.layers:
                layer.setup_caches(
                    batch_size=batch_size,
                    dtype=dtype,
                    encoder_max_seq_len=self.max_seq_len,
                    decoder_max_seq_len=self.max_seq_len,
                )
            self.cache_enabled = True

    def reset_cache(self, ):
        if self.cache_enabled:
            for layer in self.layers:
                layer.reset_cache()

    def forward(self, seqs, quals, refs, lens, return_logits=False):
        """
        :param seqs: B x N x T
        :param quals: B x N x T
        :param refs: B x L
        :param lens: B
        :param return_logits: bool
        :return:
        """
        device = seqs.device

        # Base & Quality embedding
        base_embeds = self.base_embedder(seqs)  # B x N x T x C
        qual_embeds = self.qual_embedder(quals)  # B x N x T x C
        x = self.fusion_layer(
            torch.cat([base_embeds, qual_embeds], dim=-1)
        )  # B x N x T x C

        # Merge features over aligned sequences
        x = torch.sum(x.transpose(1, 2), dim=2)  # B x T x N x C => B x T x C

        pad_refs = torch.cat([
            refs.new_zeros((refs.size(0), 1), device=device, dtype=refs.dtype),
            refs
        ], dim=1)
        y = pad_refs[:, :-1]
        targets = pad_refs[:, 1:] - 1

        y = self.ref_embedder(y)  # B x L x C

        # Create attention mask
        T = x.size(1)
        L = y.size(1)
        mask = build_prefix_lm_attention_mask(lens, T, L)

        # Transformer forward
        z = torch.cat([x, y], dim=1)
        for layer in self.layers:
            z = layer(z, mask=mask)

        # Output logits
        logits = self.output_layer(self.final_norm(z)).float()[:, T:, :]

        # Compute cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.contiguous().view(-1, self.output_dim),
            targets.view(-1),
            reduction="mean",
        )

        if return_logits:
            return loss, logits
        return loss
