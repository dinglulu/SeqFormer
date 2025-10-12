import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtune
import torchtune.models.llama3_1
from typing import Optional, List, Tuple, Dict


class Model_v2(nn.Module):
    def __init__(
        self,
        base_vocab_size=7,   # padding(0),A(1),C(2),G(3),T(4),-(5), N(6)
        # ref_vocab_size=5,  # SOS(0), A(1), C(2), G(3), T(4)
        qual_vocab_size=41,  # "-" 质量值为 1, padding 位置质量值为 0, Illumina 碱基质量值最大为 40, ONT 碱基质量值最大为 94 左右(需要从数据集中获取最大质量值).
        dim=512,
        output_dim=5,
        num_heads=8,
        max_seq_len=1024,
        # decode_len=300,
        num_layers=8,
        norm_eps=1e-05,
        disable_qual=False,
    ):
        super().__init__()
        self.base_vocab_size = base_vocab_size
        # self.ref_vocab_size = ref_vocab_size
        self.qual_vocab_size = qual_vocab_size
        self.dim = dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        # self.decode_len = decode_len
        self.num_layers = num_layers
        self.cache_enabled = False
        self.disable_qual = disable_qual

        self.base_embedder = torch.nn.Sequential(
            nn.Embedding(num_embeddings=base_vocab_size, embedding_dim=dim, padding_idx=0),
            torchtune.modules.RMSNorm(dim, eps=norm_eps),
        )

        if not self.disable_qual:
            self.qual_embedder = torch.nn.Sequential(
                nn.Embedding(num_embeddings=qual_vocab_size, embedding_dim=dim, padding_idx=0),
                torchtune.modules.RMSNorm(dim, eps=norm_eps),
            )
            self.fusion_layer = nn.Linear(2 * dim, dim)
        else:
            self.qual_embedder = None
            self.fusion_layer = None

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

    def inference(self, seqs, quals, lens):
        device = seqs.device

        # Base & Quality embedding
        if not self.disable_qual:
            base_embeds = self.base_embedder(seqs)  # B x N x T x C
            qual_embeds = self.qual_embedder(quals)  # B x N x T x C
            x = self.fusion_layer(
                torch.cat([base_embeds, qual_embeds], dim=-1)
            )  # B x N x T x C
        else:
            base_embeds = self.base_embedder(seqs)  # B x N x T x C
            x = base_embeds

        # Merge features over aligned sequences
        x = torch.sum(x.transpose(1, 2), dim=2)  # B x T x N x C => B x T x C

        B = x.size(0)
        T = x.size(1)
        mask = torch.zeros([B, T, T], device=device, dtype=torch.bool)
        for b in range(B):
            mask[b, :lens[b], :lens[b]] = True

        for layer in self.layers:
            x = layer(x, mask=mask)

        # Output logits
        probs = torch.nn.functional.softmax(
            self.output_layer(self.final_norm(x)).float(),
            dim=-1,
        )
        _, predictions = torch.max(probs, dim=-1)
        predictions = predictions + 1

        ret_seqs = []
        ret_probs = []
        for b in range(B):
            ret_seqs.append(
                ''.join(['0ACGT-'[ch] for ch in predictions[b, :lens[b]].cpu().numpy()]).replace('-', '')
            )
            ret_probs.append(
                probs[b, :lens[b], :].cpu().numpy()
            )

        return ret_seqs, ret_probs

    def forward(self, seqs, quals, refs, lens, return_logits=False):
        """
        :param seqs: B x N x T
        :param quals: B x N x T
        :param refs: B x T
        :param lens: B
        :param return_logits: bool
        :return:
        """
        device = seqs.device

        # Base & Quality embedding
        if not self.disable_qual:
            base_embeds = self.base_embedder(seqs)  # B x N x T x C
            qual_embeds = self.qual_embedder(quals)  # B x N x T x C
            x = self.fusion_layer(
                torch.cat([base_embeds, qual_embeds], dim=-1)
            )  # B x N x T x C
        else:
            base_embeds = self.base_embedder(seqs)
            x = base_embeds

        # Merge features over aligned sequences
        x = torch.sum(x.transpose(1, 2), dim=2)  # B x T x N x C => B x T x C

        # ignore idx: -1
        targets = refs - 1

        B = x.size(0)
        T = x.size(1)
        mask = torch.zeros([B, T, T], device=device, dtype=torch.bool)
        for b in range(B):
            mask[b, :lens[b], :lens[b]] = True

        for layer in self.layers:
            x = layer(x, mask=mask)

        # Output logits
        logits = self.output_layer(self.final_norm(x)).float()

        # Compute cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.contiguous().view(-1, self.output_dim),
            targets.view(-1),
            ignore_index=-1,
            reduction="mean",
        )

        if return_logits:
            return loss, logits

        return loss

