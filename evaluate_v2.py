import torch
import torch.cuda
import torch.backends.cudnn
import os
import re
import random
import numpy as np
import collections
import parasail
from Bio import SeqIO
from spoa import poa
from tqdm import tqdm
from scheduler import linear_warmup_cosine_decay
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from model_v2 import Model_v2
from p_tqdm import p_map
from data_preprocess import encode_base_seq, qual_convertion


class ConsensusPredictor(object):
    def load_model(
        self,
        disable_qual=False,
    ):
        model = Model_v2(
            base_vocab_size=7,
            qual_vocab_size=self.qual_vocab_size,
            dim=512,
            output_dim=5,
            num_heads=8,
            max_seq_len=1024,
            num_layers=8,
            disable_qual=disable_qual,
        )
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(self.model_path)
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        model = model.cuda(self.gpu)
        model.eval()
        return model

    def __init__(
        self,
        model_path,
        qual_vocab_size,
        context_len,
        max_depth=100,
        batch_size=1,
        disable_qual=False,
        gpu=0,
    ):
        """
        :param model_path: path of saved model checkpoint.
        :param qual_vocab_size: vocabulary size of quality scores.
        :param context_len: maximum length of multiple-sequence-alignment.
        :param max_depth: maximum depth of multiple sequence-alignment (default: 100).
        :param batch_size: batch size (default: 1).
        :param gpu: gpu device id (default: 0).
        """
        super().__init__()
        self.model_path = model_path
        self.qual_vocab_size = qual_vocab_size
        self.context_len = context_len
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.gpu = gpu
        self.disable_qual = disable_qual
        self.model = self.load_model(disable_qual)
        print('Load model done.')

    def inference(self, seqs, quals, lens):
        """
        :param seqs: B x N x T
        :param quals: B x N x T
        :param lens: B
        :return:
        """
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True):
                ret_seqs, ret_probs = self.model.inference(
                    seqs.cuda(self.gpu, non_blocking=True),
                    quals.cuda(self.gpu, non_blocking=True),
                    lens.cuda(self.gpu, non_blocking=True),
                )
        return ret_seqs, ret_probs

    def predict(self, data, bsalign_msa_dict):
        assert len(data) <= self.batch_size
        clusters = []
        poa_consensus_db = []
        batched_seqs = torch.zeros((self.batch_size, self.max_depth, self.context_len), dtype=torch.long)
        batched_quals = torch.zeros((self.batch_size, self.max_depth, self.context_len), dtype=torch.long)
        batched_lens = torch.zeros(self.batch_size, dtype=torch.long)

        for batch_idx, (cluster, item) in enumerate(data.items()):
            clusters.append(cluster)
            seqs = item['seqs']
            quals = item['quals']
            if len(seqs) > self.max_depth:
                raise RuntimeError(f'Number of sequences in {cluster} exceeds maximum depth.')

            msa = bsalign_msa_dict[cluster]
            poa_consensus_db.append("")
            batched_lens[batch_idx] = len(msa[0])
            for i in range(len(msa)):
                ali_seq = encode_base_seq(msa[i]).astype(np.int64)
                ali_qual = np.ones(len(ali_seq), dtype=np.int64)
                ali_qual[np.where(ali_seq != 5)[0]] = qual_convertion(quals[i]).astype(np.int64)
                batched_seqs[batch_idx, i, :len(ali_seq)] = torch.from_numpy(ali_seq)
                batched_quals[batch_idx, i, :len(ali_qual)] = torch.from_numpy(ali_qual)

        new_consensus_db, consensus_probs_db = self.inference(
            seqs=batched_seqs, quals=batched_quals, lens=batched_lens
        )
        ret = {}
        for cluster, poa_consensus, new_consensus, consensus_probs in zip(clusters, poa_consensus_db, new_consensus_db, consensus_probs_db):
            ret[cluster] = {
                'poa_consensus': poa_consensus,
                'new_consensus': new_consensus,
                'consensus_probs': consensus_probs.tolist(),
            }
        return ret
