"""
将第一个版本的数据集(Torch张量, 自带多序列比对) 转为 第二个版本的数据集(HDF5, 仅保存序列和质量值, 不做比对)
"""

import os
import gzip
import re
import torch
import torch.nn as nn
import collections
import numpy as np
import parasail
import pandas as pd
import collections
from tqdm import tqdm
from spoa import poa
from argparse import ArgumentParser
import h5py


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--depth', type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    out_path = os.path.join(args.output, 'dataset.hdf5')
    out_f = h5py.File(out_path, 'w')

    data = torch.load(os.path.join(args.input, 'dataset.pth.tar'))
    print('Loaded dataset finish.')

    seqs = data['seqs']
    quals = data['quals']
    refs = data['refs']
    lens = data['lens']

    for chunk_idx, (seq_dat, qual_dat, len_dat, ref_dat) in tqdm(enumerate(zip(seqs, quals, lens, refs))):
        ref_str = ''.join(['0ACGT-N'[ch] for ch in ref_dat])
        curr_len = len_dat.item()
        _seq_dat = seq_dat[:, :curr_len].to(torch.long)
        _qual_dat = qual_dat[:, :curr_len].to(torch.long)
        curr_seqs = []
        curr_quals = []
        for _s, _q in zip(_seq_dat, _qual_dat):  # type: torch.Tensor, torch.Tensor
            indices = torch.where(_s != 5)[0]
            curr_seqs.append(
                ''.join(['0ACGT-N'[ch] for ch in _s[indices]])
            )
            curr_quals.append(
                (_q[indices].numpy() + 33).astype(np.uint8).tobytes().decode('ascii')
            )
        grp = out_f.create_group(str(chunk_idx))
        grp.create_dataset('refs', data=[ref_str])
        grp.create_dataset('seqs', data=curr_seqs)
        grp.create_dataset('quals', data=curr_quals)
    out_f.close()
